/*
 * sporf.cuh
 *
 * Notes:
 * 
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuml/ensemble/sporf.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/accuracy.cuh>
#include <raft/stats/regression_metrics.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <decisiontree/decisiontree.cuh>
#include <decisiontree/sporfdecisiontree.cuh>
#include <decisiontree/treelite_util.h>
#include <algorithm>
#include <chrono>
#include <deque>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num()  0
#define omp_get_max_threads() 1
#endif

#include <map>

namespace ML {

template <typename T, typename L>
__global__ void finalize_rf_predictions_kernel(const T* d_probs,
                                               int n_rows,
                                               int num_outputs,
                                               int rf_type,
                                               L* d_predictions)
{
  int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n_rows) { return; }
  if (num_outputs <= 0) { return; }

  if (rf_type == RF_type::CLASSIFICATION) {
    int row_base = row * num_outputs;
    int best_k   = 0;
    T best_prob  = d_probs[row_base];
    for (int k = 1; k < num_outputs; k++) {
      T p = d_probs[row_base + k];
      if (p > best_prob) {
        best_prob = p;
        best_k    = k;
      }
    }
    d_predictions[row] = static_cast<L>(best_k);
  } else {
    d_predictions[row] = static_cast<L>(d_probs[row * num_outputs]);
  }
}

template <typename T>
__global__ void add_prediction_buffers_kernel(T* d_dst, const T* d_src, std::size_t n)
{
  auto i = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) { return; }
  d_dst[i] += d_src[i];
}

  /*
  * Extended implementation of cuml RandomForest class.
  *
  * - random forest parameters include additional values for SPORF
  */
template <class T, class L>
class SPORF {
 protected:
  SPORF_params rf_params; // structure containing RF hyperparameters
  int rf_type;            // 0 for classification 1 for regression

  void get_row_sample(int tree_id,
                      int n_rows,
                      rmm::device_uvector<int>* selected_rows,
                      const cudaStream_t stream)
  {
    raft::common::nvtx::range fun_scope("bootstrapping row IDs @SPORF.cuh");

    // Hash these together so they are uncorrelated
    auto rs = DT::fnv1a32_basis;
    rs      = DT::fnv1a32(rs, rf_params.seed);
    rs      = DT::fnv1a32(rs, tree_id);
    raft::random::Rng rng(rs, raft::random::GenPhilox);
    if (rf_params.bootstrap) {
      // Use bootstrapped sample set
      rng.uniformInt<int>(selected_rows->data(), selected_rows->size(), 0, n_rows, stream);

    } else {
      // Use all the samples from the dataset
      thrust::sequence(thrust::cuda::par.on(stream), selected_rows->begin(), selected_rows->end());
    }
  }

  void error_checking(const T* input, L* predictions, int n_rows, int n_cols, bool predict) const
  {
    if (predict) {
      ASSERT(predictions != nullptr, "Error! User has not allocated memory for predictions.");
    }
    ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
    ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

    bool input_is_dev_ptr = DT::is_dev_ptr(input);
    bool preds_is_dev_ptr = DT::is_dev_ptr(predictions);

    if (!input_is_dev_ptr || (input_is_dev_ptr != preds_is_dev_ptr)) {
      ASSERT(false,
             "RF Error: Expected both input and labels/predictions to be GPU "
             "pointers");
    }
  }

 public:
  /**
   * @brief Construct RandomForest object.
   * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
   * @param[in] cfg_rf_type: Task type: 0 for classification, 1 for regression
   */
  SPORF(SPORF_params cfg_rf_params, int cfg_rf_type = RF_type::CLASSIFICATION)
    : rf_params(cfg_rf_params), rf_type(cfg_rf_type) {};

  /**
   * @brief Build (i.e., fit, train) random forest for input data.
   * @param[in] user_handle: raft::handle_t
   * @param[in] input: train data (n_rows samples, n_cols features) in column major format,
   *   excluding labels. Device pointer.
   * @param[in] n_rows: number of training data samples.
   * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
   * @param[in] labels: 1D array of target predictions/labels. Device Pointer.
            For classification task, only labels of type int are supported.
              Assumption: labels were preprocessed to map to ascending numbers from 0;
              needed for current gini impl in decision tree
            For regression task, the labels (predictions) can be float or double data type.
  * @param[in] n_unique_labels: (meaningful only for classification) #unique label values (known
  during preprocessing)
  * @param[in] forest: CPU point to RandomForestMetaData struct.
  */
  void fit(const raft::handle_t& user_handle,
           const T* input,
           int n_rows,
           int n_cols,
           L* labels,
           int n_unique_labels,
           SPORFMetaData<T, L>*& forest)
  {
    raft::common::nvtx::range fun_scope("SPORF::fit @sporf.cuh");
    using clock_t = std::chrono::steady_clock;
    auto to_ms = [](clock_t::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };
    auto t_fit_wall_start = clock_t::now();
    this->error_checking(input, labels, n_rows, n_cols, false);
    const raft::handle_t& handle = user_handle;
    double t_workspace_setup = 0.0;
    double t_row_sampling = 0.0;
    double t_tree_fit_envelope = 0.0;
    double t_final_sync = 0.0;
    int n_sampled_rows           = 0;
    if (this->rf_params.bootstrap) {
      n_sampled_rows = std::round(this->rf_params.max_samples * n_rows);
    } else {
      if (this->rf_params.max_samples != 1.0) {
        CUML_LOG_WARN(
          "If bootstrap sampling is disabled, max_samples value is ignored and "
          "whole dataset is used for building each tree");
        this->rf_params.max_samples = 1.0;
      }
      n_sampled_rows = n_rows;
    }
    int n_streams = this->rf_params.n_streams;
    ASSERT(static_cast<std::size_t>(n_streams) <= handle.get_stream_pool_size(),
           "rf_params.n_streams (=%d) should be <= raft::handle_t.n_streams (=%lu)",
           n_streams,
           handle.get_stream_pool_size());

    // n_streams should not be less than n_trees
    if (this->rf_params.n_trees < n_streams) n_streams = this->rf_params.n_trees;

    DT::SPORFDeviceBatchingPolicy device_batching_policy;
    {
      cudaDeviceProp prop;
      RAFT_CUDA_TRY(cudaGetDeviceProperties(&prop, handle.get_device()));
      device_batching_policy.num_sms            = prop.multiProcessorCount;
      device_batching_policy.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
      device_batching_policy.max_resident_threads =
        static_cast<std::size_t>(device_batching_policy.num_sms) *
        static_cast<std::size_t>(device_batching_policy.max_threads_per_sm);
      device_batching_policy.target_rows_per_batch =
        std::max<std::size_t>(1, device_batching_policy.max_resident_threads);
      device_batching_policy.target_blocks_per_batch =
        static_cast<std::size_t>(device_batching_policy.num_sms) * 4;
    }

    if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) {
      std::cout << "SPORF::fit device batching policy: num_sms=" << device_batching_policy.num_sms
                << " max_threads_per_sm=" << device_batching_policy.max_threads_per_sm
                << " max_resident_threads=" << device_batching_policy.max_resident_threads
                << " target_rows_per_batch=" << device_batching_policy.target_rows_per_batch
                << " target_blocks_per_batch=" << device_batching_policy.target_blocks_per_batch
                << std::endl;
    }
    // Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
    // selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device
    // ptr.
    // Use a deque instead of vector because it can be used on objects with a deleted copy
    // constructor
    auto t_workspace_setup_start = clock_t::now();
    std::deque<rmm::device_uvector<int>> selected_rows;
    for (int i = 0; i < n_streams; i++) {
      selected_rows.emplace_back(n_sampled_rows, handle.get_stream_from_stream_pool(i));
    }

    using TrainingWs = DT::SPORFTrainingProjectionWorkspace<T, L, int>;
    std::deque<TrainingWs> training_workspaces;
    for (int i = 0; i < n_streams; i++) {
      training_workspaces.emplace_back(
        static_cast<size_t>(n_rows),
        static_cast<size_t>(this->rf_params.tree_params.max_batch_size),
        handle.get_stream_from_stream_pool(i));
    }
    t_workspace_setup += to_ms(clock_t::now() - t_workspace_setup_start);


    #pragma omp parallel for num_threads(n_streams)
    for (int i = 0; i < this->rf_params.n_trees; i++) {
      int stream_id = omp_get_thread_num();
      auto s        = handle.get_stream_from_stream_pool(stream_id);
      RAFT_CUDA_TRY(cudaSetDevice(handle.get_device()));

      auto t_row_sampling_start = clock_t::now();
      this->get_row_sample(i, n_rows, &selected_rows[stream_id], s);
      auto row_sampling_ms = to_ms(clock_t::now() - t_row_sampling_start);
#ifdef _OPENMP
#pragma omp atomic
#endif
      t_row_sampling += row_sampling_ms;

      /* Build individual tree in the forest.
        - input is a pointer to orig data that have n_cols features and n_rows rows.
        - n_sampled_rows: # rows sampled for tree's bootstrap sample.
        - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements)
          used to build the bootstrapped sample.
          Expectation: Each tree node will contain (a) # n_sampled_rows and
          (b) a pointer to a list of row numbers w.r.t original data.
      */

      auto t_tree_fit_start = clock_t::now();
      forest->trees[i] = DT::SPORFDecisionTree::fit(handle,
                                                    s,
                                                    input,
                                                    n_cols,
                                                    n_rows,
                                                    labels,
                                                    &selected_rows[stream_id],
                                                    n_unique_labels,
                                                    this->rf_params.tree_params,
                                                    this->rf_params.seed,
                                                    i,
                                                    training_workspaces[stream_id],
                                                    device_batching_policy);
      auto tree_fit_ms = to_ms(clock_t::now() - t_tree_fit_start);
#ifdef _OPENMP
#pragma omp atomic
#endif
      t_tree_fit_envelope += tree_fit_ms;
    }
    // Cleanup
    auto t_final_sync_start = clock_t::now();
    handle.sync_stream_pool();
    handle.sync_stream();
    t_final_sync += to_ms(clock_t::now() - t_final_sync_start);
    auto fit_wall_ms = to_ms(clock_t::now() - t_fit_wall_start);

    std::size_t workspace_device_bytes_total = 0;
    std::size_t selected_rows_device_bytes_total = 0;
    for (int i = 0; i < n_streams; ++i) {
      auto& ws = training_workspaces[i];
      std::size_t workspace_device_bytes = 0;
      workspace_device_bytes += ws.d_workspace.size() * sizeof(char);
      workspace_device_bytes += ws.d_projection_matrices_storage.size() *
                                sizeof(DT::ProjectionMatrix<T, int>);
      workspace_device_bytes += ws.d_tree_projection_vectors_storage.size() *
                                sizeof(DT::OffsetProjectionMatrix<int>);
      workspace_device_bytes += ws.d_tree_projection_max_node_idx_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_tree_projection_indptr_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_tree_projection_indices_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_tree_projection_coeffs_storage.size() * sizeof(T);
      workspace_device_bytes += ws.d_tree_projection_winning_nnz_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_tree_projection_winning_offsets_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_generation_keep_mask_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_generation_dense_values_storage.size() * sizeof(T);
      workspace_device_bytes += ws.d_generation_indptr_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_generation_indices_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_generation_sparse_data_storage.size() * sizeof(T);
      workspace_device_bytes += ws.d_generation_nnz_counter_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_sparse_sampling_component_counts_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_sparse_sampling_component_offsets_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_sparse_sampling_candidate_indices_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_sparse_sampling_unique_indices_storage.size() * sizeof(int);
      workspace_device_bytes += ws.d_sparse_sampling_unique_counts_storage.size() * sizeof(int);
      workspace_device_bytes_total += workspace_device_bytes;
      auto selected_rows_device_bytes = selected_rows[i].size() * sizeof(int);
      selected_rows_device_bytes_total += selected_rows_device_bytes;
      auto pct = [](std::size_t used, std::size_t cap) {
        return cap > 0 ? (100.0 * static_cast<double>(used) / static_cast<double>(cap)) : 0.0;
      };

      if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) {
        std::cout << "SPORF::fit workspace capacity: stream=" << i
                  << " device_bytes=" << workspace_device_bytes
                  << " selected_rows_bytes=" << selected_rows_device_bytes
                  << " d_workspace_bytes=" << ws.d_workspace.size()
                  << " cap_work_items=" << ws.meta.projection.cap_work_items
                  << " peak_work_items=" << ws.peak_projection_work_items
                  << " slack_work_items="
                  << (static_cast<std::size_t>(ws.meta.projection.cap_work_items) >
                          ws.peak_projection_work_items
                        ? static_cast<std::size_t>(ws.meta.projection.cap_work_items) -
                            ws.peak_projection_work_items
                        : std::size_t{0})
                  << " util_work_items_pct="
                  << pct(ws.peak_projection_work_items,
                         static_cast<std::size_t>(ws.meta.projection.cap_work_items))
                  << " cap_chunks=" << ws.meta.projection.cap_chunks
                  << " peak_projection_chunks=" << ws.peak_projection_chunks
                  << " peak_generation_chunks=" << ws.peak_generation_chunks
                  << " slack_chunks="
                  << (static_cast<std::size_t>(ws.meta.projection.cap_chunks) >
                          std::max(ws.peak_projection_chunks, ws.peak_generation_chunks)
                        ? static_cast<std::size_t>(ws.meta.projection.cap_chunks) -
                            std::max(ws.peak_projection_chunks, ws.peak_generation_chunks)
                        : std::size_t{0})
                  << " util_chunks_pct="
                  << pct(std::max(ws.peak_projection_chunks, ws.peak_generation_chunks),
                         static_cast<std::size_t>(ws.meta.projection.cap_chunks))
                  << " cap_block_tasks=" << ws.meta.projection.cap_block_tasks
                  << " peak_projection_block_tasks=" << ws.peak_projection_block_tasks
                  << " peak_generation_block_tasks=" << ws.peak_generation_block_tasks
                  << " slack_block_tasks="
                  << (static_cast<std::size_t>(ws.meta.projection.cap_block_tasks) >
                          std::max(ws.peak_projection_block_tasks, ws.peak_generation_block_tasks)
                        ? static_cast<std::size_t>(ws.meta.projection.cap_block_tasks) -
                            std::max(ws.peak_projection_block_tasks, ws.peak_generation_block_tasks)
                        : std::size_t{0})
                  << " util_block_tasks_pct="
                  << pct(std::max(ws.peak_projection_block_tasks, ws.peak_generation_block_tasks),
                         static_cast<std::size_t>(ws.meta.projection.cap_block_tasks))
                  << " cap_tree_projection_vectors=" << ws.meta.cap_tree_projection_vectors
                  << " peak_tree_projection_vectors=" << ws.peak_tree_projection_vectors
                  << " slack_tree_projection_vectors="
                  << (static_cast<std::size_t>(ws.meta.cap_tree_projection_vectors) >
                          ws.peak_tree_projection_vectors
                        ? static_cast<std::size_t>(ws.meta.cap_tree_projection_vectors) -
                            ws.peak_tree_projection_vectors
                        : std::size_t{0})
                  << " util_tree_projection_vectors_pct="
                  << pct(ws.peak_tree_projection_vectors,
                         static_cast<std::size_t>(ws.meta.cap_tree_projection_vectors))
                  << " tree_projection_indptr=" << ws.d_tree_projection_indptr_storage.size()
                  << " tree_projection_indices=" << ws.d_tree_projection_indices_storage.size()
                  << " peak_tree_projection_payload_nnz="
                  << ws.peak_tree_projection_payload_nnz
                  << " slack_tree_projection_payload_nnz="
                  << (ws.d_tree_projection_indices_storage.size() >
                          ws.peak_tree_projection_payload_nnz
                        ? ws.d_tree_projection_indices_storage.size() -
                            ws.peak_tree_projection_payload_nnz
                        : std::size_t{0})
                  << " util_tree_projection_payload_pct="
                  << pct(ws.peak_tree_projection_payload_nnz,
                         ws.d_tree_projection_indices_storage.size())
                  << " tree_projection_coeffs=" << ws.d_tree_projection_coeffs_storage.size()
                  << " tree_projection_winning_nnz="
                  << ws.d_tree_projection_winning_nnz_storage.size()
                  << " peak_tree_projection_batch_work_items="
                  << ws.peak_tree_projection_batch_work_items
                  << " slack_tree_projection_winning_nnz="
                  << (ws.d_tree_projection_winning_nnz_storage.size() >
                          ws.peak_tree_projection_batch_work_items
                        ? ws.d_tree_projection_winning_nnz_storage.size() -
                            ws.peak_tree_projection_batch_work_items
                        : std::size_t{0})
                  << " tree_projection_winning_offsets="
                  << ws.d_tree_projection_winning_offsets_storage.size()
                  << " generation_keep_mask=" << ws.d_generation_keep_mask_storage.size()
                  << " generation_dense_values=" << ws.d_generation_dense_values_storage.size()
                  << " peak_generation_dense_len=" << ws.peak_generation_dense_len
                  << " slack_generation_dense_len="
                  << (ws.d_generation_dense_values_storage.size() > ws.peak_generation_dense_len
                        ? ws.d_generation_dense_values_storage.size() - ws.peak_generation_dense_len
                        : std::size_t{0})
                  << " util_generation_dense_pct="
                  << pct(ws.peak_generation_dense_len, ws.d_generation_dense_values_storage.size())
                  << " generation_indptr=" << ws.d_generation_indptr_storage.size()
                  << " peak_generation_indptr_len=" << ws.peak_generation_indptr_len
                  << " slack_generation_indptr_len="
                  << (ws.d_generation_indptr_storage.size() > ws.peak_generation_indptr_len
                        ? ws.d_generation_indptr_storage.size() - ws.peak_generation_indptr_len
                        : std::size_t{0})
                  << " util_generation_indptr_pct="
                  << pct(ws.peak_generation_indptr_len, ws.d_generation_indptr_storage.size())
                  << " generation_indices=" << ws.d_generation_indices_storage.size()
                  << " generation_sparse_data=" << ws.d_generation_sparse_data_storage.size()
                  << " sparse_component_counts="
                  << ws.d_sparse_sampling_component_counts_storage.size()
                  << " peak_sparse_component_count="
                  << ws.peak_sparse_sampling_component_count
                  << " sparse_component_offsets="
                  << ws.d_sparse_sampling_component_offsets_storage.size()
                  << " sparse_candidate_indices="
                  << ws.d_sparse_sampling_candidate_indices_storage.size()
                  << " peak_sparse_candidate_count="
                  << ws.peak_sparse_sampling_candidate_count
                  << " sparse_unique_indices=" << ws.d_sparse_sampling_unique_indices_storage.size()
                  << " peak_sparse_unique_count=" << ws.peak_sparse_sampling_unique_count
                  << " sparse_unique_counts=" << ws.d_sparse_sampling_unique_counts_storage.size()
                  << std::endl;
      }
    }
    if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) {
      std::cout << "SPORF::fit workspace capacity total: streams=" << n_streams
                << " workspace_device_bytes=" << workspace_device_bytes_total
                << " selected_rows_device_bytes=" << selected_rows_device_bytes_total
                << " retained_device_bytes="
                << (workspace_device_bytes_total + selected_rows_device_bytes_total)
                << std::endl;

      std::cout << "SPORF::fit timings (ms): workspace_setup=" << t_workspace_setup
                << " row_sampling=" << t_row_sampling
                << " tree_fit_envelope_accum=" << t_tree_fit_envelope
                << " final_sync=" << t_final_sync
                << " wall_total=" << fit_wall_ms
                << " (n_trees=" << this->rf_params.n_trees
                << " n_streams=" << n_streams
                << ")" << std::endl;
    }
  }

  /**
   * @brief Predict target feature for input data
   * @param[in] user_handle: raft::handle_t.
   * @param[in] input: test data (n_rows samples, n_cols features) in column major format. GPU
   * pointer.
   * @param[in] n_rows: number of  data samples.
   * @param[in] n_cols: number of features (excluding target feature).
   * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
   * @param[in] verbosity: verbosity level for logging messages during execution
   */
  void predict(const raft::handle_t& user_handle,
               const T* input,
               int n_rows,
               int n_cols,
               L* predictions,
               const SPORFMetaData<T, L>* forest,
               rapids_logger::level_enum verbosity) const
  {
    ML::default_logger().set_level(verbosity);
    this->error_checking(input, predictions, n_rows, n_cols, true);
    cudaStream_t stream = user_handle.get_stream();
    bool do_timing = ML::default_logger().should_log(rapids_logger::level_enum::debug);

    cudaEvent_t ev_total_start{}, ev_total_stop{}, ev_tree_start{}, ev_tree_stop{}, ev_reduce_start{},
      ev_reduce_stop{}, ev_finalize_start{}, ev_finalize_stop{};
    if (do_timing) {
      RAFT_CUDA_TRY(cudaEventCreate(&ev_total_start));
      RAFT_CUDA_TRY(cudaEventCreate(&ev_total_stop));
      RAFT_CUDA_TRY(cudaEventCreate(&ev_tree_start));
      RAFT_CUDA_TRY(cudaEventCreate(&ev_tree_stop));
      RAFT_CUDA_TRY(cudaEventCreate(&ev_reduce_start));
      RAFT_CUDA_TRY(cudaEventCreate(&ev_reduce_stop));
      RAFT_CUDA_TRY(cudaEventCreate(&ev_finalize_start));
      RAFT_CUDA_TRY(cudaEventCreate(&ev_finalize_stop));
      RAFT_CUDA_TRY(cudaEventRecord(ev_total_start, stream));
    }

    rmm::device_uvector<T> d_prediction_buffer(n_rows * forest->trees[0]->num_outputs, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_prediction_buffer.data(), 0, sizeof(T) * d_prediction_buffer.size(), stream));

    int predict_streams = 1;
    int pool_streams    = static_cast<int>(user_handle.get_stream_pool_size());
    if (pool_streams > 0) {
      predict_streams = std::min(4, std::min(pool_streams, this->rf_params.n_trees));
    }

    std::vector<cudaStream_t> tree_streams;
    std::deque<rmm::device_uvector<T>> stream_prediction_buffers;
    using PredictWs = DT::SPORFDecisionTreeWorkspace<T, L, int>;
    std::vector<std::unique_ptr<PredictWs>> stream_workspaces;
    if (predict_streams > 1) {
      tree_streams.reserve(predict_streams);
      stream_workspaces.reserve(predict_streams);
      for (int s = 0; s < predict_streams; s++) {
        auto tree_stream = user_handle.get_stream_from_stream_pool(s);
        tree_streams.push_back(tree_stream);
        stream_prediction_buffers.emplace_back(d_prediction_buffer.size(), tree_stream);
        RAFT_CUDA_TRY(cudaMemsetAsync(stream_prediction_buffers.back().data(),
                                      0,
                                      sizeof(T) * stream_prediction_buffers.back().size(),
                                      tree_stream));
        stream_workspaces.emplace_back(
          std::make_unique<PredictWs>(static_cast<size_t>(n_rows),
                                      static_cast<size_t>(this->rf_params.tree_params.max_batch_size),
                                      tree_stream));
      }
    } else {
      stream_workspaces.emplace_back(
        std::make_unique<PredictWs>(static_cast<size_t>(n_rows),
                                    static_cast<size_t>(this->rf_params.tree_params.max_batch_size),
                                    stream));
    }

    // TODO(sporf): Predict now expects column-major GPU input for classifier path.
    // Regressor CPU predict path still packs X as row-major ('C') in sporfregressor.pyx
    // and must be updated before enabling this layout assumption for regression.

    // std::vector<T> h_input(std::size_t(n_rows) * n_cols);
    // raft::update_host(h_input.data(), input, std::size_t(n_rows) * n_cols, stream);
    // user_handle.sync_stream(stream);

    default_logger().set_pattern("%v");
    if (do_timing) { RAFT_CUDA_TRY(cudaEventRecord(ev_tree_start, stream)); }
    for (int i = 0; i < this->rf_params.n_trees; i++) {
      int sid = (predict_streams > 1) ? (i % predict_streams) : 0;
      auto out_ptr = (predict_streams > 1) ? stream_prediction_buffers[sid].data() : d_prediction_buffer.data();
      auto pred_stream = (predict_streams > 1) ? tree_streams[sid] : stream;
      auto& ws = *stream_workspaces[sid];
      DT::SPORFDecisionTree::predict(user_handle,
        *forest->trees[i],
        this->rf_params.tree_params.max_batch_size,
        input, // h_input.data(),
        n_rows,
        n_cols,
        1.0 / this->rf_params.n_trees,  // scale (accumulate unscaled; normalize later)
        out_ptr,
        forest->trees[i]->num_outputs,
        verbosity,
        ws,
        pred_stream);
    }
    if (do_timing) {
      if (predict_streams > 1) {
        user_handle.sync_stream_pool();
        RAFT_CUDA_TRY(cudaEventRecord(ev_tree_stop, stream));
      } else {
        RAFT_CUDA_TRY(cudaEventRecord(ev_tree_stop, stream));
      }
    }

    if (do_timing) { RAFT_CUDA_TRY(cudaEventRecord(ev_reduce_start, stream)); }
    if (predict_streams > 1) {
      user_handle.sync_stream_pool();
      constexpr int TPB = 256;
      auto n = d_prediction_buffer.size();
      dim3 block(TPB);
      dim3 grid((n + TPB - 1) / TPB);
      for (int s = 0; s < predict_streams; s++) {
        add_prediction_buffers_kernel<T><<<grid, block, 0, stream>>>(
          d_prediction_buffer.data(), stream_prediction_buffers[s].data(), n);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
    }
    if (do_timing) { RAFT_CUDA_TRY(cudaEventRecord(ev_reduce_stop, stream)); }

    if (do_timing) { RAFT_CUDA_TRY(cudaEventRecord(ev_finalize_start, stream)); }
    {
      constexpr int TPB = 256;
      int num_outputs = forest->trees[0]->num_outputs;
      dim3 block(TPB);
      dim3 grid((n_rows + TPB - 1) / TPB);
      finalize_rf_predictions_kernel<T, L><<<grid, block, 0, stream>>>(
        d_prediction_buffer.data(), n_rows, num_outputs, rf_type, predictions);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    if (do_timing) {
      RAFT_CUDA_TRY(cudaEventRecord(ev_finalize_stop, stream));
      RAFT_CUDA_TRY(cudaEventRecord(ev_total_stop, stream));
    }
    user_handle.sync_stream(stream);

    if (do_timing) {
      RAFT_CUDA_TRY(cudaEventSynchronize(ev_total_stop));
      float ms_total = 0.0f, ms_tree = 0.0f, ms_reduce = 0.0f, ms_finalize = 0.0f;
      RAFT_CUDA_TRY(cudaEventElapsedTime(&ms_total, ev_total_start, ev_total_stop));
      RAFT_CUDA_TRY(cudaEventElapsedTime(&ms_tree, ev_tree_start, ev_tree_stop));
      RAFT_CUDA_TRY(cudaEventElapsedTime(&ms_reduce, ev_reduce_start, ev_reduce_stop));
      RAFT_CUDA_TRY(cudaEventElapsedTime(&ms_finalize, ev_finalize_start, ev_finalize_stop));
      CUML_LOG_DEBUG(
        "SPORF::predict timings (ms): total=%f tree_loop=%f reduce=%f finalize=%f (n_trees=%d streams=%d)",
        ms_total,
        ms_tree,
        ms_reduce,
        ms_finalize,
        this->rf_params.n_trees,
        predict_streams);

      RAFT_CUDA_TRY(cudaEventDestroy(ev_total_start));
      RAFT_CUDA_TRY(cudaEventDestroy(ev_total_stop));
      RAFT_CUDA_TRY(cudaEventDestroy(ev_tree_start));
      RAFT_CUDA_TRY(cudaEventDestroy(ev_tree_stop));
      RAFT_CUDA_TRY(cudaEventDestroy(ev_reduce_start));
      RAFT_CUDA_TRY(cudaEventDestroy(ev_reduce_stop));
      RAFT_CUDA_TRY(cudaEventDestroy(ev_finalize_start));
      RAFT_CUDA_TRY(cudaEventDestroy(ev_finalize_stop));
    }
    default_logger().set_pattern(default_pattern());
  }

  /**
   * @brief Predict target feature for input data and score against ref_labels.
   * @param[in] user_handle: raft::handle_t.
   * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
   * @param[in] n_rows: number of  data samples.
   * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
   * @param[in] verbosity: verbosity level for logging messages during execution
   * @param[in] rf_type: task type: 0 for classification, 1 for regression
   */
  static RF_metrics score(const raft::handle_t& user_handle,
                          const L* ref_labels,
                          int n_rows,
                          const L* predictions,
                          rapids_logger::level_enum verbosity,
                          int rf_type = RF_type::CLASSIFICATION)
  {
    ML::default_logger().set_level(verbosity);
    cudaStream_t stream = user_handle.get_stream();
    RF_metrics stats;
    if (rf_type == RF_type::CLASSIFICATION) {  // task classifiation: get classification metrics
      float accuracy = raft::stats::accuracy(predictions, ref_labels, n_rows, stream);
      stats          = set_rf_metrics_classification(accuracy);
      if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) print(stats);

      /* TODO: Potentially augment RF_metrics w/ more metrics (e.g., precision, F1, etc.).
        For non binary classification problems (i.e., one target and  > 2 labels), need avg.
        for each of these metrics */
    } else {  // regression task: get regression metrics
      double mean_abs_error, mean_squared_error, median_abs_error;
      raft::stats::regression_metrics(predictions,
                                      ref_labels,
                                      n_rows,
                                      stream,
                                      mean_abs_error,
                                      mean_squared_error,
                                      median_abs_error);
      stats = set_rf_metrics_regression(mean_abs_error, mean_squared_error, median_abs_error);
      if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) print(stats);
    }

    return stats;
  }
};

// class specializations
template class SPORF<float, int>;
template class SPORF<float, float>;
template class SPORF<double, int>;
template class SPORF<double, double>;

}  // End namespace ML
