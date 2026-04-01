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

#include <decisiontree/decisiontree.cuh>
#include <decisiontree/sporfdecisiontree.cuh>
#include <decisiontree/treelite_util.h>
#include <algorithm>
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
    this->error_checking(input, labels, n_rows, n_cols, false);
    const raft::handle_t& handle = user_handle;
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

    // Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
    // selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device
    // ptr.
    // Use a deque instead of vector because it can be used on objects with a deleted copy
    // constructor
    std::deque<rmm::device_uvector<int>> selected_rows;
    for (int i = 0; i < n_streams; i++) {
      selected_rows.emplace_back(n_sampled_rows, handle.get_stream_from_stream_pool(i));
    }


    #pragma omp parallel for num_threads(n_streams)
    for (int i = 0; i < this->rf_params.n_trees; i++) {
      int stream_id = omp_get_thread_num();
      auto s        = handle.get_stream_from_stream_pool(stream_id);
      RAFT_CUDA_TRY(cudaSetDevice(handle.get_device()));

      this->get_row_sample(i, n_rows, &selected_rows[stream_id], s);

      /* Build individual tree in the forest.
        - input is a pointer to orig data that have n_cols features and n_rows rows.
        - n_sampled_rows: # rows sampled for tree's bootstrap sample.
        - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements)
          used to build the bootstrapped sample.
          Expectation: Each tree node will contain (a) # n_sampled_rows and
          (b) a pointer to a list of row numbers w.r.t original data.
      */

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
                                                    i);
    }
    // Cleanup
    handle.sync_stream_pool();
    handle.sync_stream();
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
