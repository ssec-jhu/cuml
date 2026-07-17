/*
 * sporfbuilder.cuh
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

#include "kernels/builder_kernels.cuh"
#include "kernels/sporf_builder_kernels.cuh"

#include <common/Timer.h>

#include <cublas_v2.h>

#include <cuml/tree/decisiontree.hpp>
#include <cuml/tree/sporfdecisiontree.hpp>
#include <cuml/tree/flatnode.h>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/matrix/matrix.cuh>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace ML {
namespace DT {

template <typename DataT, typename LabelT, typename IdxT>
struct SPORFTrainingProjectionWorkspace;
template <typename DataT, typename LabelT, typename IdxT>
struct TrainingProjectionWorkspacePointers;
template <typename DataT, typename LabelT, typename IdxT>
struct TrainingProjectionWorkspaceMeta;
struct PersistWinningTreeProjectionTimings;
template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_training_projection_kernel(
  const TrainingProjectionWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const TrainingProjectionWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  cudaStream_t stream);
template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_training_random_matrix_sparse_kernel(
  const TrainingProjectionWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const TrainingProjectionWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  cudaStream_t stream);
template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_training_random_matrix_dense_kernel(
  const TrainingProjectionWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const TrainingProjectionWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  cudaStream_t stream);
template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_training_quantile_sampling_kernel(
  const TrainingProjectionWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const TrainingProjectionWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  IdxT* d_quantile_indices,
  IdxT max_n_bins,
  IdxT min_samples_leaf,
  cudaStream_t stream);
template <typename DataT, typename LabelT, typename IdxT>
void launch_capture_best_training_projection_kernel(
  const TrainingProjectionWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const TrainingProjectionWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  const Split<DataT, IdxT>* d_splits,
  cudaStream_t stream);
template <typename DataT, typename LabelT, typename IdxT>
void launch_restore_best_training_projection_kernel(
  const TrainingProjectionWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const TrainingProjectionWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  Split<DataT, IdxT>* d_splits,
  cudaStream_t stream);
template <typename DataT, typename LabelT, typename IdxT>
void launch_store_winning_tree_projection_vectors_kernel(
  const TrainingProjectionWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const TrainingProjectionWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  const Split<DataT, IdxT>* d_splits,
  std::size_t payload_base_offset,
  cudaStream_t stream);
template <typename DataT, typename LabelT, typename IdxT>
PersistWinningTreeProjectionTimings persist_winning_tree_projection_vectors(
  SPORFTrainingProjectionWorkspace<DataT, LabelT, IdxT>& workspace,
  const Split<DataT, IdxT>* d_splits,
  cudaStream_t stream);


/**
 * Structure that manages the iterative batched-level training and building of nodes
 * in the host.
 */
template <typename DataT, typename LabelT>
class SPORFNodeQueue {
  using IdxT = int;
  using NodeT = SparseTreeNode<DataT, LabelT>;
  using TreeMetaDataNodeT = DT::ObliqueTreeMetaDataNode<DataT, LabelT>;
  const SPORFDecisionTreeParams params;
  const IdxT n_features;
  const std::size_t min_rows_per_batch_;
  std::shared_ptr<TreeMetaDataNodeT> tree;
  std::vector<SPORFDT::InstanceRange> node_instances_;
  std::deque<SPORFDT::NodeWorkItem> work_items_;

 public:
  struct PopBatch {
    std::vector<SPORFDT::NodeWorkItem> popped;
    std::vector<DT::BlockTask<IdxT>> projection_block_tasks;
    std::vector<DT::NodeWorkItemChunk<IdxT>> projection_chunks;
    std::vector<DT::BlockTask<IdxT>> projection_matrix_block_tasks;
    std::vector<DT::NodeWorkItemChunk<IdxT>> projection_matrix_chunks;

    void clear()
    {
      popped.clear();
      projection_block_tasks.clear();
      projection_chunks.clear();
      projection_matrix_block_tasks.clear();
      projection_matrix_chunks.clear();
    }

    void reserve(std::size_t batch_cap)
    {
      popped.reserve(batch_cap);
      projection_block_tasks.reserve(batch_cap);
      projection_chunks.reserve(batch_cap);
      projection_matrix_block_tasks.reserve(batch_cap);
      projection_matrix_chunks.reserve(batch_cap);
    }
  };

  SPORFNodeQueue(SPORFDecisionTreeParams params,
                 size_t max_nodes,
                 size_t sampled_rows,
                 int num_outputs,
                 IdxT n_features_,
                 std::size_t min_rows_per_batch)
    : params(params),
      n_features(n_features_),
      min_rows_per_batch_(std::max<std::size_t>(1, min_rows_per_batch)),
      tree(std::make_shared<TreeMetaDataNodeT>())
  {
    tree->num_outputs = num_outputs;
    tree->sparsetree.reserve(max_nodes);
    tree->sparsetree.emplace_back(NodeT::CreateLeafNode(sampled_rows));
    tree->projection_vectors.reserve(max_nodes);
    tree->projection_vectors.resize(max_nodes, DT::OffsetProjectionMatrix<IdxT>{0, 0, 0, 0});
    tree->leaf_counter  = 1;
    tree->depth_counter = 0;
    node_instances_.reserve(max_nodes);
    node_instances_.emplace_back(SPORFDT::InstanceRange{0, sampled_rows});
    if (this->IsExpandable(tree->sparsetree.back(), 0)) {
      work_items_.emplace_back(SPORFDT::NodeWorkItem{0, 0, 0, node_instances_.back()});
    }
  }

  std::shared_ptr<TreeMetaDataNodeT> GetTree() { return tree; }
  const std::vector<SPORFDT::InstanceRange>& GetInstanceRanges() { return node_instances_; }

  bool HasWork() { return work_items_.size() > 0; }

  std::size_t Pop(PopBatch& batch,
                  double* queue_extract_ms = nullptr,
                  double* projection_build_ms = nullptr)
  {
    size_t total_rows = 0;
    auto batch_cap = std::min(size_t(params.max_batch_size), work_items_.size());
    batch.clear();
    batch.reserve(batch_cap);
    auto& popped = batch.popped;
    auto& projection_block_tasks = batch.projection_block_tasks;
    auto& projection_chunks = batch.projection_chunks;
    auto& projection_matrix_block_tasks = batch.projection_matrix_block_tasks;
    auto& projection_matrix_chunks = batch.projection_matrix_chunks;

    while (work_items_.size() > 0 && popped.size() < std::size_t(params.max_batch_size) &&
           (popped.empty() || total_rows < min_rows_per_batch_)) {
      auto t_queue_extract = std::chrono::steady_clock::now();
      popped.emplace_back(work_items_.front());
      work_items_.pop_front();
      total_rows += popped.back().instances.count;
      if (queue_extract_ms != nullptr) {
        *queue_extract_ms += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t_queue_extract)
                               .count();
      }

      auto* work_item = &popped.back();
      auto count = static_cast<IdxT>(work_item->instances.count);
      if (count < static_cast<IdxT>(params.min_samples_split)) { continue; }

      auto t_projection_build = std::chrono::steady_clock::now();
      for (IdxT threads_left = count,
                instances_begin = static_cast<IdxT>(work_item->instances.begin);
           threads_left > 0;) {
        if (projection_block_tasks.empty() || projection_block_tasks.back().count == DT::BLOCK_TASK_SIZE) {
          projection_block_tasks.emplace_back(DT::BlockTask<IdxT>{});
          projection_block_tasks.back().count = 0;
        }

        auto* block_task = &projection_block_tasks.back();
        IdxT thread_count =
          std::min<IdxT>(DT::BLOCK_TASK_SIZE - block_task->count, threads_left);

        projection_chunks.emplace_back(DT::NodeWorkItemChunk<IdxT>{
          static_cast<IdxT>(popped.size() - 1),       // work_item_idx (batch-local)
          static_cast<IdxT>(work_item->idx),          // node_id (global tree node id)
          instances_begin,                            // instances_begin
          thread_count,                               // instances_count
          static_cast<IdxT>(projection_block_tasks.size() - 1),  // block_task_idx
          block_task->count,                          // thread_local_begin
          0,                                          // nLeft
          0,                                          // nRight
          0,                                          // loff
          0                                           // roff
        });

        auto chunk_idx = static_cast<IdxT>(projection_chunks.size() - 1);
        auto* chunk = &projection_chunks.back();
        for (IdxT i = chunk->thread_local_begin;
             i < chunk->thread_local_begin + chunk->instances_count;
             i++) {
          block_task->work_item_chunk_ids[i] = chunk_idx;
        }

        block_task->count += thread_count;
        instances_begin += thread_count;
        threads_left -= thread_count;
      }

      for (IdxT threads_left = n_features, feature_begin = 0; threads_left > 0;) {
        if (projection_matrix_block_tasks.empty() ||
            projection_matrix_block_tasks.back().count == DT::BLOCK_TASK_SIZE) {
          projection_matrix_block_tasks.emplace_back(DT::BlockTask<IdxT>{});
          projection_matrix_block_tasks.back().count = 0;
        }

        auto* block_task = &projection_matrix_block_tasks.back();
        IdxT thread_count =
          std::min<IdxT>(DT::BLOCK_TASK_SIZE - block_task->count, threads_left);

        projection_matrix_chunks.emplace_back(DT::NodeWorkItemChunk<IdxT>{
          static_cast<IdxT>(popped.size() - 1),                    // work_item_idx (batch-local)
          static_cast<IdxT>(work_item->idx),                       // node_id
          feature_begin,                                           // feature_begin
          thread_count,                                            // feature_count
          static_cast<IdxT>(projection_matrix_block_tasks.size() - 1),  // block_task_idx
          block_task->count,                                       // thread_local_begin
          0,
          0,
          0,
          0
        });

        auto chunk_idx = static_cast<IdxT>(projection_matrix_chunks.size() - 1);
        auto* chunk = &projection_matrix_chunks.back();
        for (IdxT i = chunk->thread_local_begin;
             i < chunk->thread_local_begin + chunk->instances_count;
             i++) {
          block_task->work_item_chunk_ids[i] = chunk_idx;
        }

        block_task->count += thread_count;
        feature_begin += thread_count;
        threads_left -= thread_count;
      }
      if (projection_build_ms != nullptr) {
        *projection_build_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t_projection_build)
                                  .count();
      }
    }
    return total_rows;
  }

  // This node is allowed to be expanded further (if its split gain is high enough)
  bool IsExpandable(const NodeT& n, int depth)
  {
    if (depth >= params.max_depth) return false;
    if (int(n.InstanceCount()) < params.min_samples_split) return false;
    if (params.max_leaves != -1 && tree->leaf_counter >= params.max_leaves) return false;
    return true;
  }

  template <typename SplitT>
  void Push(const std::vector<SPORFDT::NodeWorkItem>& work_items, SplitT* h_splits)
  {
    // Update node queue based on splits
    for (std::size_t i = 0; i < work_items.size(); i++) {

      auto split        = h_splits[i];
      auto item         = work_items[i];
      auto parent_range = node_instances_.at(item.idx);
      if (SplitNotValid(
            split, params.min_impurity_decrease, params.min_samples_leaf, parent_range.count)) {
        continue;
      }

      if (params.max_leaves != -1 && tree->leaf_counter >= params.max_leaves) break;

      // parent
      tree->sparsetree.at(item.idx) = NodeT::CreateSplitNode(split.colid,
                                                             split.quesval,
                                                             split.best_metric_val,
                                                             int64_t(tree->sparsetree.size()),
                                                             parent_range.count);
      tree->leaf_counter++;

      // left
      tree->sparsetree.emplace_back(NodeT::CreateLeafNode(split.nLeft));
      node_instances_.emplace_back(SPORFDT::InstanceRange{parent_range.begin, std::size_t(split.nLeft)});

      // Do not add a work item if this child is definitely a leaf
      if (this->IsExpandable(tree->sparsetree.back(), item.depth + 1)) {
        work_items_.emplace_back(
          SPORFDT::NodeWorkItem{tree->sparsetree.size() - 1, item.depth + 1, 0, node_instances_.back()});
      }

      // right
      tree->sparsetree.emplace_back(NodeT::CreateLeafNode(parent_range.count - split.nLeft));
      node_instances_.emplace_back(
        SPORFDT::InstanceRange{parent_range.begin + split.nLeft, parent_range.count - split.nLeft});

      // Do not add a work item if this child is definitely a leaf
      if (this->IsExpandable(tree->sparsetree.back(), item.depth + 1)) {
        work_items_.emplace_back(
          SPORFDT::NodeWorkItem{tree->sparsetree.size() - 1, item.depth + 1, 0, node_instances_.back()});
      }

      // update depth
      tree->depth_counter = max(tree->depth_counter, item.depth + 1);
    }
  }
};

/**
 * Internal struct used to do all the heavy-lifting required for tree building
 */
template <typename ObjectiveT>
struct SPORFBuilder {
  typedef typename ObjectiveT::DataT DataT;
  typedef typename ObjectiveT::LabelT LabelT;
  typedef typename ObjectiveT::IdxT IdxT;
  typedef typename ObjectiveT::BinT BinT;
  typedef SparseTreeNode<DataT, LabelT, IdxT> NodeT;
  typedef ObliqueTreeMetaDataNode<DataT, LabelT> TreeMetaDataNodeT;
  typedef Split<DataT, IdxT> SplitT;
  typedef Dataset<DataT, LabelT, IdxT> DatasetT;
  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 128;
  // When n_bins <= TPB, using >1 item per thread makes sorted quantiles
  // vanish into out-of-range slots. Keep this at 1.
  static constexpr int ITEMS_PER_THREAD = 1;
  /** handle to get device properties */
  const raft::handle_t& handle;
  /** stream to launch kernels */
  cudaStream_t builder_stream;
  /** DT params */
  SPORFDecisionTreeParams params;
  /** input dataset */
  DatasetT dataset;
  DatasetT dataset_proj;
  /** Tree index */
  IdxT treeid;
  /** Seed used for randomization */
  uint64_t seed;
  /** number of nodes created in the current batch */
  IdxT* n_nodes;
  /** buffer of segmented histograms*/
  BinT* histograms;
  /** threadblock arrival count */
  int* done_count;
  /** mutex array used for atomically updating best split */
  int* mutex;
  /** best splits for the current batch of nodes */
  SplitT* splits;
  /** current batch of nodes */
  SPORFDT::NodeWorkItem* d_work_items;
  /** device AOS to map CTA blocks along dimx to nodes of a batch */
  SPORFDT::WorkloadInfo<IdxT>* workload_info;
  /** host AOS to map CTA blocks along dimx to nodes of a batch */
  SPORFDT::WorkloadInfo<IdxT>* h_workload_info;
  /** maximum CTA blocks along dimx */
  int max_blocks_dimx = 0;
  /** host array of splits */
  SplitT* h_splits;
  /** number of blocks used to parallelize column-wise computations */
  int n_blks_for_cols = 10;
  SPORFTrainingProjectionWorkspace<DataT, LabelT, IdxT>& projection_ws;
  DT::SPORFDeviceBatchingPolicy device_batching_policy;

  struct Stats {
    double t_pop;
    double t_pop_queue_extract;
    double t_pop_projection_build;
    double t_push;
    double t_h2d;
    double t_d2h;
    double t_kernels;
    double t_kernels_pre_sync;
    double t_workload_info_cpu;
    double t_generation_storage_setup;
    double t_random_matrix_kernel;
    double t_training_projection_kernel;
    double t_quantile_sampling_kernel;
    double t_compute_split_kernel;
    double t_node_split_kernel;
    double t_node_split_d2h_sync;
    double t_split_postprocess_cpu;
    double t_projection_store_pre_sync;
    double t_projection_store_device;
    double t_tree_projection_finalize;
    double t_leaf_predictions;
    std::size_t pop_batches;
    std::size_t pop_total_rows;
    std::size_t pop_max_rows;

    Stats()
      : t_pop(0),
        t_pop_queue_extract(0),
        t_pop_projection_build(0),
        t_push(0),
        t_h2d(0),
        t_d2h(0),
        t_kernels(0),
        t_kernels_pre_sync(0),
        t_workload_info_cpu(0),
        t_generation_storage_setup(0),
        t_random_matrix_kernel(0),
        t_training_projection_kernel(0),
        t_quantile_sampling_kernel(0),
        t_compute_split_kernel(0),
        t_node_split_kernel(0),
        t_node_split_d2h_sync(0),
        t_split_postprocess_cpu(0),
        t_projection_store_pre_sync(0),
        t_projection_store_device(0),
        t_tree_projection_finalize(0),
        t_leaf_predictions(0),
        pop_batches(0),
        pop_total_rows(0),
        pop_max_rows(0)
    {
    }
  };

  struct HyperparameterDiagnostics {
    bool enabled = false;

    double density_specified = 0.0;
    double density_used      = 0.0;
    double expected_nnz      = 0.0;
    double max_features_rate = 0.0;

    IdxT projections_specified = 0;

    std::size_t nnz_count = 0;
    std::size_t nnz_sum   = 0;
    IdxT nnz_min          = std::numeric_limits<IdxT>::max();
    IdxT nnz_max          = 0;

    std::size_t projection_node_count  = 0;
    std::size_t projection_attempt_sum = 0;
    IdxT projection_attempt_min        = std::numeric_limits<IdxT>::max();
    IdxT projection_attempt_max        = 0;
    std::vector<IdxT> nnz_values;
    std::vector<IdxT> projection_attempt_values;

    void recordNnz(IdxT nnz)
    {
      nnz_count++;
      nnz_sum += static_cast<std::size_t>(nnz);
      nnz_min = std::min(nnz_min, nnz);
      nnz_max = std::max(nnz_max, nnz);
      nnz_values.push_back(nnz);
    }

    void recordProjectionAttempts(IdxT n_attempts)
    {
      projection_node_count++;
      projection_attempt_sum += static_cast<std::size_t>(n_attempts);
      projection_attempt_min = std::min(projection_attempt_min, n_attempts);
      projection_attempt_max = std::max(projection_attempt_max, n_attempts);
      projection_attempt_values.push_back(n_attempts);
    }
  };

  Stats stats;
  HyperparameterDiagnostics hparam_debug;

  SPORFBuilder(const raft::handle_t& handle,
          cudaStream_t s,
          IdxT treeid,
          uint64_t seed,
          const SPORFDecisionTreeParams& p,
          const DataT* data,
          const LabelT* labels,
          IdxT n_rows,
          IdxT n_cols,
          rmm::device_uvector<IdxT>* row_ids,
          IdxT n_classes,
          SPORFTrainingProjectionWorkspace<DataT, LabelT, IdxT>& projection_ws_,
          const DT::SPORFDeviceBatchingPolicy& device_batching_policy_)
    : handle(handle),
      builder_stream(s),
      treeid(treeid),
      seed(seed),
      params(p),
      dataset{data,
              labels,
              n_rows,
              n_cols,
              int(row_ids->size()),
              max(1, IdxT(params.max_features * n_cols)),
              row_ids->data(),
              n_classes
      },
      projection_ws(projection_ws_),
      device_batching_policy(device_batching_policy_),
      dataset_proj{
        0,
        labels,
        n_rows,
        max(1, IdxT(params.max_features * n_cols)),
        int(row_ids->size()),
        max(1, IdxT(params.max_features * n_cols)),
        row_ids->data(),
        n_classes
      }
  {
    max_blocks_dimx = 1 + params.max_batch_size + dataset.n_sampled_rows / TPB_DEFAULT;
    ASSERT(n_classes >= 1, "n_classes should be at least 1");
    ASSERT(TPB_DEFAULT * ITEMS_PER_THREAD >= params.max_n_bins,
      "max_n_bins must be <= 2048 for proper functioning of quantile sorting.");

    projection_ws.ensure_tree_projection_vector_capacity(this->maxNodes(), builder_stream);
    projection_ws.clear_tree_projection_state(builder_stream);
    size_t max_len_histograms = size_t(params.max_batch_size) * size_t(params.max_n_bins) *
                                size_t(n_blks_for_cols) * size_t(dataset.num_outputs);
    projection_ws.ensure_histogram_storage(sizeof(BinT) * max_len_histograms, builder_stream);
    constexpr IdxT projection_tile_size = IdxT{4};
    projection_ws.ensure_training_projection_values_storage(
      size_t(dataset.n_sampled_rows) * size_t(projection_tile_size), builder_stream);
    projection_ws.ensure_quantile_indices_storage(
      size_t(params.max_batch_size) * size_t(projection_tile_size) * size_t(params.max_n_bins),
      builder_stream);
    projection_ws.ensure_split_scratch_storage(size_t(params.max_batch_size),
                                               size_t(n_blks_for_cols),
                                               size_t(max_blocks_dimx),
                                               builder_stream);
    projection_ws.ensure_host_split_scratch_storage(size_t(params.max_batch_size),
                                                    size_t(max_blocks_dimx));

    n_nodes = projection_ws.d_split_n_nodes_storage.data();
    histograms = reinterpret_cast<BinT*>(projection_ws.d_histogram_storage.data());
    done_count = projection_ws.d_split_done_count_storage.data();
    mutex = projection_ws.d_split_mutex_storage.data();
    splits = projection_ws.d_split_storage.data();
    d_work_items = projection_ws.pointers.projection.d_work_items;
    workload_info = projection_ws.d_workload_info_storage.data();
    h_workload_info =
      reinterpret_cast<SPORFDT::WorkloadInfo<IdxT>*>(projection_ws.h_workload_info_storage.data());
    h_splits = reinterpret_cast<SplitT*>(projection_ws.h_split_storage.data());

    hparam_debug.enabled =
      ML::default_logger().should_log(rapids_logger::level_enum::debug);
    hparam_debug.density_specified = static_cast<double>(params.density_specified);
    hparam_debug.density_used      = static_cast<double>(params.density);
    hparam_debug.expected_nnz =
      static_cast<double>(dataset.N) * static_cast<double>(params.density);
    hparam_debug.max_features_rate      = static_cast<double>(params.max_features);
    hparam_debug.projections_specified  = dataset.n_sampled_cols;
  }

  /**
   * @brief returns maximum nodes possible per tree
   * @return maximum nodes possible per tree
   */
  size_t maxNodes() const
  {
    auto levels = static_cast<unsigned int>(params.max_depth + 1);
    if (levels >= sizeof(size_t) * 8) { return std::numeric_limits<size_t>::max(); }
    return (size_t{1} << levels) - 1;
  }

  /**
   * @brief trains the tree, builds the nodes
   *
   * @return trained tree structure
   */
  std::shared_ptr<TreeMetaDataNodeT> train()
  {
    raft::common::nvtx::range fun_scope("SPORFBuilder::train @sporfbuilder.cuh [batched-levelalgo]");
    MLCommon::TimerCPU timer;
    auto t_train_wall_start = std::chrono::steady_clock::now();
    double t_doSplit_wall = 0.0;
    SPORFNodeQueue<DataT, LabelT> queue(
      params,
      this->maxNodes(),
      dataset.n_sampled_rows,
      dataset.num_outputs,
      dataset.N,
      device_batching_policy.target_rows_per_batch);
    typename SPORFNodeQueue<DataT, LabelT>::PopBatch popped_batch;
    while (queue.HasWork()) {
      auto t_pop = std::chrono::steady_clock::now();
      double t_pop_queue_extract = 0.0;
      double t_pop_projection_build = 0.0;
      auto pop_total_rows = queue.Pop(popped_batch, &t_pop_queue_extract, &t_pop_projection_build);
      stats.t_pop +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pop).count();
      stats.t_pop_queue_extract += t_pop_queue_extract;
      stats.t_pop_projection_build += t_pop_projection_build;
      stats.pop_batches += 1;
      stats.pop_total_rows += pop_total_rows;
      stats.pop_max_rows = std::max(stats.pop_max_rows, pop_total_rows);
      auto& work_items                    = popped_batch.popped;
      auto& projection_chunks             = popped_batch.projection_chunks;
      auto& projection_block_tasks        = popped_batch.projection_block_tasks;
      auto& projection_matrix_chunks      = popped_batch.projection_matrix_chunks;
      auto& projection_matrix_block_tasks = popped_batch.projection_matrix_block_tasks;
      auto t_doSplit_start = std::chrono::steady_clock::now();
      auto [splits_host_ptr, splits_count] =
        doSplit(work_items,
                projection_chunks,
                projection_block_tasks,
                projection_matrix_chunks,
                projection_matrix_block_tasks,
                projection_ws);
      t_doSplit_wall +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  t_doSplit_start)
          .count();
      auto t_push = std::chrono::steady_clock::now();
      queue.Push(work_items, splits_host_ptr);
      stats.t_push +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_push).count();
    }
    auto tree = queue.GetTree();
    tree->treeid = treeid;

    {
      auto t_tree_projection_finalize = std::chrono::steady_clock::now();
      IdxT h_tree_projection_max_node_idx = IdxT{-1};
      raft::update_host(&h_tree_projection_max_node_idx,
                        projection_ws.pointers.d_tree_projection_max_node_idx,
                        std::size_t{1},
                        builder_stream);
      handle.sync_stream(builder_stream);
      auto max_projection_node_idx = std::min<std::size_t>(
        tree->projection_vectors.size(),
        h_tree_projection_max_node_idx >= 0
          ? static_cast<std::size_t>(h_tree_projection_max_node_idx + 1)
          : std::size_t{0});
      std::vector<DT::OffsetProjectionMatrix<IdxT>> h_tree_projection_vectors(
        max_projection_node_idx);
      std::vector<IdxT> h_tree_projection_indptr_storage(max_projection_node_idx * 2);
      std::vector<IdxT> h_tree_projection_indices_storage(
        static_cast<std::size_t>(projection_ws.meta.tree_projection_payload_nnz));
      std::vector<DataT> h_tree_projection_coeffs_storage(
        static_cast<std::size_t>(projection_ws.meta.tree_projection_payload_nnz));
      if (!h_tree_projection_vectors.empty()) {
        raft::update_host(h_tree_projection_vectors.data(),
                          projection_ws.pointers.d_tree_projection_vectors,
                          h_tree_projection_vectors.size(),
                          builder_stream);
        raft::update_host(h_tree_projection_indptr_storage.data(),
                          projection_ws.pointers.d_tree_projection_indptr_storage,
                          h_tree_projection_indptr_storage.size(),
                          builder_stream);
      }
      if (!h_tree_projection_indices_storage.empty()) {
        raft::update_host(h_tree_projection_indices_storage.data(),
                          projection_ws.pointers.d_tree_projection_indices_storage,
                          h_tree_projection_indices_storage.size(),
                          builder_stream);
        raft::update_host(h_tree_projection_coeffs_storage.data(),
                          projection_ws.pointers.d_tree_projection_coeffs_storage,
                          h_tree_projection_coeffs_storage.size(),
                          builder_stream);
      }
      handle.sync_stream(builder_stream);

      tree->projection_vectors.assign(
        tree->sparsetree.size(), DT::OffsetProjectionMatrix<IdxT>{0, 0, 0, 0});
      for (size_t node_idx = 0; node_idx < max_projection_node_idx; ++node_idx) {
        tree->projection_vectors[node_idx] = h_tree_projection_vectors[node_idx];
      }
      tree->projection_indptr_storage = std::move(h_tree_projection_indptr_storage);
      tree->projection_indices_storage = std::move(h_tree_projection_indices_storage);
      tree->projection_coeffs_storage = std::move(h_tree_projection_coeffs_storage);
      stats.t_tree_projection_finalize +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  t_tree_projection_finalize)
          .count();
    }

    auto t_leaf_predictions = std::chrono::steady_clock::now();
    this->SetLeafPredictions(tree, queue.GetInstanceRanges());
    stats.t_leaf_predictions +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                t_leaf_predictions)
        .count();
    tree->train_time = timer.getElapsedMilliseconds();
    auto train_wall_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t_train_wall_start)
                           .count();

    if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) {
      std::cout << "SPORFBuilder::train: pop: " << stats.t_pop <<
        " ms, pop_queue_extract: " << stats.t_pop_queue_extract <<
        " ms, pop_projection_build: " << stats.t_pop_projection_build <<
        " ms, pop_batches: " << stats.pop_batches <<
        ", pop_total_rows: " << stats.pop_total_rows <<
        ", pop_avg_rows: " << (stats.pop_batches > 0
                                ? static_cast<double>(stats.pop_total_rows) /
                                    static_cast<double>(stats.pop_batches)
                                : 0.0) <<
        ", pop_max_rows: " << stats.pop_max_rows <<
        " ms, push: " << stats.t_push <<
        " ms, h2d: " << stats.t_h2d <<
        " ms, d2h: " << stats.t_d2h <<
        " ms, kernels: " << stats.t_kernels <<
        " ms, kernels_pre_sync: " << stats.t_kernels_pre_sync <<
        " ms, workload_info_cpu: " << stats.t_workload_info_cpu <<
        " ms, generation_storage_setup: " << stats.t_generation_storage_setup <<
        " ms, random_matrix_kernel: " << stats.t_random_matrix_kernel <<
        " ms, training_projection_kernel: " << stats.t_training_projection_kernel <<
        " ms, quantile_sampling_kernel: " << stats.t_quantile_sampling_kernel <<
        " ms, compute_split_kernel: " << stats.t_compute_split_kernel <<
        " ms, node_split_kernel: " << stats.t_node_split_kernel <<
        " ms, node_split_d2h_sync: " << stats.t_node_split_d2h_sync <<
        " ms, split_postprocess_cpu: " << stats.t_split_postprocess_cpu <<
        " ms, projection_store_pre_sync: " << stats.t_projection_store_pre_sync <<
        " ms, projection_store_device: " << stats.t_projection_store_device <<
        " ms, tree_projection_finalize: " << stats.t_tree_projection_finalize <<
        " ms, leaf_predictions: " << stats.t_leaf_predictions <<
        " ms, wall_total: " << train_wall_ms <<
        " ms, wall_doSplit: " << t_doSplit_wall <<
        " ms" << std::endl;
      printHyperparameterDiagnostics();
    }


    return tree;
  }

 private:
  auto updateWorkloadInfo(const std::vector<SPORFDT::NodeWorkItem>& work_items)
  {
    int n_large_nodes = 0;  // large nodes are nodes having training instances larger than block
                            // size, hence require global memory for histogram construction
    int n_blocks_dimx = 0;  // gridDim.x required for computeSplitKernel
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto item = work_items[i];
      int n_blocks_per_node =
        std::max(raft::ceildiv(item.instances.count, size_t(TPB_DEFAULT)), size_t(1));

      if (n_blocks_per_node > 1) ++n_large_nodes;

      for (int b = 0; b < n_blocks_per_node; b++) {
        ASSERT(n_blocks_dimx + b < max_blocks_dimx,
               "SPORF workload_info overflow: n_blocks_dimx=%d block=%d max_blocks_dimx=%d "
               "work_items=%zu item=%zu item_rows=%zu",
               n_blocks_dimx,
               b,
               max_blocks_dimx,
               work_items.size(),
               i,
               item.instances.count);
        h_workload_info[n_blocks_dimx + b] = {int(i), n_large_nodes - 1, b, n_blocks_per_node};
      }
      n_blocks_dimx += n_blocks_per_node;
    }
    return std::make_pair(n_blocks_dimx, n_large_nodes);
  }

  auto doSplit(const std::vector<SPORFDT::NodeWorkItem>& work_items,
               const std::vector<DT::NodeWorkItemChunk<IdxT>>& projection_chunks,
               const std::vector<DT::BlockTask<IdxT>>& projection_block_tasks,
               const std::vector<DT::NodeWorkItemChunk<IdxT>>& projection_matrix_chunks,
               const std::vector<DT::BlockTask<IdxT>>& projection_matrix_block_tasks,
               SPORFTrainingProjectionWorkspace<DataT, LabelT, IdxT>& projection_ws)
  {
    raft::common::nvtx::range fun_scope("SPORFBuilder::doSplit @sporfbuilder.cuh [batched-levelalgo]");
    const bool do_timing = ML::default_logger().should_log(rapids_logger::level_enum::debug);
    auto t_cpu = std::chrono::steady_clock::now();
    auto [n_blocks_dimx, n_large_nodes] = this->updateWorkloadInfo(work_items);

    stats.t_workload_info_cpu +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cpu).count();

    auto t_generation_storage_setup = std::chrono::steady_clock::now();
    projection_ws.reset(builder_stream);
    projection_ws.meta.input_n_rows = dataset.M;
    projection_ws.meta.projection.n_rows = dataset.n_sampled_rows;
    projection_ws.meta.projection.n_cols = dataset.N;
    projection_ws.meta.projection.n_work_items = static_cast<IdxT>(work_items.size());
    projection_ws.meta.projection.n_chunks = static_cast<IdxT>(projection_chunks.size());
    projection_ws.meta.projection.n_block_tasks = static_cast<IdxT>(projection_block_tasks.size());
    projection_ws.peak_projection_work_items =
      std::max(projection_ws.peak_projection_work_items, work_items.size());
    projection_ws.peak_projection_chunks =
      std::max(projection_ws.peak_projection_chunks, projection_chunks.size());
    projection_ws.peak_projection_block_tasks =
      std::max(projection_ws.peak_projection_block_tasks, projection_block_tasks.size());
    // In the SPORF builder, `n_sampled_cols` is reused as the random-projection
    // output dimensionality, i.e. the number of projection components per node.
    IdxT total_proj_components = dataset.n_sampled_cols;
    bool use_dense_generation =
      useDenseProjectionGeneration(work_items.size(), total_proj_components);
    IdxT projection_tile_size = use_dense_generation ? total_proj_components : IdxT{4};
    projection_ws.meta.projection.n_proj_components = projection_tile_size;
    projection_ws.meta.n_generation_chunks = static_cast<IdxT>(projection_matrix_chunks.size());
    projection_ws.meta.n_generation_block_tasks =
      static_cast<IdxT>(projection_matrix_block_tasks.size());
    projection_ws.meta.generation_n_features = dataset.N;
    projection_ws.meta.generation_min_samples_split = static_cast<IdxT>(params.min_samples_split);
    projection_ws.meta.generation_density = static_cast<DataT>(params.density);
    projection_ws.meta.generation_nnz_per_component = dataset.N;
    projection_ws.meta.generation_total_proj_components = total_proj_components;
    projection_ws.meta.generation_fixed_capacity = !use_dense_generation;
    recordProjectionAttemptDiagnostics(work_items, total_proj_components);
    auto generation_random_state = fnv1a32_basis;
    generation_random_state = fnv1a32(generation_random_state, static_cast<uint32_t>(seed));
    generation_random_state = fnv1a32(generation_random_state, static_cast<uint32_t>(seed >> 32));
    generation_random_state = fnv1a32(generation_random_state, static_cast<uint32_t>(treeid));
    projection_ws.meta.generation_random_state =
      static_cast<int>(generation_random_state & 0x7fffffffU);
    projection_ws.pointers.d_input_col_major = dataset.data;
    projection_ws.pointers.d_row_ids = dataset.row_ids;
    projection_ws.ensure_generation_metadata_capacity(projection_ws.meta.n_generation_chunks,
                                                      projection_ws.meta.n_generation_block_tasks,
                                                      builder_stream);
    projection_ws.ensure_training_projection_values_storage(
      size_t(dataset.n_sampled_rows) * size_t(projection_tile_size), builder_stream);
    projection_ws.ensure_quantile_indices_storage(
      size_t(work_items.size()) * size_t(projection_tile_size) * size_t(params.max_n_bins),
      builder_stream);
    d_work_items = projection_ws.pointers.projection.d_work_items;
    size_t generation_len = static_cast<size_t>(work_items.size()) *
                            static_cast<size_t>(projection_tile_size) *
                            static_cast<size_t>(dataset.N);
    size_t generation_indptr_len = static_cast<size_t>(work_items.size()) *
                                   static_cast<size_t>(projection_tile_size + 1);

    if (use_dense_generation) {
      projection_ws.resize_dense_generation_storage(generation_len, generation_indptr_len, builder_stream);
    } else {
      projection_ws.resize_generation_storage(generation_len, generation_indptr_len, builder_stream);
      projection_ws.resize_best_projection_storage(static_cast<size_t>(work_items.size()) *
                                                     static_cast<size_t>(dataset.N),
                                                   static_cast<size_t>(work_items.size()) * 2,
                                                   builder_stream);
    }
    stats.t_generation_storage_setup +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                t_generation_storage_setup)
        .count();

    ASSERT(projection_ws.meta.projection.n_work_items <= projection_ws.meta.projection.cap_work_items,
           "Training projection workspace overflow: work_items");
    ASSERT(projection_ws.meta.projection.n_chunks <= projection_ws.meta.projection.cap_chunks,
           "Training projection workspace overflow: chunks");
    ASSERT(projection_ws.meta.projection.n_block_tasks <= projection_ws.meta.projection.cap_block_tasks,
           "Training projection workspace overflow: block_tasks");

    auto t_kernel = std::chrono::steady_clock::now();
    // start fresh on the number of *new* nodes created in this batch
    RAFT_CUDA_TRY(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), builder_stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      done_count, 0, sizeof(int) * params.max_batch_size * n_blks_for_cols, builder_stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(mutex, 0, sizeof(int) * params.max_batch_size, builder_stream));
    initSplit<DataT, IdxT, TPB_DEFAULT>(splits, work_items.size(), builder_stream);
    stats.t_kernels +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_kernel).count();

    auto t_h2d = std::chrono::steady_clock::now();
    // get the current set of nodes to be worked upon
    if (!work_items.empty()) {
      raft::update_device(d_work_items, work_items.data(), work_items.size(), builder_stream);
    }
    raft::update_device(workload_info, h_workload_info, n_blocks_dimx, builder_stream);
    if (!projection_chunks.empty()) {
      raft::update_device(projection_ws.pointers.projection.d_chunks,
                          projection_chunks.data(),
                          projection_ws.meta.projection.n_chunks,
                          builder_stream);
    }
    if (!projection_block_tasks.empty()) {
      raft::update_device(projection_ws.pointers.projection.d_block_tasks,
                          projection_block_tasks.data(),
                          projection_ws.meta.projection.n_block_tasks,
                          builder_stream);
    }
    if (!projection_matrix_chunks.empty()) {
      raft::update_device(projection_ws.pointers.d_generation_chunks,
                          projection_matrix_chunks.data(),
                          projection_ws.meta.n_generation_chunks,
                          builder_stream);
    }
    if (!projection_matrix_block_tasks.empty()) {
      raft::update_device(projection_ws.pointers.d_generation_block_tasks,
                          projection_matrix_block_tasks.data(),
                          projection_ws.meta.n_generation_block_tasks,
                          builder_stream);
    }
    stats.t_h2d +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_h2d).count();

    cudaEvent_t ev_kernel_start{}, ev_kernel_stop{};
    if (do_timing) {
      RAFT_CUDA_TRY(cudaEventCreate(&ev_kernel_start));
      RAFT_CUDA_TRY(cudaEventCreate(&ev_kernel_stop));
    }

    t_kernel = std::chrono::steady_clock::now();
    if (do_timing) { RAFT_CUDA_TRY(cudaEventRecord(ev_kernel_start, builder_stream)); }
    dataset_proj.data = projection_ws.pointers.projection.d_trans;
    dataset_proj.N = projection_tile_size;
    dataset_proj.n_sampled_cols = projection_tile_size;
    t_kernel = std::chrono::steady_clock::now();
    if (do_timing) { RAFT_CUDA_TRY(cudaEventRecord(ev_kernel_start, builder_stream)); }
    if (use_dense_generation) {
      projection_ws.meta.generation_projection_offset = IdxT{0};
      projection_ws.meta.projection.n_proj_components = total_proj_components;
      dataset_proj.N = total_proj_components;
      dataset_proj.n_sampled_cols = total_proj_components;
      launch_batched_training_random_matrix_dense_kernel<DataT, LabelT, IdxT>(
        projection_ws.pointers, projection_ws.meta, builder_stream);
      recordGeneratedNnzDiagnostics(work_items, total_proj_components, false);
      launch_batched_training_projection_kernel<DataT, LabelT, IdxT>(
        projection_ws.pointers, projection_ws.meta, builder_stream);
      launch_batched_training_quantile_sampling_kernel<DataT, LabelT, IdxT>(
        projection_ws.pointers,
        projection_ws.meta,
        projection_ws.d_quantile_indices_storage.data(),
        static_cast<IdxT>(params.max_n_bins),
        static_cast<IdxT>(params.min_samples_leaf),
        builder_stream);
      for (IdxT c = 0; c < total_proj_components; c += n_blks_for_cols) {
        RAFT_CUDA_TRY(cudaMemsetAsync(done_count,
                                      0,
                                      sizeof(int) * params.max_batch_size * n_blks_for_cols,
                                      builder_stream));
        computeSplit(c, c, dataset_proj, n_blocks_dimx, n_large_nodes, work_items.size());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
    } else {
      for (IdxT c = 0; c < total_proj_components; c += projection_tile_size) {
        auto current_tile_size = std::min<IdxT>(projection_tile_size, total_proj_components - c);
        projection_ws.meta.projection.n_proj_components = current_tile_size;
        dataset_proj.N = current_tile_size;
        dataset_proj.n_sampled_cols = current_tile_size;
        projection_ws.meta.generation_projection_offset = c;
        RAFT_CUDA_TRY(cudaMemsetAsync(done_count,
                                      0,
                                      sizeof(int) * params.max_batch_size * n_blks_for_cols,
                                      builder_stream));
        launch_batched_training_random_matrix_sparse_kernel<DataT, LabelT, IdxT>(
          projection_ws.pointers, projection_ws.meta, builder_stream);
        recordGeneratedNnzDiagnostics(work_items, current_tile_size, true);
        launch_batched_training_projection_kernel<DataT, LabelT, IdxT>(
          projection_ws.pointers, projection_ws.meta, builder_stream);
        launch_batched_training_quantile_sampling_kernel<DataT, LabelT, IdxT>(
          projection_ws.pointers,
          projection_ws.meta,
          projection_ws.d_quantile_indices_storage.data(),
          static_cast<IdxT>(params.max_n_bins),
          static_cast<IdxT>(params.min_samples_leaf),
          builder_stream);
        computeSplit(IdxT{0}, c, dataset_proj, n_blocks_dimx, n_large_nodes, work_items.size());
        launch_capture_best_training_projection_kernel<DataT, LabelT, IdxT>(
          projection_ws.pointers, projection_ws.meta, splits, builder_stream);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
    }
    stats.t_kernels +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_kernel).count();
    if (do_timing) { RAFT_CUDA_TRY(cudaEventRecord(ev_kernel_stop, builder_stream)); }
    if (do_timing) {
      RAFT_CUDA_TRY(cudaEventSynchronize(ev_kernel_stop));
      float stage_ms = 0.0f;
      RAFT_CUDA_TRY(cudaEventElapsedTime(&stage_ms, ev_kernel_start, ev_kernel_stop));
      stats.t_compute_split_kernel += static_cast<double>(stage_ms);
    }

    if (!use_dense_generation) {
      projection_ws.meta.generation_fixed_capacity = false;
      projection_ws.meta.projection.n_proj_components = IdxT{1};
      dataset_proj.N = IdxT{1};
      dataset_proj.n_sampled_cols = IdxT{1};
      launch_restore_best_training_projection_kernel<DataT, LabelT, IdxT>(
        projection_ws.pointers, projection_ws.meta, splits, builder_stream);
      launch_batched_training_projection_kernel<DataT, LabelT, IdxT>(
        projection_ws.pointers, projection_ws.meta, builder_stream);
    }

    if (do_timing) {
      RAFT_CUDA_TRY(cudaEventDestroy(ev_kernel_stop));
      RAFT_CUDA_TRY(cudaEventDestroy(ev_kernel_start));
    }

    auto t_kernels_pre_sync = std::chrono::steady_clock::now();
    if (do_timing) {
      RAFT_CUDA_TRY(cudaStreamSynchronize(builder_stream));
      stats.t_kernels_pre_sync +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  t_kernels_pre_sync)
          .count();
    }

    auto t_projection_store_pre_sync = std::chrono::steady_clock::now();
    if (do_timing) {
      stats.t_projection_store_pre_sync +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  t_projection_store_pre_sync)
          .count();
    }

    auto t_projection_store_device = std::chrono::steady_clock::now();
    persist_winning_tree_projection_vectors<DataT, LabelT, IdxT>(
      projection_ws, splits, builder_stream);
    stats.t_projection_store_device +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                t_projection_store_device)
        .count();

    auto t_d2h = std::chrono::steady_clock::now();
    raft::update_host(h_splits, splits, work_items.size(), builder_stream);
    stats.t_d2h +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_d2h).count();

    // create child nodes (or make the current ones leaf)
    t_cpu = std::chrono::steady_clock::now();
    stats.t_split_postprocess_cpu +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cpu).count();

    t_kernel = std::chrono::steady_clock::now();
    raft::common::nvtx::push_range("nodeSplitKernel @sporfbuilder.cuh [batched-levelalgo]");
    if (do_timing) {
      RAFT_CUDA_TRY(cudaEventCreate(&ev_kernel_start));
      RAFT_CUDA_TRY(cudaEventCreate(&ev_kernel_stop));
      RAFT_CUDA_TRY(cudaEventRecord(ev_kernel_start, builder_stream));
    }
    SPORFDT::launchNodeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>(params.min_samples_leaf,
                                                                     params.min_impurity_decrease,
                                                                     dataset_proj,
                                                                     d_work_items,
                                                                     work_items.size(),
                                                                     splits,
                                                                     builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    if (do_timing) { RAFT_CUDA_TRY(cudaEventRecord(ev_kernel_stop, builder_stream)); }

    raft::common::nvtx::pop_range();
    stats.t_kernels +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_kernel).count();
    if (do_timing) {
      RAFT_CUDA_TRY(cudaEventSynchronize(ev_kernel_stop));
      float stage_ms = 0.0f;
      RAFT_CUDA_TRY(cudaEventElapsedTime(&stage_ms, ev_kernel_start, ev_kernel_stop));
      stats.t_node_split_kernel += static_cast<double>(stage_ms);
    }
    if (do_timing) {
      RAFT_CUDA_TRY(cudaEventDestroy(ev_kernel_stop));
      RAFT_CUDA_TRY(cudaEventDestroy(ev_kernel_start));
    }

    t_d2h = std::chrono::steady_clock::now();
    raft::update_host(h_splits, splits, work_items.size(), builder_stream);
    handle.sync_stream(builder_stream);
    stats.t_node_split_d2h_sync +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_d2h).count();
    stats.t_d2h +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_d2h).count();

    return std::make_tuple(h_splits, work_items.size());
  }

  auto computeSplitSmemSize()
  {
    size_t smem_size_1 =
      params.max_n_bins * dataset.num_outputs * sizeof(BinT) +  // shared_histogram size
      params.max_n_bins * sizeof(DataT) +                       // shared_quantiles size
      sizeof(int);                                              // shared_done size
    // Extra room for alignment (see alignPointer in
    // computeSplitKernel)
    smem_size_1 += sizeof(DataT) + 3 * sizeof(int);
    // Calculate the shared memory needed for evalBestSplit
    size_t smem_size_2 = raft::ceildiv(TPB_DEFAULT, raft::WarpSize) * sizeof(SplitT);
    // Pick the max of two
    auto available_smem = handle.get_device_properties().sharedMemPerBlock;
    size_t smem_size    = std::max(smem_size_1, smem_size_2);
    ASSERT(available_smem >= smem_size, "Not enough shared memory. Consider reducing max_n_bins.");

    // printf( "SPORFBuilder::computeSplitSmemSize: smem_size=%ld\n", static_cast<size_t>(smem_size) );

    return smem_size;
  }

  bool useDenseProjectionGeneration(std::size_t n_work_items, IdxT total_proj_components) const
  {
    if (n_work_items == 0 || total_proj_components <= 0 || dataset.N <= 0) { return false; }

    auto dense_len = static_cast<long double>(n_work_items) *
                     static_cast<long double>(total_proj_components) *
                     static_cast<long double>(dataset.N);
    auto dense_required =
      dense_len * static_cast<long double>(sizeof(int) + sizeof(DataT) + sizeof(int) + sizeof(DataT));
    dense_required += static_cast<long double>(dataset.n_sampled_rows) *
                      static_cast<long double>(total_proj_components) *
                      static_cast<long double>(sizeof(DataT));
    dense_required += static_cast<long double>(n_work_items) *
                      static_cast<long double>(total_proj_components) *
                      static_cast<long double>(params.max_n_bins) *
                      static_cast<long double>(sizeof(IdxT));

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));

    auto stream_count = std::max<std::size_t>(std::size_t{1}, handle.get_stream_pool_size());
    constexpr long double dense_total_fraction = 0.02L;
    constexpr long double dense_free_fraction  = 0.05L;
    constexpr long double max_dense_budget     = 512.0L * 1024.0L * 1024.0L;
    auto total_budget = static_cast<long double>(total_bytes) * dense_total_fraction /
                        static_cast<long double>(stream_count);
    auto free_budget = static_cast<long double>(free_bytes) * dense_free_fraction /
                       static_cast<long double>(stream_count);
    auto budget = std::min({total_budget, free_budget, max_dense_budget});
    return dense_required <= budget;
  }

  void recordProjectionAttemptDiagnostics(const std::vector<SPORFDT::NodeWorkItem>& work_items,
                                          IdxT n_attempts)
  {
    if (!hparam_debug.enabled) { return; }

    for (std::size_t i = 0; i < work_items.size(); ++i) {
      ASSERT(n_attempts == hparam_debug.projections_specified,
             "SPORF projection-attempt diagnostic failed: treeid=%d node=%zu depth=%d "
             "attempted=%d specified=%d",
             static_cast<int>(treeid),
             static_cast<std::size_t>(work_items[i].idx),
             static_cast<int>(work_items[i].depth),
             static_cast<int>(n_attempts),
             static_cast<int>(hparam_debug.projections_specified));
      hparam_debug.recordProjectionAttempts(n_attempts);
    }
  }

  void recordGeneratedNnzDiagnostics(const std::vector<SPORFDT::NodeWorkItem>& work_items,
                                     IdxT n_components,
                                     bool fixed_capacity)
  {
    if (!hparam_debug.enabled || work_items.empty() || n_components <= 0) { return; }

    auto indptr_stride = static_cast<std::size_t>(n_components + 1);
    auto indptr_len    = work_items.size() * indptr_stride;
    std::vector<int> h_generation_indptr(indptr_len);
    raft::update_host(h_generation_indptr.data(),
                      projection_ws.pointers.d_generation_indptr,
                      h_generation_indptr.size(),
                      builder_stream);
    handle.sync_stream(builder_stream);

    for (std::size_t work_item_idx = 0; work_item_idx < work_items.size(); ++work_item_idx) {
      if (static_cast<IdxT>(work_items[work_item_idx].instances.count) <
          static_cast<IdxT>(params.min_samples_split)) {
        continue;
      }

      auto const* indptr = h_generation_indptr.data() + work_item_idx * indptr_stride;
      for (IdxT comp = 0; comp < n_components; ++comp) {
        auto nnz = fixed_capacity ? indptr[comp + 1] : indptr[comp + 1] - indptr[comp];
        ASSERT(nnz >= 0,
               "SPORF NNZ diagnostic failed: treeid=%d node=%zu component=%d nnz=%d",
               static_cast<int>(treeid),
               static_cast<std::size_t>(work_items[work_item_idx].idx),
               static_cast<int>(comp),
               static_cast<int>(nnz));
        hparam_debug.recordNnz(static_cast<IdxT>(nnz));
      }
    }
  }

  void printHyperparameterDiagnostics() const
  {
    if (!hparam_debug.enabled) { return; }

    auto nnz_mean =
      hparam_debug.nnz_count > 0
        ? static_cast<double>(hparam_debug.nnz_sum) / static_cast<double>(hparam_debug.nnz_count)
        : 0.0;
    auto nnz_min = hparam_debug.nnz_count > 0 ? hparam_debug.nnz_min : IdxT{0};
    auto attempts_mean = hparam_debug.projection_node_count > 0
                           ? static_cast<double>(hparam_debug.projection_attempt_sum) /
                               static_cast<double>(hparam_debug.projection_node_count)
                           : 0.0;
    auto attempts_min =
      hparam_debug.projection_node_count > 0 ? hparam_debug.projection_attempt_min : IdxT{0};

    auto sorted_nnz = hparam_debug.nnz_values;
    std::sort(sorted_nnz.begin(), sorted_nnz.end());
    auto sorted_attempts = hparam_debug.projection_attempt_values;
    std::sort(sorted_attempts.begin(), sorted_attempts.end());
    auto quantile = [](const std::vector<IdxT>& values, double q) {
      if (values.empty()) { return IdxT{0}; }
      auto idx = static_cast<std::size_t>(q * static_cast<double>(values.size() - 1));
      return values[idx];
    };

    std::cout << "SPORF hyperparameter diagnostics: treeid=" << treeid
              << ", density_specified=" << hparam_debug.density_specified
              << ", density_used=" << hparam_debug.density_used
              << ", expected_nnz=" << hparam_debug.expected_nnz
              << ", nnz_observations=" << hparam_debug.nnz_count
              << ", nnz_min=" << nnz_min
              << ", nnz_q01=" << quantile(sorted_nnz, 0.01)
              << ", nnz_q05=" << quantile(sorted_nnz, 0.05)
              << ", nnz_q25=" << quantile(sorted_nnz, 0.25)
              << ", nnz_q50=" << quantile(sorted_nnz, 0.50)
              << ", nnz_q75=" << quantile(sorted_nnz, 0.75)
              << ", nnz_q95=" << quantile(sorted_nnz, 0.95)
              << ", nnz_q99=" << quantile(sorted_nnz, 0.99)
              << ", nnz_mean=" << nnz_mean
              << ", nnz_max=" << hparam_debug.nnz_max
              << ", max_features_rate_specified=" << hparam_debug.max_features_rate
              << ", projections_per_node_specified=" << hparam_debug.projections_specified
              << ", projection_nodes_observed=" << hparam_debug.projection_node_count
              << ", projection_attempts_min=" << attempts_min
              << ", projection_attempts_q01=" << quantile(sorted_attempts, 0.01)
              << ", projection_attempts_q05=" << quantile(sorted_attempts, 0.05)
              << ", projection_attempts_q25=" << quantile(sorted_attempts, 0.25)
              << ", projection_attempts_q50=" << quantile(sorted_attempts, 0.50)
              << ", projection_attempts_q75=" << quantile(sorted_attempts, 0.75)
              << ", projection_attempts_q95=" << quantile(sorted_attempts, 0.95)
              << ", projection_attempts_q99=" << quantile(sorted_attempts, 0.99)
              << ", projection_attempts_mean=" << attempts_mean
              << ", projection_attempts_max=" << hparam_debug.projection_attempt_max
              << std::endl;
  }

  void computeSplit(IdxT col,
                    IdxT split_col,
                    DatasetT& dataset,
                    size_t n_blocks_dimx,
                    size_t n_large_nodes,
                    size_t n_work_items)
  {
    // if no instances to split, return
    if (n_blocks_dimx == 0) return;
    raft::common::nvtx::range fun_scope("SPORFBuilder::computeSplit @sporfbuilder.cuh [batched-levelalgo]");
    auto n_bins    = params.max_n_bins;
    auto n_classes = dataset.num_outputs;
    // if columns left to be processed lesser than `n_blks_for_cols`, shrink the blocks along dimy
    auto n_blocks_dimy = std::min(n_blks_for_cols, dataset.n_sampled_cols - col);
    // compute required dynamic shared memory
    auto smem_size = computeSplitSmemSize();
    dim3 grid(n_blocks_dimx, n_blocks_dimy, 1);
    // required total length (in bins) of the global segmented histograms over all
    // classes, features and (large)nodes.
    int len_histograms = n_bins * n_classes * n_blocks_dimy * n_large_nodes;
    auto max_len_histograms =
      projection_ws.d_histogram_storage.size() / sizeof(BinT);
    ASSERT(n_blocks_dimx <= static_cast<size_t>(max_blocks_dimx),
           "SPORF computeSplit workload_info overflow: n_blocks_dimx=%zu max_blocks_dimx=%d "
           "n_large_nodes=%zu",
           n_blocks_dimx,
           max_blocks_dimx,
           n_large_nodes);
    ASSERT(n_large_nodes <= static_cast<size_t>(params.max_batch_size),
           "SPORF computeSplit histogram overflow: n_large_nodes=%zu max_batch_size=%d",
           n_large_nodes,
           params.max_batch_size);
    ASSERT(static_cast<size_t>(len_histograms) <= static_cast<size_t>(max_len_histograms),
           "SPORF computeSplit histogram overflow: len_histograms=%d max_len_histograms=%zu "
           "n_bins=%d n_classes=%d n_blocks_dimy=%d n_large_nodes=%zu",
           len_histograms,
           static_cast<size_t>(max_len_histograms),
           n_bins,
           n_classes,
           n_blocks_dimy,
           n_large_nodes);
    RAFT_CUDA_TRY(cudaMemsetAsync(histograms, 0, sizeof(BinT) * len_histograms, builder_stream));
    // create the objective function object
    ObjectiveT objective(dataset.num_outputs, params.min_samples_leaf);
    // call the computeSplitKernel
    // printf( "HELLO FROM %s LINE %d\n", __FILE__, __LINE__ );
    raft::common::nvtx::range kernel_scope("computeSplitKernel @sporfbuilder.cuh [batched-levelalgo]");
    launchComputeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT, ITEMS_PER_THREAD>(histograms,
                                                                    params.max_n_bins,
                                                                    params.min_samples_split,
                                                                    params.min_samples_leaf,
                                                                    dataset,
                                                                    projection_ws.d_quantile_indices_storage.data(),
                                                                    d_work_items,
                                                                    col,
                                                                    split_col,
                                                                    done_count,
                                                                    mutex,
                                                                    splits,
                                                                    objective,
                                                                    workload_info,
                                                                    grid,
                                                                    smem_size,
                                                                    builder_stream);
    // printf( "HELLO FROM %s LINE %d\n", __FILE__, __LINE__ );
  }

  // Set the leaf value predictions in batch
  void SetLeafPredictions(std::shared_ptr<TreeMetaDataNodeT> tree,
                          const std::vector<SPORFDT::InstanceRange>& instance_ranges)
  {
    tree->vector_leaf.resize(tree->sparsetree.size() * dataset.num_outputs);
    ASSERT(tree->sparsetree.size() == instance_ranges.size(),
           "Expected instance range for each node");
    // do this in batch to reduce peak memory usage in extreme cases
    std::size_t max_batch_size = min(std::size_t(100000), tree->sparsetree.size());
    rmm::device_uvector<NodeT> d_tree(max_batch_size, builder_stream);
    rmm::device_uvector<SPORFDT::InstanceRange> d_instance_ranges(max_batch_size, builder_stream);
    rmm::device_uvector<DataT> d_leaves(max_batch_size * dataset.num_outputs, builder_stream);

    ObjectiveT objective(dataset.num_outputs, params.min_samples_leaf);
    for (std::size_t batch_begin = 0; batch_begin < tree->sparsetree.size();
         batch_begin += max_batch_size) {
      std::size_t batch_end  = min(batch_begin + max_batch_size, tree->sparsetree.size());
      std::size_t batch_size = batch_end - batch_begin;
      raft::update_device(
        d_tree.data(), tree->sparsetree.data() + batch_begin, batch_size, builder_stream);
      raft::update_device(
        d_instance_ranges.data(), instance_ranges.data() + batch_begin, batch_size, builder_stream);

      RAFT_CUDA_TRY(
        cudaMemsetAsync(d_leaves.data(), 0, sizeof(DataT) * d_leaves.size(), builder_stream));
      size_t smem_size = sizeof(BinT) * dataset.num_outputs;
      SPORFDT::launchLeafKernel(objective,
                       dataset,
                       d_tree.data(),
                       d_instance_ranges.data(),
                       d_leaves.data(),
                       batch_size,
                       smem_size,
                       builder_stream);
      raft::update_host(tree->vector_leaf.data() + batch_begin * dataset.num_outputs,
                        d_leaves.data(),
                        batch_size * dataset.num_outputs,
                        builder_stream);
    }
    handle.sync_stream(builder_stream);
  }
};  // end Builder

}  // namespace DT
}  // namespace ML
