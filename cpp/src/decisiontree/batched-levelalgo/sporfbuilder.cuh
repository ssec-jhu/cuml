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

#include <cuml/common/pinned_host_vector.hpp>
#include <cuml/tree/decisiontree.hpp>
#include <cuml/tree/sporfdecisiontree.hpp>
#include <cuml/tree/flatnode.h>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/matrix/matrix.cuh>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <deque>
#include <memory>
#include <utility>

namespace ML {
namespace DT {

template <typename DataT, typename LabelT, typename IdxT>
struct SPORFTrainingProjectionWorkspace;
template <typename DataT, typename LabelT, typename IdxT>
struct TrainingProjectionWorkspacePointers;
template <typename DataT, typename LabelT, typename IdxT>
struct TrainingProjectionWorkspaceMeta;
template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_training_projection_kernel(
  const TrainingProjectionWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const TrainingProjectionWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  cudaStream_t stream);
template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_training_random_matrix_bernoulli_kernel(
  const TrainingProjectionWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const TrainingProjectionWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  cudaStream_t stream);
template <typename DataT, typename LabelT, typename IdxT>
void launch_store_winning_tree_projection_vectors_kernel(
  const TrainingProjectionWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const TrainingProjectionWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  const Split<DataT, IdxT>* d_splits,
  std::size_t payload_base_offset,
  cudaStream_t stream);
template <typename DataT, typename LabelT, typename IdxT>
void persist_winning_tree_projection_vectors(
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
  std::shared_ptr<TreeMetaDataNodeT> tree;
  std::vector<SPORFDT::InstanceRange> node_instances_;
  std::deque<SPORFDT::NodeWorkItem> work_items_;
  const size_t MIN_ROWS_PER_BATCH = 65536;  // heuristic to ensure enough parallelism for GPU kernels

 public:
  SPORFNodeQueue(SPORFDecisionTreeParams params,
                 size_t max_nodes,
                 size_t sampled_rows,
                 int num_outputs,
                 IdxT n_features_)
    : params(params), n_features(n_features_), tree(std::make_shared<TreeMetaDataNodeT>())
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

  auto Pop()
  {
    size_t total_rows = 0;
    std::vector<SPORFDT::NodeWorkItem> popped;
    std::vector<DT::BlockTask<IdxT>> projection_block_tasks;
    std::vector<DT::NodeWorkItemChunk<IdxT>> projection_chunks;
    std::vector<DT::BlockTask<IdxT>> projection_matrix_block_tasks;
    std::vector<DT::NodeWorkItemChunk<IdxT>> projection_matrix_chunks;
    popped.reserve(std::min(size_t(params.max_batch_size), work_items_.size()));
    projection_block_tasks.reserve(std::min(size_t(params.max_batch_size), work_items_.size()));
    projection_chunks.reserve(std::min(size_t(params.max_batch_size), work_items_.size()));
    projection_matrix_block_tasks.reserve(std::min(size_t(params.max_batch_size), work_items_.size()));
    projection_matrix_chunks.reserve(std::min(size_t(params.max_batch_size), work_items_.size()));

    while (work_items_.size() > 0 && popped.size() < std::size_t(params.max_batch_size)) {
      popped.emplace_back(work_items_.front());
      work_items_.pop_front();
      total_rows += popped.back().instances.count;

      auto* work_item = &popped.back();
      auto count = static_cast<IdxT>(work_item->instances.count);
      if (count < static_cast<IdxT>(params.min_samples_split)) { continue; }

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
    }
    return std::make_tuple(std::move(popped),
                           std::move(projection_chunks),
                           std::move(projection_block_tasks),
                           std::move(projection_matrix_chunks),
                           std::move(projection_matrix_block_tasks));
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
  /** Memory alignment value */
  const size_t align_value = 512;
  IdxT* d_colids;
  IdxT* h_colids;
  /** rmm device workspace buffer */
  rmm::device_uvector<char> d_buff;
  SPORFTrainingProjectionWorkspace<DataT, LabelT, IdxT>& projection_ws;
  /** pinned host buffer to store the trained nodes */
  ML::pinned_host_vector<char> h_buff;
  std::vector<IdxT> h_quantile_indices;
  rmm::device_uvector<IdxT> d_quantile_indices;
  rmm::device_uvector<DataT> d_trans;

  struct Stats {
    double t_pop;
    double t_push;
    double t_h2d;
    double t_d2h;
    double t_kernels;
    double t_workload_info_cpu;
    double t_quantile_sampling_cpu;
    double t_split_postprocess_cpu;
    double t_projection_store_device;
    double t_tree_projection_finalize;
    double t_leaf_predictions;

    Stats()
      : t_pop(0),
        t_push(0),
        t_h2d(0),
        t_d2h(0),
        t_kernels(0),
        t_workload_info_cpu(0),
        t_quantile_sampling_cpu(0),
        t_split_postprocess_cpu(0),
        t_projection_store_device(0),
        t_tree_projection_finalize(0),
        t_leaf_predictions(0)
    {
    }
  } stats;

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
          SPORFTrainingProjectionWorkspace<DataT, LabelT, IdxT>& projection_ws_)
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
      d_quantile_indices(0, builder_stream),
      d_buff(0, builder_stream),
      d_trans(0, builder_stream),
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

    h_quantile_indices.resize(params.max_batch_size * dataset.n_sampled_cols * params.max_n_bins);

    size_t req_bytes     = size_t(params.max_batch_size) * size_t(dataset.n_sampled_cols) * params.max_n_bins * sizeof(IdxT);
    size_t aligned_bytes = calculateAlignedBytes(req_bytes);
    size_t aligned_elems = aligned_bytes / sizeof(IdxT);
    d_quantile_indices.resize(aligned_elems, builder_stream);

    req_bytes     = size_t(dataset.n_sampled_rows) * size_t(dataset.n_sampled_cols) * sizeof(DataT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(DataT);
    d_trans.resize(aligned_elems, builder_stream);

    projection_ws.ensure_tree_projection_vector_capacity(this->maxNodes(), builder_stream);
    projection_ws.clear_tree_projection_state(builder_stream);

    auto [device_workspace_size, host_workspace_size] = workspaceSize();
    d_buff.resize(device_workspace_size, builder_stream);
    h_buff.resize(host_workspace_size);
    assignWorkspace(d_buff.data(), h_buff.data());
  }

  /**
   * @brief calculates nearest aligned size of input w.r.t an `align_value`.
   *
   * @param[in] actual_size actual size in bytes of input
   * @return aligned size
   */
  size_t calculateAlignedBytes(const size_t actual_size) const
  {
    return raft::alignTo(actual_size, align_value);
  }

  /**
   * @brief returns maximum nodes possible per tree
   * @return maximum nodes possible per tree
   */
  size_t maxNodes() const
  {
    if (params.max_depth < 13) {
      // Start with allocation for a dense tree for depth < 13
      return pow(2, (params.max_depth + 1)) - 1;
    } else {
      // Start with fixed size allocation for depth >= 13
      return 8191;
    }
  }

  /**
   * @brief calculate the workspace size required
   *
   * @return a pair of device workspace and host workspace size requirements
   */
  auto workspaceSize() const
  {
    size_t d_wsize = 0, h_wsize = 0;
    raft::common::nvtx::range fun_scope("SPORFBuilder::workspaceSize @sporfbuilder.cuh [batched-levelalgo]");
    auto max_batch = params.max_batch_size;
    size_t max_len_histograms =
      max_batch * params.max_n_bins * n_blks_for_cols * dataset.num_outputs;

    d_wsize += calculateAlignedBytes(sizeof(IdxT));                               // n_nodes
    d_wsize += calculateAlignedBytes(sizeof(BinT) * max_len_histograms);          // histograms
    d_wsize += calculateAlignedBytes(sizeof(int) * max_batch * n_blks_for_cols);  // done_count
    d_wsize += calculateAlignedBytes(sizeof(int) * max_batch);                    // mutex
    d_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);                 // splits
    d_wsize += calculateAlignedBytes(sizeof(SPORFDT::NodeWorkItem) * max_batch);           // d_work_Items
    d_wsize +=                                                                    // workload_info
      calculateAlignedBytes(sizeof(SPORFDT::WorkloadInfo<IdxT>) * max_blocks_dimx);
    d_wsize += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);  // colids

    // all nodes in the tree
    h_wsize +=  // h_workload_info
      calculateAlignedBytes(sizeof(SPORFDT::WorkloadInfo<IdxT>) * max_blocks_dimx);
    h_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);  // splits
    h_wsize += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);  // colids



    //if( this->builder_stream == handle.get_stream_from_stream_pool(0) )
  // printf( "SPORFBuilder::workspaceSize (%s line %d): d_wsize=%lu h_wsize=%lu\n", __FILE__, __LINE__, d_wsize, h_wsize );


    return std::make_pair(d_wsize, h_wsize);
  }

  /**
   * @brief assign workspace to the current state
   *
   * @param[in] d_wspace device buffer allocated by the user for the workspace.
   *                     Its size should be at least workspaceSize()
   * @param[in] h_wspace pinned host buffer needed to store the learned nodes
   */
  void assignWorkspace(char* d_wspace, char* h_wspace)
  {
    raft::common::nvtx::range fun_scope(
      "SPORFBuilder::assignWorkspace @sporfbuilder.cuh [batched-levelalgo]");
    auto max_batch  = params.max_batch_size;
    auto n_col_blks = n_blks_for_cols;
    size_t max_len_histograms =
      max_batch * (params.max_n_bins) * n_blks_for_cols * dataset.num_outputs;
    // device
    n_nodes = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT));
    histograms = reinterpret_cast<BinT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(BinT) * max_len_histograms);
    done_count = reinterpret_cast<int*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(int) * max_batch * n_col_blks);
    mutex = reinterpret_cast<int*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(int) * max_batch);
    splits = reinterpret_cast<SplitT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
    d_work_items = reinterpret_cast<SPORFDT::NodeWorkItem*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(SPORFDT::NodeWorkItem) * max_batch);
    workload_info = reinterpret_cast<SPORFDT::WorkloadInfo<IdxT>*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(SPORFDT::WorkloadInfo<IdxT>) * max_blocks_dimx);
    d_colids = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);

    RAFT_CUDA_TRY(
      cudaMemsetAsync(done_count, 0, sizeof(int) * max_batch * n_col_blks, builder_stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, builder_stream));

    // host
    h_workload_info = reinterpret_cast<SPORFDT::WorkloadInfo<IdxT>*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(SPORFDT::WorkloadInfo<IdxT>) * max_blocks_dimx);
    h_splits = reinterpret_cast<SplitT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
    h_colids = reinterpret_cast<IdxT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);
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
    SPORFNodeQueue<DataT, LabelT> queue(
      params, this->maxNodes(), dataset.n_sampled_rows, dataset.num_outputs, dataset.N);
    while (queue.HasWork()) {
      auto t_pop = std::chrono::steady_clock::now();
      auto popped_batch                    = queue.Pop();
      stats.t_pop +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pop).count();
      auto& work_items                     = std::get<0>(popped_batch);
      auto& projection_chunks              = std::get<1>(popped_batch);
      auto& projection_block_tasks         = std::get<2>(popped_batch);
      auto& projection_matrix_chunks       = std::get<3>(popped_batch);
      auto& projection_matrix_block_tasks  = std::get<4>(popped_batch);
      auto [splits_host_ptr, splits_count] =
        doSplit(work_items,
                projection_chunks,
                projection_block_tasks,
                projection_matrix_chunks,
                projection_matrix_block_tasks,
                projection_ws);
      auto t_push = std::chrono::steady_clock::now();
      queue.Push(work_items, splits_host_ptr);
      stats.t_push +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_push).count();
    }
    auto tree = queue.GetTree();

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

    std::cout << "SPORFBuilder::train: pop: " << stats.t_pop <<
      " ms, push: " << stats.t_push <<
      " ms, h2d: " << stats.t_h2d <<
      " ms, d2h: " << stats.t_d2h <<
      " ms, kernels: " << stats.t_kernels <<
      " ms, workload_info_cpu: " << stats.t_workload_info_cpu <<
      " ms, quantile_sampling_cpu: " << stats.t_quantile_sampling_cpu <<
      " ms, split_postprocess_cpu: " << stats.t_split_postprocess_cpu <<
      " ms, projection_store_device: " << stats.t_projection_store_device <<
      " ms, tree_projection_finalize: " << stats.t_tree_projection_finalize <<
      " ms, leaf_predictions: " << stats.t_leaf_predictions <<
      " ms" << std::endl;


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
    auto t_cpu = std::chrono::steady_clock::now();
    auto [n_blocks_dimx, n_large_nodes] = this->updateWorkloadInfo(work_items);

    for(IdxT i = 0; i < params.max_batch_size; i++ ) {
      for( IdxT c = 0; c < dataset.n_sampled_cols; c++ ) {
        h_colids[i * dataset.n_sampled_cols + c] = c;
      }
    }

    projection_ws.reset(builder_stream);
    projection_ws.meta.input_n_rows = dataset.M;
    projection_ws.meta.projection.n_rows = dataset.n_sampled_rows;
    projection_ws.meta.projection.n_cols = dataset.N;
    projection_ws.meta.projection.n_work_items = static_cast<IdxT>(work_items.size());
    projection_ws.meta.projection.n_chunks = static_cast<IdxT>(projection_chunks.size());
    projection_ws.meta.projection.n_block_tasks = static_cast<IdxT>(projection_block_tasks.size());
    // In the SPORF builder, `n_sampled_cols` is reused as the random-projection
    // output dimensionality, i.e. the number of projection components per node.
    projection_ws.meta.projection.n_proj_components = dataset.n_sampled_cols;
    projection_ws.meta.n_generation_chunks = static_cast<IdxT>(projection_matrix_chunks.size());
    projection_ws.meta.n_generation_block_tasks =
      static_cast<IdxT>(projection_matrix_block_tasks.size());
    projection_ws.meta.generation_n_features = dataset.N;
    projection_ws.meta.generation_min_samples_split = static_cast<IdxT>(params.min_samples_split);
    projection_ws.meta.generation_density = static_cast<DataT>(params.density);
    projection_ws.meta.generation_random_state =
      static_cast<int>((seed + static_cast<uint64_t>(treeid)) & 0x7fffffffULL);
    projection_ws.pointers.d_input_col_major = dataset.data;
    projection_ws.pointers.d_row_ids = dataset.row_ids;
    projection_ws.pointers.projection.d_trans = d_trans.data();
    projection_ws.ensure_generation_metadata_capacity(projection_ws.meta.n_generation_chunks,
                                                      projection_ws.meta.n_generation_block_tasks,
                                                      builder_stream);
    size_t dense_generation_len = static_cast<size_t>(work_items.size()) *
                                  static_cast<size_t>(std::max<IdxT>(1, dataset.n_sampled_cols)) *
                                  static_cast<size_t>(dataset.N);
    size_t generation_indptr_len = static_cast<size_t>(work_items.size()) *
                                   static_cast<size_t>(std::max<IdxT>(1, dataset.n_sampled_cols) + 1);

    projection_ws.resize_generation_storage(dense_generation_len, generation_indptr_len, builder_stream);

    ASSERT(projection_ws.meta.projection.n_work_items <= projection_ws.meta.projection.cap_work_items,
           "Training projection workspace overflow: work_items");
    ASSERT(projection_ws.meta.projection.n_chunks <= projection_ws.meta.projection.cap_chunks,
           "Training projection workspace overflow: chunks");
    ASSERT(projection_ws.meta.projection.n_block_tasks <= projection_ws.meta.projection.cap_block_tasks,
           "Training projection workspace overflow: block_tasks");
    stats.t_workload_info_cpu +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cpu).count();

    auto t_kernel = std::chrono::steady_clock::now();
    // start fresh on the number of *new* nodes created in this batch
    RAFT_CUDA_TRY(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), builder_stream));
    initSplit<DataT, IdxT, TPB_DEFAULT>(splits, work_items.size(), builder_stream);
    stats.t_kernels +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_kernel).count();

    auto t_h2d = std::chrono::steady_clock::now();
    // get the current set of nodes to be worked upon
    raft::update_device(d_work_items, work_items.data(), work_items.size(), builder_stream);
    raft::update_device(workload_info, h_workload_info, n_blocks_dimx, builder_stream);
    raft::update_device(d_colids, h_colids, work_items.size() * dataset.n_sampled_cols, builder_stream);
    if (!work_items.empty()) {
      raft::update_device(projection_ws.pointers.projection.d_work_items,
                          work_items.data(),
                          projection_ws.meta.projection.n_work_items,
                          builder_stream);
    }
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

    t_kernel = std::chrono::steady_clock::now();
    launch_batched_training_random_matrix_bernoulli_kernel<DataT, LabelT, IdxT>(
      projection_ws.pointers, projection_ws.meta, builder_stream);
    stats.t_kernels +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_kernel).count();

    t_kernel = std::chrono::steady_clock::now();
    launch_batched_training_projection_kernel<DataT, LabelT, IdxT>(
      projection_ws.pointers, projection_ws.meta, builder_stream);
    stats.t_kernels +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_kernel).count();

    t_cpu = std::chrono::steady_clock::now();
    for (size_t i = 0; i < work_items.size(); i++) {
      auto count = work_items[i].instances.count;
      if (count < static_cast<unsigned long>(params.min_samples_split)) continue;

      std::vector<IdxT> universe(count);
      std::iota(universe.begin(), universe.end(), 0);

      for (int c = 0; c < dataset.n_sampled_cols; c++) {
        std::mt19937_64 rng(seed + static_cast<uint64_t>(treeid) + i + c);
        // TODO: randomize the order here!
        std::sample(
          universe.begin(), universe.end(), h_quantile_indices.begin() + (i * dataset.n_sampled_cols * params.max_n_bins) + (c * params.max_n_bins), params.max_n_bins, rng
        );
      }
    }
    stats.t_quantile_sampling_cpu +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cpu).count();

    t_h2d = std::chrono::steady_clock::now();
    raft::update_device(
      d_quantile_indices.data(), h_quantile_indices.data(), h_quantile_indices.size(), builder_stream);
    stats.t_h2d +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_h2d).count();

    dataset_proj.data = d_trans.data();
    t_kernel = std::chrono::steady_clock::now();
    // iterate through a batch of columns (to reduce the memory pressure) and
    // compute the best split at the end
    for (IdxT c = 0; c < dataset.n_sampled_cols; c += n_blks_for_cols) {
      computeSplit(c, dataset_proj, n_blocks_dimx, n_large_nodes);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    stats.t_kernels +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_kernel).count();

    auto t_projection_store_device = std::chrono::steady_clock::now();
    persist_winning_tree_projection_vectors<DataT, LabelT, IdxT>(projection_ws, splits, builder_stream);
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
    SPORFDT::launchNodeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>(params.max_depth,
                                                                     params.min_samples_leaf,
                                                                     params.min_samples_split,
                                                                     params.max_leaves,
                                                                     params.min_impurity_decrease,
                                                                     dataset_proj,
                                                                     d_work_items,
                                                                     work_items.size(),
                                                                     splits,
                                                                     builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    raft::common::nvtx::pop_range();
    stats.t_kernels +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_kernel).count();

    t_d2h = std::chrono::steady_clock::now();
    raft::update_host(h_splits, splits, work_items.size(), builder_stream);
    handle.sync_stream(builder_stream);
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

  void computeSplit(IdxT col, DatasetT& dataset, size_t n_blocks_dimx, size_t n_large_nodes)
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
    RAFT_CUDA_TRY(cudaMemsetAsync(histograms, 0, sizeof(BinT) * len_histograms, builder_stream));
    // create the objective function object
    ObjectiveT objective(dataset.num_outputs, params.min_samples_leaf);
    // call the computeSplitKernel
    // printf( "HELLO FROM %s LINE %d\n", __FILE__, __LINE__ );
    raft::common::nvtx::range kernel_scope("computeSplitKernel @sporfbuilder.cuh [batched-levelalgo]");
    launchComputeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT, ITEMS_PER_THREAD>(histograms,
                                                                    params.max_n_bins,
                                                                    params.max_depth,
                                                                    params.min_samples_split,
                                                                    params.min_samples_leaf,
                                                                    params.max_leaves,
                                                                    dataset,
                                                                    d_quantile_indices.data(),
                                                                    d_work_items,
                                                                    col,
                                                                    d_colids,
                                                                    done_count,
                                                                    mutex,
                                                                    splits,
                                                                    objective,
                                                                    treeid,
                                                                    workload_info,
                                                                    seed,
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
  }
};  // end Builder

}  // namespace DT
}  // namespace ML
