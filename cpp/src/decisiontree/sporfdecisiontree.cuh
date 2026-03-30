/*
 * sporfdecisiontree.cuh
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

#include "batched-levelalgo/builder.cuh"
#include "batched-levelalgo/objectives.cuh"
#include "batched-levelalgo/sporfbuilder.cuh"
#include "batched-levelalgo/kernels/sporf_builder_kernels.cuh"
#include "batched-levelalgo/quantiles.cuh"
#include "treelite_util.h"

#include <cublas_v2.h>

#include <cuml/common/logger.hpp>
#include <cuml/tree/flatnode.h>
#include <cuml/tree/sporfdecisiontree.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/sequence.h>

#include <treelite/c_api.h>
#include <treelite/tree.h>

#include <algorithm>
#include <climits>
#include <iomanip>
#include <locale>
#include <map>
#include <numeric>
#include <random>
#include <vector>

/** check for treelite runtime API errors and assert accordingly */

#define TREELITE_CHECK_RET(call)                                                                 \
  do {                                                                                           \
    int status = call;                                                                           \
    ASSERT(status >= 0, "TREELITE FAIL: call='%s'. Reason:%s\n", #call, TreeliteGetLastError()); \
  } while (0)

namespace ML {

/* namespace tl = treelite;
*/

namespace DT {
  static constexpr int TPB_DEFAULT = 128;
  static constexpr size_t MIN_ROWS_PER_BATCH = 16384;  // heuristic to ensure enough parallelism for GPU kernels

  /**
 * Structure that manages the iterative batched-level training and building of nodes
 * in the host.
 */
template <typename DataT, typename IdxT, typename LabelT>
class SPORFPredictNodeQueue {
  using NodeT = SparseTreeNode<DataT, LabelT>;
  using TreeMetaDataNodeT = DT::ObliqueTreeMetaDataNode<DataT, LabelT>;
  using NodeWorkItem = SPORFDT::NodeWorkItem;
  const TreeMetaDataNodeT& tree;
  size_t max_batch_size;
  std::vector<SPORFDT::InstanceRange> node_instances_;
  std::vector<NodeWorkItem> leaves_;
  std::deque<NodeWorkItem> work_items_;

 public:
  SPORFPredictNodeQueue(const TreeMetaDataNodeT& tree, size_t n_rows, size_t max_batch_size)
    : tree(tree), max_batch_size(max_batch_size)
  {
    node_instances_.reserve(tree.sparsetree.size());
    node_instances_.emplace_back(SPORFDT::InstanceRange{0, n_rows});
    work_items_.emplace_back(NodeWorkItem{0, 0, 0, node_instances_.back()});
  }

  const std::vector<NodeWorkItem>& GetLeaves() { return leaves_; }
  const TreeMetaDataNodeT& GetTree() { return tree; }
  const std::vector<SPORFDT::InstanceRange>& GetInstanceRanges() { return node_instances_; }

  bool HasWork() { return work_items_.size() > 0; }

  
  /**
   * Pop the next prediction batch and build GPU-ready metadata.
   *
   * Returns:
   * 1) `popped`: all dequeued work items (including leaves/empty nodes so queue semantics stay intact).
   * 2) `chunks`: per-(work item segment) metadata for batched projection/count/partition kernels.
   * 3) `block_tasks`: fixed-size row chunks for batched projection/count/partition kernels.
   *
   * Guarantees:
   * - `block_tasks[*].work_item_ids` are batch-local ids into `popped` (`popped_idx`).
   * - Rows for a given `work_item_id` are appended contiguously in block-task order.
   * - `row_ids_ids` are collated positions (`instances.begin + local_row`) into dataset.row_ids
   *   and projected output buffers.
   * - `chunks[*].node_id` is global tree node id used by device kernels to read node projection/split metadata.
   */
  auto Pop()
  {
    size_t total_rows = 0;
    std::vector<NodeWorkItem> popped;
    std::vector<BlockTask<IdxT>> block_tasks;
    std::vector<NodeWorkItemChunk<IdxT>> chunks;
    popped.reserve(std::min(max_batch_size, work_items_.size()));
    block_tasks.reserve(std::min(max_batch_size, work_items_.size()));
    chunks.reserve(std::min(max_batch_size, work_items_.size()));

    while (work_items_.size() > 0 && total_rows < MIN_ROWS_PER_BATCH && popped.size() < max_batch_size) {
      // inline block so we can reuse the `work_item` identifier
      {
        auto work_item = std::move(work_items_.front());
        work_items_.pop_front();

        bool needs_split = !(tree.sparsetree[work_item.idx].IsLeaf() || work_item.instances.count == 0);
        popped.emplace_back(std::move(work_item));

        if (!needs_split) { continue; }  // leaves and empty nodes don't need split tasks
      }
      auto* work_item = &popped.back();

      total_rows += work_item->instances.count;

      for(IdxT threads_left = work_item->instances.count, instances_begin = work_item->instances.begin; threads_left > 0; ) {
        if(block_tasks.empty() || block_tasks.back().count == BLOCK_TASK_SIZE) {
          block_tasks.emplace_back(BlockTask{});
          block_tasks.back().count = 0;
        }

        auto* block_task = &block_tasks.back();
        IdxT thread_count = std::min<IdxT>(BLOCK_TASK_SIZE - block_task->count, threads_left); // remaining thread capacity in current block task

        chunks.emplace_back(NodeWorkItemChunk<IdxT>{
          static_cast<IdxT>(popped.size() - 1),              // work_item_idx
          static_cast<IdxT>(work_item->idx),                 // node_id
          instances_begin,                                   // instances_begin
          thread_count,                                      // instances_count
          static_cast<IdxT>(block_tasks.size() - 1),         // block_task_idx
          block_task->count,                                 // thread_local_begin
          0,                                                 // nLeft, to be updated by projection/count kernel
          0,                                                 // nRight, to be updated by projection/count kernel
          0,                                                 // loff, to be updated by offset scan kernel
          0                                                  // roff, to be updated by offset scan kernel
        });

        auto* chunk = &chunks.back();
        for(IdxT i = chunk->thread_local_begin; i < chunk->thread_local_begin + chunk->instances_count; i++) {
          block_task->work_item_chunk_ids[i] = chunks.size() - 1;
        }

        block_task->count += thread_count;
        instances_begin += thread_count;
        threads_left -= thread_count;
      }
    }

    return std::make_tuple(std::move(popped), std::move(chunks), std::move(block_tasks));
  }

  void Push(const std::vector<NodeWorkItem>& work_items)
  {
    // Update node queue based on partitioning results
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto item         = work_items[i];
      auto parent         = tree.sparsetree[item.idx];
      auto parent_range = item.instances;

      if (parent.IsLeaf()) {
        if (parent_range.count > 0) { leaves_.push_back(item); }
        continue;
      }

      auto nLeft = item.nLeft;
      auto nRight = parent_range.count - nLeft;

      // left
      work_items_.emplace_back(
        NodeWorkItem{static_cast<size_t>(parent.LeftChildId()), item.depth + 1, 0, SPORFDT::InstanceRange{parent_range.begin, nLeft}});

      // right
      work_items_.emplace_back(
        NodeWorkItem{static_cast<size_t>(parent.RightChildId()), item.depth + 1, 0, SPORFDT::InstanceRange{parent_range.begin + nLeft, nRight}});
    }
  }
};

template <typename DataT, typename IdxT>
struct ObliqueNode {
  ProjectionMatrix<DataT, IdxT> projection;
  DataT quesval;
};


template <typename DataT, typename LabelT, typename IdxT>
struct PredictWorkspaceMeta {
  IdxT n_rows;
  IdxT n_cols;
  IdxT max_batch_size;
  IdxT cap_work_items;
  IdxT cap_chunks;
  IdxT cap_block_tasks;
  IdxT cap_prediction_leaves;
  IdxT n_nodes;
  IdxT n_work_items;
  IdxT n_chunks;
  IdxT n_block_tasks;
  IdxT n_leaves;
  IdxT n_vector_leaf;
};

template <typename DataT, typename LabelT, typename IdxT>
struct PredictWorkspacePointers {
  const DataT*                    d_input_col_major;
  IdxT*                           d_row_ids;
  IdxT*                           d_row_ids_scratch;
  DataT*                          d_trans;
  IdxT*                           d_work_item_nleft;
  ObliqueNode<DataT, IdxT>*       d_nodes;
  SPORFDT::NodeWorkItem*          d_work_items;
  NodeWorkItemChunk<IdxT>*        d_chunks;
  BlockTask<IdxT>*                d_block_tasks;
  IdxT*                           d_prediction_leaves;
};

template <typename DataT, typename LabelT, typename IdxT>
struct SPORFDecisionTreeWorkspace {
  rmm::device_uvector<char>      d_workspace;
  PredictWorkspacePointers<DataT, LabelT, IdxT> pointers{};
  PredictWorkspaceMeta<DataT, LabelT, IdxT> meta{};
  rmm::device_uvector<SPORFDT::NodeWorkItem> d_leaves;
  rmm::device_uvector<DataT>     d_vector_leaf;
  rmm::device_uvector<ObliqueNode<DataT, IdxT>> d_nodes_storage;

  SPORFDecisionTreeWorkspace(size_t n_rows_, size_t max_batch_size_, cudaStream_t stream)
    : d_workspace(0, stream), d_leaves(0, stream), d_vector_leaf(0, stream), d_nodes_storage(0, stream)
  {
    auto align_bytes = [](size_t actual_size) {
      constexpr size_t align = 256;
      return raft::alignTo(actual_size, align);
    };
    auto ceil_div = [](size_t a, size_t b) { return (a + b - 1) / b; };

    meta.n_rows = static_cast<IdxT>(n_rows_);
    meta.n_cols = 0;
    meta.max_batch_size = static_cast<IdxT>(max_batch_size_);
    meta.cap_work_items = static_cast<IdxT>(max_batch_size_);
    meta.cap_chunks = static_cast<IdxT>(std::max<size_t>(1, n_rows_));
    meta.cap_block_tasks =
      static_cast<IdxT>(std::max<size_t>(1, ceil_div(n_rows_, static_cast<size_t>(BLOCK_TASK_SIZE))));
    meta.cap_prediction_leaves = static_cast<IdxT>(std::max<size_t>(1, n_rows_));
    meta.n_nodes = 0;
    meta.n_work_items = 0;
    meta.n_chunks = 0;
    meta.n_block_tasks = 0;
    meta.n_leaves = 0;
    meta.n_vector_leaf = 0;

    size_t workspace_bytes = 0;
    workspace_bytes += align_bytes(n_rows_ * sizeof(IdxT));                         // row_ids
    workspace_bytes += align_bytes(n_rows_ * sizeof(IdxT));                         // row_ids_scratch
    workspace_bytes += align_bytes(n_rows_ * sizeof(DataT));                        // projected values
    workspace_bytes += align_bytes(static_cast<size_t>(meta.cap_work_items) * sizeof(IdxT)); // nLeft per work item
    workspace_bytes +=
      align_bytes(static_cast<size_t>(meta.cap_work_items) * sizeof(SPORFDT::NodeWorkItem));
    workspace_bytes += align_bytes(static_cast<size_t>(meta.cap_chunks) * sizeof(NodeWorkItemChunk<IdxT>));
    workspace_bytes += align_bytes(static_cast<size_t>(meta.cap_block_tasks) * sizeof(BlockTask<IdxT>));
    workspace_bytes += align_bytes(static_cast<size_t>(meta.cap_prediction_leaves) * sizeof(IdxT));

    d_workspace.resize(workspace_bytes, stream);

    auto* base = d_workspace.data();
    size_t off = 0;
    auto carve = [&](size_t bytes) {
      auto* p = base + off;
      off += align_bytes(bytes);
      return p;
    };

    pointers.d_input_col_major = nullptr;
    pointers.d_row_ids = reinterpret_cast<IdxT*>(carve(n_rows_ * sizeof(IdxT)));
    pointers.d_row_ids_scratch = reinterpret_cast<IdxT*>(carve(n_rows_ * sizeof(IdxT)));
    pointers.d_trans = reinterpret_cast<DataT*>(carve(n_rows_ * sizeof(DataT)));
    pointers.d_work_item_nleft =
      reinterpret_cast<IdxT*>(carve(static_cast<size_t>(meta.cap_work_items) * sizeof(IdxT)));
    pointers.d_nodes = nullptr;
    pointers.d_work_items = reinterpret_cast<SPORFDT::NodeWorkItem*>(
      carve(static_cast<size_t>(meta.cap_work_items) * sizeof(SPORFDT::NodeWorkItem)));
    pointers.d_chunks = reinterpret_cast<NodeWorkItemChunk<IdxT>*>(
      carve(static_cast<size_t>(meta.cap_chunks) * sizeof(NodeWorkItemChunk<IdxT>)));
    pointers.d_block_tasks = reinterpret_cast<BlockTask<IdxT>*>(
      carve(static_cast<size_t>(meta.cap_block_tasks) * sizeof(BlockTask<IdxT>)));
    pointers.d_prediction_leaves =
      reinterpret_cast<IdxT*>(carve(static_cast<size_t>(meta.cap_prediction_leaves) * sizeof(IdxT)));
  }

  template <typename TreeMetaDataNodeT>
  void reset(const TreeMetaDataNodeT& tree, cudaStream_t stream)
  {
    meta.n_work_items = 0;
    meta.n_chunks = 0;
    meta.n_block_tasks = 0;
    meta.n_nodes = static_cast<IdxT>(tree.sparsetree.size());
    meta.n_leaves = static_cast<IdxT>(tree.sparsetree.size());
    meta.n_vector_leaf = static_cast<IdxT>(tree.vector_leaf.size());

    if (d_leaves.size() < static_cast<size_t>(meta.n_leaves)) {
      d_leaves.resize(static_cast<size_t>(meta.n_leaves), stream);
    }
    if (d_vector_leaf.size() < static_cast<size_t>(meta.n_vector_leaf)) {
      d_vector_leaf.resize(static_cast<size_t>(meta.n_vector_leaf), stream);
    }
    if (d_nodes_storage.size() < static_cast<size_t>(meta.n_nodes)) {
      d_nodes_storage.resize(static_cast<size_t>(meta.n_nodes), stream);
    }

    std::vector<ObliqueNode<DataT, IdxT>> nodes_host;
    nodes_host.reserve(tree.sparsetree.size());
    for (size_t i = 0; i < tree.sparsetree.size(); i++) {
      auto* random_matrix = tree.projection_vectors[i].get();
      ProjectionMatrix<DataT, IdxT> proj{0, nullptr, nullptr, nullptr};
      if (random_matrix) {
        proj.n_proj_components = static_cast<IdxT>(random_matrix->indptr.size()) - 1;
        proj.d_proj_indptr = random_matrix->indptr.data();
        proj.d_proj_indices = random_matrix->indices.data();
        proj.d_proj_coeffs = random_matrix->sparse_data.data();
      }
      nodes_host.push_back(ObliqueNode<DataT, IdxT>{proj, tree.sparsetree[i].QueryValue()});
    }
    if (!nodes_host.empty()) {
      raft::update_device(d_nodes_storage.data(), nodes_host.data(), nodes_host.size(), stream);
    }
    pointers.d_nodes = d_nodes_storage.data();
  }
};


class SPORFDecisionTree {
  using NodeWorkItem = SPORFDT::NodeWorkItem;
  using IdxT = unsigned long;

  template <class... Args>
  using TreeMetaDataNode = typename DT::ObliqueTreeMetaDataNode<Args...>;

 public:
  template <class DataT, class LabelT>
  static std::shared_ptr<TreeMetaDataNode<DataT, LabelT>> fit(
    const raft::handle_t& handle,
    const cudaStream_t s,
    const DataT* data,
    const int ncols,
    const int nrows,
    const LabelT* labels,
    rmm::device_uvector<int>* row_ids,
    int unique_labels,
    SPORFDecisionTreeParams params,
    uint64_t seed,
    const Quantiles<DataT, int>& quantiles,
    int treeid)
  {
    if (params.split_criterion ==
        CRITERION::CRITERION_END) {  // Set default to GINI (classification) or MSE (regression)
      CRITERION default_criterion =
        (std::numeric_limits<LabelT>::is_integer) ? CRITERION::GINI : CRITERION::MSE;
      params.split_criterion = default_criterion;
    }
    using IdxT = int;
    // Dispatch objective
    if (not std::is_same<DataT, LabelT>::value and params.split_criterion == CRITERION::GINI) {
      return SPORFBuilder<GiniObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                      s,
                                                                      treeid,
                                                                      seed,
                                                                      params,
                                                                      data,
                                                                      labels,
                                                                      nrows,
                                                                      ncols,
                                                                      row_ids,
                                                                      unique_labels,
                                                                      quantiles)
        .train();
    } else if (not std::is_same<DataT, LabelT>::value and
               params.split_criterion == CRITERION::ENTROPY) {
      return SPORFBuilder<EntropyObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                         s,
                                                                         treeid,
                                                                         seed,
                                                                         params,
                                                                         data,
                                                                         labels,
                                                                         nrows,
                                                                         ncols,
                                                                         row_ids,
                                                                         unique_labels,
                                                                         quantiles)
        .train();
    } else if (std::is_same<DataT, LabelT>::value and params.split_criterion == CRITERION::MSE) {
      return SPORFBuilder<MSEObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                     s,
                                                                     treeid,
                                                                     seed,
                                                                     params,
                                                                     data,
                                                                     labels,
                                                                     nrows,
                                                                     ncols,
                                                                     row_ids,
                                                                     unique_labels,
                                                                     quantiles)
        .train();
    } else if (std::is_same<DataT, LabelT>::value and
               params.split_criterion == CRITERION::POISSON) {
      return SPORFBuilder<PoissonObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                         s,
                                                                         treeid,
                                                                         seed,
                                                                         params,
                                                                         data,
                                                                         labels,
                                                                         nrows,
                                                                         ncols,
                                                                         row_ids,
                                                                         unique_labels,
                                                                         quantiles)
        .train();
    } else if (std::is_same<DataT, LabelT>::value and params.split_criterion == CRITERION::GAMMA) {
      return SPORFBuilder<GammaObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                       s,
                                                                       treeid,
                                                                       seed,
                                                                       params,
                                                                       data,
                                                                       labels,
                                                                       nrows,
                                                                       ncols,
                                                                       row_ids,
                                                                       unique_labels,
                                                                       quantiles)
        .train();
    } else if (std::is_same<DataT, LabelT>::value and
               params.split_criterion == CRITERION::INVERSE_GAUSSIAN) {
      return SPORFBuilder<InverseGaussianObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                                 s,
                                                                                 treeid,
                                                                                 seed,
                                                                                 params,
                                                                                 data,
                                                                                 labels,
                                                                                 nrows,
                                                                                 ncols,
                                                                                 row_ids,
                                                                                 unique_labels,
                                                                                 quantiles)
        .train();
    } else {
      ASSERT(false, "Unknown split criterion.");
    }
  }

  inline static size_t calculateAlignedBytes(size_t actual_size) {
    constexpr size_t align = 256;  // same alignTo used in builder
    return raft::alignTo(actual_size, align);
  }

  template <typename DataT, typename LabelT, typename IndexT = int>
  static void predict(const raft::handle_t& handle,
                      const TreeMetaDataNode<DataT, LabelT>& tree,
                      size_t max_batch_size,
                      const DataT* rows,
                      std::size_t n_rows,
                      std::size_t n_cols,
                      double scale,
                      DataT* predictions,
                      int num_outputs,
                      rapids_logger::level_enum verbosity);

  template <typename DataT, typename LabelT, typename IndexT = int>
  static void predict(const raft::handle_t& handle,
                      const TreeMetaDataNode<DataT, LabelT>& tree,
                      size_t max_batch_size,
                      const DataT* rows,
                      std::size_t n_rows,
                      std::size_t n_cols,
                      double scale,
                      DataT* predictions,
                      int num_outputs,
                      rapids_logger::level_enum verbosity,
                      cudaStream_t stream);

  template <typename DataT, typename LabelT, typename IndexT = int>
  static void predict(const raft::handle_t& handle,
                      const TreeMetaDataNode<DataT, LabelT>& tree,
                      size_t max_batch_size,
                      const DataT* rows,
                      std::size_t n_rows,
                      std::size_t n_cols,
                      double scale,
                      DataT* predictions,
                      int num_outputs,
                      rapids_logger::level_enum verbosity,
                      SPORFDecisionTreeWorkspace<DataT, LabelT, IndexT>& ws,
                      cudaStream_t stream);

  /*
  template <class DataT, class LabelT>
  static void predict(const raft::handle_t& handle,
                      const TreeMetaDataNode<DataT, LabelT>& tree,
                      const DataT* rows,
                      std::size_t n_rows,
                      std::size_t n_cols,
                      DataT* predictions,
                      int num_outputs,
                      rapids_logger::level_enum verbosity)
  {
    if (verbosity >= rapids_logger::level_enum::off) { default_logger().set_level(verbosity); }
    ASSERT(is_host_ptr(rows) && is_host_ptr(predictions),
           "DT Error: Current impl. expects both input and predictions to be CPU "
           "pointers.\n");

    ASSERT(tree.sparsetree.size() != 0,
           "Cannot predict w/ empty tree, tree size %zu",
           tree.sparsetree.size());

    predict_all(tree, rows, n_rows, n_cols, predictions, num_outputs);
  }

  template <class DataT, class LabelT>
  static void predict_all(const TreeMetaDataNode<DataT, LabelT>& tree,
                          const DataT* rows,
                          std::size_t n_rows,
                          std::size_t n_cols,
                          DataT* preds,
                          int num_outputs)
  {
    for (std::size_t row_id = 0; row_id < n_rows; row_id++) {
      predict_one(&rows[row_id * n_cols], tree, preds + row_id * num_outputs, num_outputs);
    }
  }

  template <class DataT, class LabelT>
  static void predict_one(const DataT* row,
                          const TreeMetaDataNode<DataT, LabelT>& tree,
                          DataT* preds_out,
                          int num_outputs)
  {
    std::size_t idx = 0;
    auto n          = tree.sparsetree[idx];
    while (!n.IsLeaf()) {
      if (row[n.ColumnId()] <= n.QueryValue()) {
        idx = n.LeftChildId();
      } else {
        idx = n.RightChildId();
      }
      n = tree.sparsetree[idx];
    }
    for (int i = 0; i < num_outputs; i++) {
      preds_out[i] += tree.vector_leaf[idx * num_outputs + i];
    }
  }
    */

};  // End DecisionTree Class

template <typename DataT, typename LabelT, typename IdxT>
__global__ void batched_projection_kernel(
  PredictWorkspacePointers<DataT, LabelT, IdxT> pointers,
  PredictWorkspaceMeta<DataT, LabelT, IdxT> meta
);

template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_projection_kernel(
  const PredictWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const PredictWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  cudaStream_t stream
);

template <typename DataT, typename LabelT, typename IdxT, int TPB>
DI void partition_samples2(
  const DT::Dataset<DataT, LabelT, IdxT>& dataset,
  NodeWorkItemChunk<IdxT>* d_chunks,
  BlockTask<IdxT>* d_block_tasks,
  IdxT n_block_tasks,
  char* smem
);

template <typename DataT, typename LabelT, typename IdxT, int TPB>
void launch_partition_samples2(
  const DT::Dataset<DataT, LabelT, IdxT>& dataset,
  NodeWorkItemChunk<IdxT>* d_chunks,
  IdxT n_chunks,
  BlockTask<IdxT>* d_block_tasks,
  IdxT n_block_tasks,
  cudaStream_t stream
);
// Class specializations
/*template tl::Tree<float, float> build_treelite_tree<float, int>(
  const DT::TreeMetaDataNode<float, int>& rf_tree, unsigned int num_class);
template tl::Tree<double, double> build_treelite_tree<double, int>(
  const DT::TreeMetaDataNode<double, int>& rf_tree, unsigned int num_class);
template tl::Tree<float, float> build_treelite_tree<float, float>(
  const DT::TreeMetaDataNode<float, float>& rf_tree, unsigned int num_class);
template tl::Tree<double, double> build_treelite_tree<double, double>(
  const DT::TreeMetaDataNode<double, double>& rf_tree, unsigned int num_class);
*/
}  // End namespace DT

}  // End namespace ML
