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

  /**
 * Structure that manages the iterative batched-level training and building of nodes
 * in the host.
 */
template <typename DataT, typename LabelT>
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

  auto Pop()
  {
    std::vector<NodeWorkItem> result;
    result.reserve(std::min(max_batch_size, work_items_.size()));
    while (work_items_.size() > 0 && result.size() < max_batch_size) {
      result.emplace_back(work_items_.front());
      work_items_.pop_front();
    }
    return result;
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

  template <class DataT, class LabelT>
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
