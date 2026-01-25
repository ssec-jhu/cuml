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
#include "batched-levelalgo/sporfbuilder.cuh"
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

  /**
 * Structure that manages the iterative batched-level training and building of nodes
 * in the host.
 */
template <typename DataT, typename LabelT>
class SPORFPredictNodeQueue {
  using NodeT = SparseTreeNode<DataT, LabelT>;
  using TreeMetaDataNodeT = DT::ObliqueTreeMetaDataNode<DataT, LabelT>;
  const SPORFDecisionTreeParams params;
  std::shared_ptr<TreeMetaDataNodeT> tree;
  std::vector<SPORFDT::InstanceRange> node_instances_;
  std::vector<SPORFDT::NodeWorkItem> leaves_;
  std::deque<SPORFDT::NodeWorkItem> work_items_;

 public:
  SPORFPredictNodeQueue(std::shared_ptr<TreeMetaDataNodeT> tree)
    : tree(tree)
  {
    node_instances_.reserve(tree->sparsetree.size());
    node_instances_.emplace_back(SPORFDT::InstanceRange{0, sampled_rows});
    work_items_.emplace_back(SPORFDT::NodeWorkItem{0, 0, node_instances_.back()});
  }

  std::shared_ptr<TreeMetaDataNodeT> GetTree() { return tree; }
  const std::vector<SPORFDT::InstanceRange>& GetInstanceRanges() { return node_instances_; }

  bool HasWork() { return work_items_.size() > 0; }

  auto Pop()
  {
    std::vector<SPORFDT::NodeWorkItem> result;
    result.reserve(std::min(size_t(params.max_batch_size), work_items_.size()));
    while (work_items_.size() > 0 && result.size() < std::size_t(params.max_batch_size)) {
      result.emplace_back(work_items_.front());
      work_items_.pop_front();
    }
    return result;
  }

  template <typename SplitT>
  void Push(const std::vector<SPORFDT::NodeWorkItem>& work_items)
  {
    // Update node queue based on partitioning results
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto item         = work_items[i];
      auto parent         = tree->sparsetree[item.idx];
      auto parent_range = item.instances;

      if(parent.IsLeaf()) {
        leaves_.push_back(item);
        continue;
      }

      auto left_child = tree->sparsetree[parent.LeftChildId()];
      auto right_child = tree->sparsetree[parent.RightChildId()];
      // left
      // Do not add a work item if this child is definitely a leaf
      if (left_child.IsLeaf() == false) {
        work_items_.emplace_back(
          SPORFDT::NodeWorkItem{parent.LeftChildId(), item.depth + 1, SPORFDT::InstanceRange{parent_range.begin, item.nLeft}});
      }

      // right
      // Do not add a work item if this child is definitely a leaf
      if (right_child.IsLeaf() == false) {
        work_items_.emplace_back(
          SPORFDT::NodeWorkItem{parent.RightChildId(), item.depth + 1, SPORFDT::InstanceRange{parent_range.begin + item.nLeft, parent_range.count - item.nLeft}});
      }
    }
  }
};


class SPORFDecisionTree {
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
    int max_batch_size = 1024;
    IdxT n_classes = 0; // Dummy variable, not used in prediction
    rmm::device_uvector<IdxT> row_ids;
    rmm::device_uvector<DataT> d_contiguous;
    rmm::device_uvector<DataT> d_trans;
    rmm::device_uvector<IdxT> smem;

    size_t req_bytes     = n_rows * sizeof(IdxT);
    size_t aligned_bytes = calculateAlignedBytes(req_bytes);
    size_t aligned_elems = aligned_bytes / sizeof(IdxT);
    row_ids.resize(aligned_elems, handle.get_stream());

    req_bytes     = n_rows * n_cols * sizeof(DataT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(DataT);
    d_contiguous.resize(aligned_elems, handle.get_stream());

    req_bytes     = n_rows * sizeof(DataT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(DataT);
    d_trans.resize(aligned_elems, handle.get_stream());

    req_bytes     = max_batch_size * 2 * TPB_DEFAULT * sizeof(IdxT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(IdxT);
    smem.resize(aligned_elems, handle.get_stream());

    thrust::sequence(thrust::cuda::par.on(handle.get_stream()), row_ids.begin(), row_ids.begin() + n_rows, 0);

    Dataset<DataT, LabelT, IdxT> dataset = {d_trans.data(), 0, n_rows, 1, n_rows, 1, row_ids.data(), n_classes};

    raft::common::nvtx::range fun_scope("SPORFBuilder::train @sporfbuilder.cuh [batched-levelalgo]");
    MLCommon::TimerCPU timer;
    SPORFPredictNodeQueue<DataT, LabelT> queue(tree);
    while (queue.HasWork()) {
      auto work_items                      = queue.Pop();

      for (int i = 0; i < work_items.size(); i++) {
        auto& work_item = work_items[i];

        if (tree->sparsetree[work_item.idx].LeftChildId() == -1) continue;

        IdxT colid = 0;
        auto node = tree->sparsetree[work_item.idx];
        auto random_matrix = tree->projection_vectors.at(work_item.idx);

        auto& begin = work_item.instances.begin;
        auto& count = work_item.instances.count;

        raft::matrix::copyRows<DataT, IdxT, size_t>(
          rows,                     // in
          count,                            // number of rows of output matrix
          n_cols,                        // number of columns of output matrix
          d_contiguous.data() + begin * n_cols,    // out
          dataset.row_ids + begin,        // row indices to copy
          count,
          builder_stream,
          false                             // do-row-major
        );
        RAFT_CUDA_TRY(cudaPeekAtLastError());

        paramsRPROJ rproj_params{
          static_cast<int>(count), // number of samples
          n_cols,               // number of features
          1,                       // number of components
          -1.0f,                   // error tolerance (not used)
          false,                   // gaussian or sparse method
          -1.0,                    // auto density (-1: auto-deduction)
          false,                   // not used
          0                        // random seed
        };
        RPROJtransform<DataT>(
          handle,
          d_contiguous.data() + begin * n_cols,
          &random_matrix,
          d_trans.data() + begin,
          &rproj_params
        );

        work_item.nLeft = thrust::count_if(
          thrust::cuda::par.on(handle.get_stream()),
          dataset.row_ids + work_item.instances.begin,
          dataset.row_ids + work_item.instances.begin + work_item.instances.count,
          [=] __device__(IdxT row_id) {
            return dataset.data[row_id * dataset.N + colid] <= node.QueryValue();
          });
        auto split = Split(node.QueryValue(), colid, node.BestMetric(), work_item.nLeft);

        partitionSamples<DataT, LabelT, IdxT, TPB_DEFAULT>(dataset, split, work_item, (char*)(smem.data() + (i * 2 * TPB_DEFAULT * sizeof(IdxT))));
      }

      queue.Push(work_items);
    }

    
    // auto tree = queue.GetTree();
    // this->SetLeafPredictions(tree, queue.GetInstanceRanges());
    // tree->train_time = timer.getElapsedMilliseconds();
    // return tree;
  }

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
