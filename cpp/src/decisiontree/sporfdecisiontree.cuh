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
#include "batched-levelalgo/quantiles.cuh"
#include "treelite_util.h"

#include <cuml/common/logger.hpp>
#include <cuml/tree/flatnode.h>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/util/cudart_utils.hpp>

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

class SPORFDecisionTree {
 public:
  template <class DataT, class LabelT>
  static std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> fit(
    const raft::handle_t& handle,
    const cudaStream_t s,
    const DataT* data,
    const int ncols,
    const int nrows,
    const LabelT* labels,
    rmm::device_uvector<int>* row_ids,
    int unique_labels,
    DecisionTreeParams params,
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
      return Builder<GiniObjectiveFunction<DataT, LabelT, IdxT>>(handle,
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
      return Builder<EntropyObjectiveFunction<DataT, LabelT, IdxT>>(handle,
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
      return Builder<MSEObjectiveFunction<DataT, LabelT, IdxT>>(handle,
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
      return Builder<PoissonObjectiveFunction<DataT, LabelT, IdxT>>(handle,
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
      return Builder<GammaObjectiveFunction<DataT, LabelT, IdxT>>(handle,
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
      return Builder<InverseGaussianObjectiveFunction<DataT, LabelT, IdxT>>(handle,
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
                      const DT::TreeMetaDataNode<DataT, LabelT>& tree,
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
  static void predict_all(const DT::TreeMetaDataNode<DataT, LabelT>& tree,
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
                          const DT::TreeMetaDataNode<DataT, LabelT>& tree,
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
