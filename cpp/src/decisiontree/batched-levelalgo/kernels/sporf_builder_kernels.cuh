/*
 * sporf_builder_kernels.cuh
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

#include "../bins.cuh"
#include "../dataset.h"
#include "../split.cuh"
#include "../objectives.cuh"
#include "../quantiles.h"

#include <cuml/common/utils.hpp>

#include <raft/core/handle.hpp>

namespace ML {
namespace SPORFDT {

// The range of instances belonging to a particular node
// This structure refers to a range in the device array dataset.row_ids
struct InstanceRange {
  std::size_t begin;
  std::size_t count;
};


struct NodeWorkItem {
  size_t idx;  // Index of the work item in the tree
  int depth;
  unsigned long nLeft;  // Number of prediction instances in the left child after partitioning
  InstanceRange instances;
};


/**
 * This struct has information about workload of a single threadblock of
 * computeSplit kernels of classification and regression
 */
template <typename IdxT>
struct WorkloadInfo {
  IdxT nodeid;        // Node in the batch on which the threadblock needs to work
  IdxT large_nodeid;  // counts only large nodes (nodes that require more than one block along x-dim
                      // for histogram calculation)
  IdxT offset_blockid;  // Offset threadblock id among all the blocks that are
                        // working on this node
  IdxT num_blocks;      // Total number of blocks that are working on the node
};

/*$ (defined in builder_kernels.cuh)
template <typename SplitT, typename DataT, typename IdxT>
HDI bool SplitNotValid(const SplitT& split,
                       DataT min_impurity_decrease,
                       IdxT min_samples_leaf,
                       std::size_t num_rows)
{
  return split.best_metric_val <= min_impurity_decrease || split.nLeft < min_samples_leaf ||
         (IdxT(num_rows) - split.nLeft) < min_samples_leaf;
}
$*/

/* Returns 'dataset' rounded up to a correctly-aligned pointer of type OutT* */
/*$ (defined in builder_kernels.cuh)
template <typename OutT, typename InT>
DI OutT* alignPointer(InT dataset)
{
  return reinterpret_cast<OutT*>(raft::alignTo(reinterpret_cast<size_t>(dataset), sizeof(OutT)));
}
$*/

template <typename DataT, typename LabelT, typename IdxT, int TPB>
void launchNodeSplitKernel(const IdxT min_samples_leaf,
                           const DataT min_impurity_decrease,
                           const DT::Dataset<DataT, LabelT, IdxT>& dataset,
                           const NodeWorkItem* work_items,
                           const size_t work_items_size,
                           const DT::Split<DataT, IdxT>* splits,
                           cudaStream_t builder_stream);

template <typename DatasetT, typename NodeT, typename ObjectiveT, typename DataT>
void launchLeafKernel(ObjectiveT objective,
                      DatasetT& dataset,
                      const NodeT* tree,
                      const InstanceRange* instance_ranges,
                      DataT* leaves,
                      int batch_size,
                      size_t smem_size,
                      cudaStream_t builder_stream);
// returns the lowest index in `array` whose value is greater or equal to `element`
template <typename DataT, typename IdxT>
HDI IdxT lower_bound(DataT* array, IdxT len, DataT element)
{
  IdxT start = 0;
  IdxT end   = len - 1;
  IdxT mid;
  while (start < end) {
    mid = (start + end) / 2;
    if (array[mid] < element) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
DI void partitionSamples(const DT::Dataset<DataT, LabelT, IdxT>& dataset,
                         const DT::Split<DataT, IdxT>& split,
                         const NodeWorkItem& work_item,
                         char* smem);

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          int IPT,
          typename ObjectiveT,
          typename BinT>
void launchComputeSplitKernel(BinT* histograms,
                              IdxT max_n_bins,
                              IdxT min_samples_split,
                              IdxT min_samples_leaf,
                              const DT::Dataset<DataT, LabelT, IdxT>& dataset,
                              const IdxT* quantile_indices,
                              const NodeWorkItem* work_items,
                              IdxT colStart,
                              IdxT splitColStart,
                              int* done_count,
                              int* mutex,
                              volatile DT::Split<DataT, IdxT>* splits,
                              ObjectiveT& objective,
                              const WorkloadInfo<IdxT>* workload_info,
                              dim3 grid,
                              size_t smem_size,
                              cudaStream_t builder_stream);

}  // namespace SPORFDT
}  // namespace ML
