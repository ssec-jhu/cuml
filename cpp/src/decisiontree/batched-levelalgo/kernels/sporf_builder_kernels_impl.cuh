/*
 * sporf_builder_kernels_impl.cuh
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

/* #include "builder_kernels.cuh" */
#include "sporf_builder_kernels.cuh"

#include <common/grid_sync.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>

#include <cub/cub.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <thrust/binary_search.h>

#include <cstdio>
#include <limits>

namespace ML {
namespace SPORFDT {

static constexpr int TPB_DEFAULT = 128;
// Keep items-per-thread small; when n_bins <= TPB, using >1 causes the extra
// keys to be written outside the valid range.
static constexpr int ITEMS_PER_THREAD = 1;

template <typename IdxT>
static __device__ void record_compute_split_debug(unsigned long long* debug,
                                                  unsigned long long code,
                                                  unsigned long long block_x,
                                                  unsigned long long block_y,
                                                  IdxT nid,
                                                  IdxT large_nid,
                                                  unsigned long long range_start,
                                                  unsigned long long range_len,
                                                  unsigned long long value0,
                                                  unsigned long long value1);


/**
 * @brief Partition the samples to left/right nodes based on the best split
 * @return the position of the left child node in the nodes list. However, this
 *         value is valid only for threadIdx.x == 0.
 * @note this should be called by only one block from all participating blocks
 *       'smem' should be at least of size `sizeof(IdxT) * TPB * 2`
 */
template <typename DataT, typename LabelT, typename IdxT, int TPB>
DI void partitionSamples(const DT::Dataset<DataT, LabelT, IdxT>& dataset,
                         const DT::Split<DataT, IdxT>& split,
                         const NodeWorkItem& work_item,
                         char* smem)
{
  typedef cub::BlockScan<int, TPB> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp1, temp2;
  volatile auto* row_ids = reinterpret_cast<volatile IdxT*>(dataset.row_ids);
  // for compaction
  size_t smemSize  = sizeof(IdxT) * TPB;
  auto* lcomp      = reinterpret_cast<IdxT*>(smem);
  auto* rcomp      = reinterpret_cast<IdxT*>(smem + smemSize);
  auto range_start = work_item.instances.begin;
  auto range_len   = work_item.instances.count;
  auto* col        = dataset.data + split.colid * std::size_t(dataset.n_sampled_rows);
  auto loffset = range_start, part = loffset + split.nLeft, roffset = part;
  auto end  = range_start + range_len;
  int lflag = 0, rflag = 0, llen = 0, rlen = 0, minlen = 0;
  auto tid = threadIdx.x;
  while (loffset < part && roffset < end) {
    // find the samples in the left that belong to right and vice-versa
    auto loff = loffset + tid, roff = roffset + tid;
    // d_trans (dataset.data) is aligned with the current row ordering,
    // so index by position within row_ids, not the raw row_id value.
    if (llen == minlen) lflag = loff < part ? col[loff] > split.quesval : 0;
    if (rlen == minlen) rflag = roff < end ? col[roff] <= split.quesval : 0;
    // scan to compute the locations for each 'misfit' in the two partitions
    int lidx, ridx;
    BlockScanT(temp1).ExclusiveSum(lflag, lidx, llen);
    BlockScanT(temp2).ExclusiveSum(rflag, ridx, rlen);
    __syncthreads();
    minlen = llen < rlen ? llen : rlen;
    // compaction to figure out the right locations to swap
    if (lflag) lcomp[lidx] = loff;
    if (rflag) rcomp[ridx] = roff;
    __syncthreads();
    // reset the appropriate flags for the longer of the two
    if (lidx < minlen) lflag = 0;
    if (ridx < minlen) rflag = 0;
    if (llen == minlen) loffset += TPB;
    if (rlen == minlen) roffset += TPB;
    // swap the 'misfit's
    if (tid < minlen) {
      auto a              = row_ids[lcomp[tid]];
      auto b              = row_ids[rcomp[tid]];
      row_ids[lcomp[tid]] = b;
      row_ids[rcomp[tid]] = a;
    }
  }
}
template <typename DataT, typename LabelT, typename IdxT, int TPB>
static __global__ void nodeSplitKernel(const IdxT max_depth,
                                       const IdxT min_samples_leaf,
                                       const IdxT min_samples_split,
                                       const IdxT max_leaves,
                                       const DataT min_impurity_decrease,
                                       const DT::Dataset<DataT, LabelT, IdxT> dataset,
                                       const NodeWorkItem* work_items,
                                       const DT::Split<DataT, IdxT>* splits)
{
  extern __shared__ char smem[];
  const auto work_item = work_items[blockIdx.x];
  const auto split     = splits[blockIdx.x];
  if (SplitNotValid(
        split, min_impurity_decrease, min_samples_leaf, IdxT(work_item.instances.count))) {
    return;
  }
  partitionSamples<DataT, LabelT, IdxT, TPB>(dataset, split, work_item, (char*)smem);
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
void launchNodeSplitKernel(const IdxT max_depth,
                           const IdxT min_samples_leaf,
                           const IdxT min_samples_split,
                           const IdxT max_leaves,
                           const DataT min_impurity_decrease,
                           const DT::Dataset<DataT, LabelT, IdxT>& dataset,
                           const NodeWorkItem* work_items,
                           const size_t work_items_size,
                           const DT::Split<DataT, IdxT>* splits,
                           cudaStream_t builder_stream)
{
  auto constexpr smem_size = 2 * sizeof(IdxT) * TPB;
  nodeSplitKernel<DataT, LabelT, IdxT, TPB>
    <<<work_items_size, TPB, smem_size, builder_stream>>>(max_depth,
                                                          min_samples_leaf,
                                                          min_samples_split,
                                                          max_leaves,
                                                          min_impurity_decrease,
                                                          dataset,
                                                          work_items,
                                                          splits);
}

template <typename DatasetT, typename NodeT, typename ObjectiveT, typename DataT>
static __global__ void leafKernel(ObjectiveT objective,
                                  DatasetT dataset,
                                  const NodeT* tree,
                                  const InstanceRange* instance_ranges,
                                  DataT* leaves)
{
  using BinT = typename ObjectiveT::BinT;
  extern __shared__ char shared_memory[];
  auto histogram = reinterpret_cast<BinT*>(shared_memory);
  auto node_id   = blockIdx.x;
  auto& node     = tree[node_id];
  auto range     = instance_ranges[node_id];
  if (!node.IsLeaf()) return;
  auto tid = threadIdx.x;
  for (int i = tid; i < dataset.num_outputs; i += blockDim.x) {
    histogram[i] = BinT();
  }
  __syncthreads();
  for (auto i = range.begin + tid; i < range.begin + range.count; i += blockDim.x) {
    auto label = dataset.labels[dataset.row_ids[i]];
    BinT::IncrementHistogram(histogram, 1, 0, label);
  }
  __syncthreads();
  if (tid == 0) {
    ObjectiveT::SetLeafVector(
      histogram, dataset.num_outputs, leaves + dataset.num_outputs * node_id);
  }
}

template <typename DatasetT, typename NodeT, typename ObjectiveT, typename DataT>
void launchLeafKernel(ObjectiveT objective,
                      DatasetT& dataset,
                      const NodeT* tree,
                      const InstanceRange* instance_ranges,
                      DataT* leaves,
                      int batch_size,
                      size_t smem_size,
                      cudaStream_t builder_stream)
{
  int num_blocks = batch_size;
  leafKernel<<<num_blocks, TPB_DEFAULT, smem_size, builder_stream>>>(
    objective, dataset, tree, instance_ranges, leaves);
}

/**
 * @brief For every threadblock, converts the smem pdf-histogram to
 *        cdf-histogram inplace using inclusive block-sum-scan and returns
 *        the total_sum
 * @return The total sum aggregated over the sumscan,
 *         as well as the modified cdf-histogram pointer
 */
template <typename BinT, typename IdxT, int TPB>
DI BinT pdf_to_cdf(BinT* shared_histogram, IdxT n_bins)
{
  // Blockscan instance preparation
  typedef cub::BlockScan<BinT, TPB> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // variable to accumulate aggregate of sumscans of previous iterations
  BinT total_aggregate = BinT();

  for (IdxT tix = threadIdx.x; tix < raft::ceildiv(n_bins, TPB) * TPB; tix += blockDim.x) {
    BinT result;
    BinT block_aggregate;
    BinT element = tix < n_bins ? shared_histogram[tix] : BinT();
    BlockScan(temp_storage).InclusiveSum(element, result, block_aggregate);
    __syncthreads();
    if (tix < n_bins) { shared_histogram[tix] = result + total_aggregate; }
    total_aggregate += block_aggregate;
  }
  // return the total sum
  return total_aggregate;
}

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          int IPT,
          typename ObjectiveT,
          typename BinT>
static __global__ void computeSplitKernel(BinT* histograms,
                                          IdxT max_n_bins,
                                          IdxT max_depth, // remove?
                                          IdxT min_samples_split,
                                          IdxT min_samples_leaf,
                                          IdxT max_leaves, // remove?
                                          const DT::Dataset<DataT, LabelT, IdxT> dataset,
                                          const IdxT* quantile_indices,
                                          const NodeWorkItem* work_items,
                                          IdxT colStart,
                                          const IdxT* colids,
                                          int* done_count,
                                          int* mutex,
                                          volatile DT::Split<DataT, IdxT>* splits,
                                          ObjectiveT objective,
                                          IdxT treeid,
                                          const WorkloadInfo<IdxT>* workload_info,
                                          uint64_t seed,
                                          unsigned long long* debug)
{
  // dynamic shared memory
  extern __shared__ char smem[];

  // Read workload info for this block
  WorkloadInfo<IdxT> workload_info_cta = workload_info[blockIdx.x];
  IdxT nid                             = workload_info_cta.nodeid;
  IdxT large_nid                       = workload_info_cta.large_nodeid;
  const auto work_item                 = work_items[nid];
  auto range_start                     = work_item.instances.begin;
  auto range_len                       = work_item.instances.count;
  bool trace_root_large = blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 &&
                          range_len >= 100000;
  if (trace_root_large) {
    debug[9] = 101;
    debug[10] = static_cast<unsigned long long>(nid);
    debug[11] = static_cast<unsigned long long>(range_start);
    debug[12] = static_cast<unsigned long long>(range_len);
    printf("SPORF computeSplit kernel stage: treeid=%d stage=101 nid=%d range_start=%llu range_len=%llu\n",
           int(treeid),
           int(nid),
           static_cast<unsigned long long>(range_start),
           static_cast<unsigned long long>(range_len));
  }

  if (range_len < min_samples_split) {
    return;
  }

  IdxT offset_blockid = workload_info_cta.offset_blockid;
  IdxT num_blocks     = workload_info_cta.num_blocks;

  // obtaining the feature to test split on
  IdxT colIndex = colStart + blockIdx.y;
  IdxT col;
  if (dataset.n_sampled_cols == dataset.N) {
    col = colIndex;
  } else {
    col           = colids[nid * dataset.n_sampled_cols + colIndex];
  }
  std::size_t col_offset = std::size_t(col) * dataset.n_sampled_rows;
  // IdxT PRINTCOL = 0;


  // getting the n_bins for that feature
  IdxT n_bins = min(max_n_bins, static_cast<IdxT>(floor(range_len / min_samples_leaf)));

  auto end                  = range_start + range_len;
  auto shared_histogram_len = n_bins * objective.NumClasses();
  auto* shared_histogram    = DT::alignPointer<BinT>(smem);
  auto* shared_quantiles    = DT::alignPointer<DataT>(shared_histogram + shared_histogram_len);
  auto* shared_done         = DT::alignPointer<int>(shared_quantiles + n_bins);
  IdxT stride               = blockDim.x * num_blocks;
  IdxT tid                  = threadIdx.x + offset_blockid * blockDim.x;
  if (trace_root_large) {
    debug[9] = 102;
    printf("SPORF computeSplit kernel stage: treeid=%d stage=102 col=%d n_bins=%d num_blocks=%d\n",
           int(treeid),
           int(col),
           int(n_bins),
           int(num_blocks));
  }

  // populating shared memory with initial values
  for (IdxT i = threadIdx.x; i < shared_histogram_len; i += blockDim.x)
    shared_histogram[i] = BinT();
  for (IdxT b = threadIdx.x; b < n_bins; b += blockDim.x) {
    IdxT quantile_index = quantile_indices[(nid * dataset.n_sampled_cols + colIndex) * max_n_bins + b];
    if (quantile_index < 0) { quantile_index = 0; }
    if (quantile_index >= range_len) { quantile_index = range_len - 1; }
    // IdxT pos = range_start + quantile_index;
    shared_quantiles[b] = dataset.data[range_start + quantile_index + col_offset];
  }
  __syncthreads();
  if (trace_root_large) {
    debug[9] = 103;
    printf("SPORF computeSplit kernel stage: treeid=%d stage=103\n", int(treeid));
  }

  using BlockSort = cub::BlockRadixSort<DataT, TPB, IPT>;
  __shared__ typename BlockSort::TempStorage sort_storage;
  DataT keys[IPT];

  // Load shared quantiles into keys
  #pragma unroll
  for (IdxT i = 0; i < IPT; ++i) {
    int idx = i * TPB + threadIdx.x;
    keys[i] = (idx < n_bins) ? shared_quantiles[idx] : std::numeric_limits<DataT>::infinity();
  }

  // sort the quantile values
  BlockSort(sort_storage).Sort(keys);
  // __syncthreads();
  if (trace_root_large) {
    debug[9] = 104;
    printf("SPORF computeSplit kernel stage: treeid=%d stage=104\n", int(treeid));
  }

  // Store sorted quantiles back to shared memory
  #pragma unroll
  for (IdxT i = 0; i < IPT; ++i) {
    int idx = i * TPB + threadIdx.x;
    if (idx < n_bins) shared_quantiles[idx] = keys[i];
  }
  __syncthreads();
  if (trace_root_large) {
    debug[9] = 105;
    printf("SPORF computeSplit kernel stage: treeid=%d stage=105\n", int(treeid));
  }


  // compute pdf shared histogram for all bins for all classes in shared mem

  // Must be 64 bit - can easily grow larger than a 32 bit int
  for (auto i = range_start + tid; i < end; i += stride) {
    // each thread works over a data point and strides to the next
    auto row   = dataset.row_ids[i]; // dataset.row_ids is "node-collated", because that's how partitioning works
    auto data  = dataset.data[i + col_offset]; // dataset.data is also node-collated because the projection step requires contiguous data
    auto label = dataset.labels[row]; // label is raw, original order, so index by row_ids indirection
    if (label < 0 || label >= objective.NumClasses()) {
      record_compute_split_debug(debug, 10, blockIdx.x, blockIdx.y, nid, large_nid,
                                 static_cast<unsigned long long>(range_start),
                                 static_cast<unsigned long long>(range_len),
                                 static_cast<unsigned long long>(row),
                                 static_cast<unsigned long long>(label));
      continue;
    }

    // `start` is lowest index such that data <= shared_quantiles[start]
    IdxT start = lower_bound(shared_quantiles, n_bins, data);
    BinT::IncrementHistogram(shared_histogram, n_bins, start, label);
  }

  // synchronizing above changes across block
  __syncthreads();
  if (trace_root_large) {
    debug[9] = 106;
    printf("SPORF computeSplit kernel stage: treeid=%d stage=106\n", int(treeid));
  }
  if (num_blocks > 1) {
    // update the corresponding global location
    auto histograms_offset =
      ((large_nid * gridDim.y) + blockIdx.y) * max_n_bins * objective.NumClasses();
    for (IdxT i = threadIdx.x; i < shared_histogram_len; i += blockDim.x) {
      BinT::AtomicAdd(histograms + histograms_offset + i, shared_histogram[i]);
    }

    __threadfence();  // for commit guarantee
    __syncthreads();
    if (trace_root_large) {
      debug[9] = 107;
      printf("SPORF computeSplit kernel stage: treeid=%d stage=107\n", int(treeid));
    }

    // last threadblock will go ahead and compute the best split
    bool last = MLCommon::signalDone(
      done_count + nid * gridDim.y + blockIdx.y, num_blocks, offset_blockid == 0, shared_done);
    // if not the last threadblock, exit
    if (!last) return;
    if (trace_root_large) {
      debug[9] = 108;
      printf("SPORF computeSplit kernel stage: treeid=%d stage=108\n", int(treeid));
    }

    // store the complete global histogram in shared memory of last block
    // indexing shenanigans to compact down to n_bins stride from max_n_bins
    for (IdxT i = threadIdx.x; i < shared_histogram_len; i += blockDim.x) {
      IdxT cls = i / n_bins;
      IdxT b = i % n_bins;
      shared_histogram[i] = histograms[histograms_offset + cls * max_n_bins + b];
    }

    __syncthreads();
  }
  if (trace_root_large) {
    debug[9] = 109;
    printf("SPORF computeSplit kernel stage: treeid=%d stage=109\n", int(treeid));
  }

  // PDF to CDF inplace in `shared_histogram`
  for (IdxT c = 0; c < objective.NumClasses(); ++c) {
    // left to right scan operation for scanning
    // "lesser-than-or-equal" counts
    // shared_histogram is laid out with stride = max_n_bins per class
    //BinT total_sum = SPORFDT::pdf_to_cdf<BinT, IdxT, TPB>(shared_histogram + max_n_bins * c, n_bins);
    BinT total_sum = SPORFDT::pdf_to_cdf<BinT, IdxT, TPB>(shared_histogram + n_bins * c, n_bins);
    // now, `shared_histogram[n_bins * c + i]` will have count of datapoints of class `c`
    // that are less than or equal to `shared_quantiles[i]`.
  }

  __syncthreads();
  if (trace_root_large) {
    debug[9] = 110;
    printf("SPORF computeSplit kernel stage: treeid=%d stage=110\n", int(treeid));
  }

  // calculate the best candidate bins (one for each thread in the block) in current feature and
  // corresponding information gain for splitting
  DT::Split<DataT, IdxT> sp =
    objective.Gain(shared_histogram, shared_quantiles, col, range_len, n_bins);
  if (trace_root_large) {
    debug[9] = 111;
    printf("SPORF computeSplit kernel stage: treeid=%d stage=111\n", int(treeid));
  }

  __syncthreads();

  // calculate best bins among candidate bins per feature using warp reduce
  // then atomically update across features to get best split per node
  // (in split[nid])
  sp.evalBestSplit(smem, splits + nid, mutex + nid);
  if (trace_root_large) {
    debug[9] = 112;
    printf("SPORF computeSplit kernel stage: treeid=%d stage=112\n", int(treeid));
  }
}

template <typename IdxT>
static __device__ void record_compute_split_debug(unsigned long long* debug,
                                                  unsigned long long code,
                                                  unsigned long long block_x,
                                                  unsigned long long block_y,
                                                  IdxT nid,
                                                  IdxT large_nid,
                                                  unsigned long long range_start,
                                                  unsigned long long range_len,
                                                  unsigned long long value0,
                                                  unsigned long long value1)
{
  if (atomicCAS(debug, 0ULL, code) == 0ULL) {
    debug[1] = block_x;
    debug[2] = block_y;
    debug[3] = static_cast<unsigned long long>(nid);
    debug[4] = static_cast<unsigned long long>(large_nid);
    debug[5] = range_start;
    debug[6] = range_len;
    debug[7] = value0;
    debug[8] = value1;
  }
}

template <typename DataT, typename LabelT, typename IdxT>
static __global__ void validateComputeSplitInputsKernel(
  const DT::Dataset<DataT, LabelT, IdxT> dataset,
  const IdxT* quantile_indices,
  const NodeWorkItem* work_items,
  IdxT n_work_items,
  IdxT colStart,
  const IdxT* colids,
  const WorkloadInfo<IdxT>* workload_info,
  IdxT n_blocks_dimx,
  IdxT n_large_nodes,
  IdxT max_n_bins,
  IdxT min_samples_split,
  IdxT min_samples_leaf,
  unsigned long long* debug)
{
  auto block_x = static_cast<IdxT>(blockIdx.x);
  if (block_x >= n_blocks_dimx) { return; }

  auto workload_info_cta = workload_info[block_x];
  IdxT nid               = workload_info_cta.nodeid;
  IdxT large_nid         = workload_info_cta.large_nodeid;

  if (nid < 0 || nid >= n_work_items) {
    record_compute_split_debug(debug, 1, blockIdx.x, blockIdx.y, nid, large_nid, 0, 0,
                               static_cast<unsigned long long>(n_work_items), 0);
    return;
  }
  if (workload_info_cta.num_blocks <= 0 ||
      workload_info_cta.offset_blockid < 0 ||
      workload_info_cta.offset_blockid >= workload_info_cta.num_blocks) {
    record_compute_split_debug(debug, 2, blockIdx.x, blockIdx.y, nid, large_nid, 0, 0,
                               static_cast<unsigned long long>(workload_info_cta.offset_blockid),
                               static_cast<unsigned long long>(workload_info_cta.num_blocks));
    return;
  }
  if (workload_info_cta.num_blocks > 1 && (large_nid < 0 || large_nid >= n_large_nodes)) {
    record_compute_split_debug(debug, 3, blockIdx.x, blockIdx.y, nid, large_nid, 0, 0,
                               static_cast<unsigned long long>(n_large_nodes), 0);
    return;
  }

  auto work_item   = work_items[nid];
  auto range_start = static_cast<unsigned long long>(work_item.instances.begin);
  auto range_len   = static_cast<unsigned long long>(work_item.instances.count);
  auto sampled_rows = static_cast<unsigned long long>(dataset.n_sampled_rows);
  if (range_start > sampled_rows || range_len > sampled_rows - range_start) {
    record_compute_split_debug(debug, 4, blockIdx.x, blockIdx.y, nid, large_nid, range_start,
                               range_len, sampled_rows, 0);
    return;
  }

  if (work_item.instances.count < static_cast<std::size_t>(min_samples_split)) { return; }

  IdxT colIndex = colStart + static_cast<IdxT>(blockIdx.y);
  if (colIndex < 0 || colIndex >= dataset.n_sampled_cols) {
    record_compute_split_debug(debug, 5, blockIdx.x, blockIdx.y, nid, large_nid, range_start,
                               range_len, static_cast<unsigned long long>(colIndex),
                               static_cast<unsigned long long>(dataset.n_sampled_cols));
    return;
  }

  IdxT col = dataset.n_sampled_cols == dataset.N
               ? colIndex
               : colids[nid * dataset.n_sampled_cols + colIndex];
  if (col < 0 || col >= dataset.N) {
    record_compute_split_debug(debug, 6, blockIdx.x, blockIdx.y, nid, large_nid, range_start,
                               range_len, static_cast<unsigned long long>(col),
                               static_cast<unsigned long long>(dataset.N));
    return;
  }

  IdxT n_bins =
    min(max_n_bins, static_cast<IdxT>(floor(work_item.instances.count / min_samples_leaf)));
  if (n_bins <= 0 || n_bins > max_n_bins) {
    record_compute_split_debug(debug, 7, blockIdx.x, blockIdx.y, nid, large_nid, range_start,
                               range_len, static_cast<unsigned long long>(n_bins),
                               static_cast<unsigned long long>(max_n_bins));
    return;
  }

  for (IdxT b = threadIdx.x; b < n_bins; b += blockDim.x) {
    IdxT quantile_index = quantile_indices[(nid * dataset.n_sampled_cols + colIndex) * max_n_bins + b];
    if (quantile_index < 0 || static_cast<unsigned long long>(quantile_index) >= range_len) {
      record_compute_split_debug(debug, 8, blockIdx.x, blockIdx.y, nid, large_nid, range_start,
                                 range_len, static_cast<unsigned long long>(b),
                                 static_cast<unsigned long long>(quantile_index));
      return;
    }
  }

  auto offset_blockid = workload_info_cta.offset_blockid;
  auto num_blocks     = workload_info_cta.num_blocks;
  IdxT stride         = blockDim.x * num_blocks;
  IdxT tid            = threadIdx.x + offset_blockid * blockDim.x;
  auto end            = work_item.instances.begin + work_item.instances.count;
  for (auto i = work_item.instances.begin + tid; i < end; i += stride) {
    auto row = dataset.row_ids[i];
    if (row < 0 || row >= dataset.M) {
      record_compute_split_debug(debug, 9, blockIdx.x, blockIdx.y, nid, large_nid, range_start,
                                 range_len, static_cast<unsigned long long>(i),
                                 static_cast<unsigned long long>(row));
      return;
    }
    auto data  = dataset.data[i + static_cast<std::size_t>(col) * dataset.n_sampled_rows];
    auto label = dataset.labels[row];
    if (data != data) {
      record_compute_split_debug(debug, 10, blockIdx.x, blockIdx.y, nid, large_nid, range_start,
                                 range_len, static_cast<unsigned long long>(i),
                                 static_cast<unsigned long long>(row));
      return;
    }
    if (label < 0 || label >= dataset.num_outputs) {
      record_compute_split_debug(debug, 11, blockIdx.x, blockIdx.y, nid, large_nid, range_start,
                                 range_len, static_cast<unsigned long long>(row),
                                 static_cast<unsigned long long>(label));
      return;
    }
  }
}

template <typename DataT, typename LabelT, typename IdxT>
void launchValidateComputeSplitInputsKernel(const DT::Dataset<DataT, LabelT, IdxT>& dataset,
                                            const IdxT* quantile_indices,
                                            const NodeWorkItem* work_items,
                                            IdxT n_work_items,
                                            IdxT colStart,
                                            const IdxT* colids,
                                            const WorkloadInfo<IdxT>* workload_info,
                                            IdxT n_blocks_dimx,
                                            IdxT n_blocks_dimy,
                                            IdxT n_large_nodes,
                                            IdxT max_n_bins,
                                            IdxT min_samples_split,
                                            IdxT min_samples_leaf,
                                            unsigned long long* debug,
                                            cudaStream_t builder_stream)
{
  if (n_blocks_dimx <= 0 || n_blocks_dimy <= 0) { return; }
  dim3 grid(n_blocks_dimx, n_blocks_dimy, 1);
  validateComputeSplitInputsKernel<DataT, LabelT, IdxT>
    <<<grid, 128, 0, builder_stream>>>(dataset,
                                       quantile_indices,
                                       work_items,
                                       n_work_items,
                                       colStart,
                                       colids,
                                       workload_info,
                                       n_blocks_dimx,
                                       n_large_nodes,
                                       max_n_bins,
                                       min_samples_split,
                                       min_samples_leaf,
                                       debug);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          int IPT,
          typename ObjectiveT,
          typename BinT>
void launchComputeSplitKernel(BinT* histograms,
                              IdxT max_n_bins,
                              IdxT max_depth,
                              IdxT min_samples_split,
                              IdxT min_samples_leaf,
                              IdxT max_leaves,
                              const DT::Dataset<DataT, LabelT, IdxT>& dataset,
                              const IdxT* quantile_indices,
                              // const DT::Quantiles<DataT, IdxT>& quantiles,
                              const NodeWorkItem* work_items,
                              IdxT colStart,
                              const IdxT* colids,
                              int* done_count,
                              int* mutex,
                              volatile DT::Split<DataT, IdxT>* splits,
                              ObjectiveT& objective,
                              IdxT treeid,
                              const WorkloadInfo<IdxT>* workload_info,
                              uint64_t seed,
                              unsigned long long* debug,
                              dim3 grid,
                              size_t smem_size,
                              cudaStream_t builder_stream)
{
  computeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT, IPT>
    <<<grid, TPB_DEFAULT, smem_size, builder_stream>>>(histograms,
                                                       max_n_bins,
                                                       max_depth,
                                                       min_samples_split,
                                                       min_samples_leaf,
                                                       max_leaves,
                                                       dataset,
                                                       quantile_indices,
                                                       work_items,
                                                       colStart,
                                                       colids,
                                                       done_count,
                                                       mutex,
                                                       splits,
                                                       objective,
                                                       treeid,
                                                       workload_info,
                                                       seed,
                                                       debug);
}

#ifndef ML_SPORF_BUILDER_SKIP_EXPLICIT_INSTANTIATIONS
template void launchNodeSplitKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT>(
  const _IdxT max_depth,
  const _IdxT min_samples_leaf,
  const _IdxT min_samples_split,
  const _IdxT max_leaves,
  const _DataT min_impurity_decrease,
  const DT::Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const NodeWorkItem* work_items,
  const size_t work_items_size,
  const DT::Split<_DataT, _IdxT>* splits,
  cudaStream_t builder_stream);

template void launchLeafKernel<_DatasetT, _NodeT, _ObjectiveT, _DataT>(
  _ObjectiveT objective,
  _DatasetT& dataset,
  const _NodeT* tree,
  const InstanceRange* instance_ranges,
  _DataT* leaves,
  int batch_size,
  size_t smem_size,
  cudaStream_t builder_stream);

template void launchComputeSplitKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT, ITEMS_PER_THREAD, _ObjectiveT, _BinT>(
  _BinT* histograms,
  _IdxT max_n_bins,
  _IdxT max_depth,
  _IdxT min_samples_split,
  _IdxT min_samples_leaf,
  _IdxT max_leaves,
  const DT::Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const _IdxT* quantile_indices,
  const NodeWorkItem* work_items,
  _IdxT colStart,
  const _IdxT* colids,
  int* done_count,
  int* mutex,
  volatile DT::Split<_DataT, _IdxT>* splits,
  _ObjectiveT& objective,
  _IdxT treeid,
  const WorkloadInfo<_IdxT>* workload_info,
  uint64_t seed,
  unsigned long long* debug,
  dim3 grid,
  size_t smem_size,
  cudaStream_t builder_stream);

template void launchValidateComputeSplitInputsKernel<_DataT, _LabelT, _IdxT>(
  const DT::Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const _IdxT* quantile_indices,
  const NodeWorkItem* work_items,
  _IdxT n_work_items,
  _IdxT colStart,
  const _IdxT* colids,
  const WorkloadInfo<_IdxT>* workload_info,
  _IdxT n_blocks_dimx,
  _IdxT n_blocks_dimy,
  _IdxT n_large_nodes,
  _IdxT max_n_bins,
  _IdxT min_samples_split,
  _IdxT min_samples_leaf,
  unsigned long long* debug,
  cudaStream_t builder_stream);
#endif
}  // namespace SPORFDT
}  // namespace ML
