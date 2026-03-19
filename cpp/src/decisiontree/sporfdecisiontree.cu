// sporfdecisiontree.cu
#include "sporfdecisiontree.cuh"               // class/decl
#include <cub/cub.cuh>
#include <cuml/tree/sporfdecisiontree.hpp>     // public API
#include <thrust/scan.h>
#include <raft/linalg/transpose.cuh>                      // for transposing input data
#include "batched-levelalgo/kernels/sporf_builder_kernels.cuh"
// pull in the device definitions but skip the explicit instantiations to avoid
// clashes with the fixed _DataT/_LabelT aliases in this TU.
#define ML_SPORF_BUILDER_SKIP_EXPLICIT_INSTANTIATIONS
#include "batched-levelalgo/kernels/sporf_builder_kernels_impl.cuh"
#undef ML_SPORF_BUILDER_SKIP_EXPLICIT_INSTANTIATIONS

namespace ML {
namespace DT {

template <typename DataT, typename LabelT, typename IdxT>
__global__ void batched_projection_kernel(const DataT* d_input_col_major,
                                          IdxT n_rows,
                                          IdxT n_cols,
                                          const IdxT* d_row_ids,
                                          Split<DataT, int>* d_out_splits,
                                          const ProjectionMatrix<DataT, int>* d_proj_mats,
                                          IdxT n_proj_mats,
                                          NodeWorkItemChunk<IdxT>* d_chunks,
                                          const BlockTask<int>* d_block_tasks,
                                          IdxT n_block_tasks,
                                          DataT* d_out_col_major,
                                          IdxT out_ld);

template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_projection_kernel(const DataT* d_input_col_major,
                                      IdxT n_rows,
                                      IdxT n_cols,
                                      const IdxT* d_row_ids,
                                      Split<DataT, int>* d_out_splits,
                                      const ProjectionMatrix<DataT, int>* d_proj_mats,
                                      IdxT n_proj_mats,
                                      NodeWorkItemChunk<IdxT>* d_chunks,
                                      const BlockTask<int>* d_block_tasks,
                                      IdxT n_block_tasks,
                                      DataT* d_out_col_major,
                                      IdxT out_ld,
                                      cudaStream_t stream);

template <typename DataT, typename LabelT, typename IdxT>
__global__ void partition_kernel(const Dataset<DataT, LabelT, IdxT> dataset,
                                 DataT split_quesval, IdxT split_colid, DataT split_best_metric_val,
                                 const SPORFDT::NodeWorkItem work_item)
{
    Split<DataT, IdxT> split{
      split_quesval, split_colid, split_best_metric_val, static_cast<IdxT>(work_item.nLeft)};
    extern __shared__ char smem[];
    SPORFDT::partitionSamples<DataT, LabelT, IdxT, TPB_DEFAULT>(dataset, split, work_item, smem);
}

template <typename DataT, typename IdxT>
__global__ void batched_partition_kernel(const IdxT* d_row_ids_in,
                                         IdxT* d_row_ids_out,
                                         const DataT* d_out_col_major,
                                         IdxT out_ld,
                                         const SPORFDT::NodeWorkItem* d_work_items,
                                         const Split<DataT, int>* d_splits,
                                         const NodeWorkItemChunk<IdxT>* d_chunks,
                                         const BlockTask<int>* d_block_tasks,
                                         IdxT n_block_tasks,
                                         IdxT n_proj_mats)
{
  auto block_id = static_cast<IdxT>(blockIdx.x);
  if (block_id >= n_block_tasks) { return; }

  auto tid = static_cast<IdxT>(threadIdx.x);
  auto task = d_block_tasks[block_id];
  bool active = tid < task.count;

  __shared__ int s_is_left[BLOCK_TASK_SIZE];
  __shared__ IdxT s_chunk_begin[BLOCK_TASK_SIZE];
  __shared__ IdxT s_local_left_rank[BLOCK_TASK_SIZE];

  NodeWorkItemChunk<IdxT> chunk{};
  IdxT slot_local = 0;
  IdxT payload_id = 0;
  IdxT node_begin = 0;
  IdxT node_count = 0;
  IdxT split_nleft = 0;
  IdxT row_pos = 0;
  bool lane_valid = false;
  bool is_left = false;
  DataT split_quesval = DataT(0);

  if (active) {
    chunk = d_chunks[task.work_item_chunk_ids[tid]];
    slot_local = tid - chunk.thread_local_begin;
    payload_id = chunk.payload_idx;
    s_chunk_begin[tid] = chunk.thread_local_begin;

    if (slot_local < chunk.instances_count && payload_id < n_proj_mats) {
      auto work_item = d_work_items[chunk.work_item_idx];
      node_begin = static_cast<IdxT>(work_item.instances.begin);
      node_count = static_cast<IdxT>(work_item.instances.count);
      row_pos = chunk.instances_begin + slot_local;

      if (row_pos < node_begin + node_count && row_pos < out_ld) {
        auto split = d_splits[payload_id];
        split_nleft = static_cast<IdxT>(split.nLeft);
        split_quesval = split.quesval;
        DataT projected = d_out_col_major[row_pos];
        is_left = projected <= split_quesval;
        lane_valid = true;
      }
    }
  }

  if (active) {
    s_is_left[tid] = (lane_valid && is_left) ? 1 : 0;
  }
  __syncthreads();

  // Build per-slot left rank once per block (segments are chunk-contiguous by Pop() contract).
  if (tid == 0) {
    IdxT run = 0;
    for (IdxT j = 0; j < task.count; j++) {
      if (j == s_chunk_begin[j]) { run = 0; }
      s_local_left_rank[j] = run;
      run += static_cast<IdxT>(s_is_left[j]);
    }
  }
  __syncthreads();

  if (!active || !lane_valid) { return; }

  IdxT local_left_rank = s_local_left_rank[tid];
  IdxT local_right_rank = slot_local - local_left_rank;
  IdxT dst_pos = is_left ? (node_begin + chunk.loff + local_left_rank)
                         : (node_begin + split_nleft + chunk.roff + local_right_rank);
  if (dst_pos >= node_begin + node_count) { return; }

  d_row_ids_out[dst_pos] = d_row_ids_in[row_pos];
}

template <typename DataT, typename IdxT>
void launch_batched_partition_kernel(const IdxT* d_row_ids_in,
                                     IdxT* d_row_ids_out,
                                     const DataT* d_out_col_major,
                                     IdxT out_ld,
                                     const SPORFDT::NodeWorkItem* d_work_items,
                                     const Split<DataT, int>* d_splits,
                                     const NodeWorkItemChunk<IdxT>* d_chunks,
                                     const BlockTask<int>* d_block_tasks,
                                     IdxT n_block_tasks,
                                     IdxT n_proj_mats,
                                     cudaStream_t stream)
{
  dim3 block(BLOCK_TASK_SIZE);
  dim3 grid(n_block_tasks);
  batched_partition_kernel<DataT, IdxT><<<grid, block, 0, stream>>>(d_row_ids_in,
                                                                     d_row_ids_out,
                                                                     d_out_col_major,
                                                                     out_ld,
                                                                     d_work_items,
                                                                     d_splits,
                                                                     d_chunks,
                                                                     d_block_tasks,
                                                                     n_block_tasks,
                                                                     n_proj_mats);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename IdxT>
__global__ void batched_copyback_kernel(const IdxT* d_row_ids_src,
                                        IdxT* d_row_ids_dst,
                                        const NodeWorkItemChunk<IdxT>* d_chunks,
                                        const BlockTask<int>* d_block_tasks,
                                        IdxT n_block_tasks,
                                        IdxT out_ld)
{
  auto block_id = static_cast<IdxT>(blockIdx.x);
  if (block_id >= n_block_tasks) { return; }

  auto tid = static_cast<IdxT>(threadIdx.x);
  auto task = d_block_tasks[block_id];
  if (tid >= task.count) { return; }

  auto chunk = d_chunks[task.work_item_chunk_ids[tid]];
  auto slot_local = tid - chunk.thread_local_begin;
  if (slot_local >= chunk.instances_count) { return; }

  auto row_pos = chunk.instances_begin + slot_local;
  if (row_pos >= out_ld) { return; }

  d_row_ids_dst[row_pos] = d_row_ids_src[row_pos];
}

template <typename IdxT>
void launch_batched_copyback_kernel(const IdxT* d_row_ids_src,
                                    IdxT* d_row_ids_dst,
                                    const NodeWorkItemChunk<IdxT>* d_chunks,
                                    const BlockTask<int>* d_block_tasks,
                                    IdxT n_block_tasks,
                                    IdxT out_ld,
                                    cudaStream_t stream)
{
  dim3 block(BLOCK_TASK_SIZE);
  dim3 grid(n_block_tasks);
  batched_copyback_kernel<IdxT><<<grid, block, 0, stream>>>(d_row_ids_src,
                                                             d_row_ids_dst,
                                                             d_chunks,
                                                             d_block_tasks,
                                                             n_block_tasks,
                                                             out_ld);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename DataT, typename IdxT>
__global__ void accumulate_predictions_by_row_kernel(const IdxT* d_prediction_leaves,
                                                     const DataT* d_vector_leaf,
                                                     IdxT n_rows,
                                                     int num_outputs,
                                                     double scale,
                                                     DataT* predictions)
{
  auto row = static_cast<IdxT>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n_rows) { return; }

  auto leaf_id = d_prediction_leaves[row];
  auto pred_base = row * static_cast<IdxT>(num_outputs);
  auto leaf_base = leaf_id * static_cast<IdxT>(num_outputs);
  DataT scale_t = static_cast<DataT>(scale);

  for (int j = 0; j < num_outputs; j++) {
    predictions[pred_base + static_cast<IdxT>(j)] += d_vector_leaf[leaf_base + static_cast<IdxT>(j)] * scale_t;
  }
}

template <typename IdxT>
__global__ void pack_chunk_scan_inputs(const NodeWorkItemChunk<IdxT>* d_chunks,
                                       IdxT n_chunks,
                                       IdxT* d_work_item_ids,
                                       IdxT* d_nleft,
                                       IdxT* d_nright)
{
  auto i = static_cast<IdxT>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_chunks) { return; }
  auto c = d_chunks[i];
  d_work_item_ids[i] = c.work_item_idx;
  d_nleft[i]         = c.nLeft;
  d_nright[i]        = c.nRight;
}

template <typename IdxT>
__global__ void unpack_chunk_scan_outputs(NodeWorkItemChunk<IdxT>* d_chunks,
                                          IdxT n_chunks,
                                          const IdxT* d_loff,
                                          const IdxT* d_roff)
{
  auto i = static_cast<IdxT>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_chunks) { return; }
  d_chunks[i].loff = d_loff[i];
  d_chunks[i].roff = d_roff[i];
}

// Template definition moved from the header
template <typename DataT, typename LabelT, typename IdxT>
void SPORFDecisionTree::predict(const raft::handle_t& handle,
                                const TreeMetaDataNode<DataT, LabelT>& tree,
                                std::size_t max_batch_size,
                                const DataT* rows, // input data in column-major format, device pointer
                                std::size_t n_rows,
                                std::size_t n_cols,
                                double scale,
                                DataT* predictions,
                                int num_outputs,
                                rapids_logger::level_enum verbosity)
  {
    RAFT_CUDA_TRY(cudaSetDevice(handle.get_device()));
    auto stream = handle.get_stream();

    IdxT n_classes = 0; // Dummy variable, not used in prediction
    rmm::device_uvector<IdxT> d_row_ids(0, stream);
    rmm::device_uvector<IdxT> d_row_ids_scratch(0, stream);
    rmm::device_uvector<DataT> d_trans(0, stream);

    size_t req_bytes     = n_rows * sizeof(IdxT);
    size_t aligned_bytes = calculateAlignedBytes(req_bytes);
    size_t aligned_elems = aligned_bytes / sizeof(IdxT);
    d_row_ids.resize(aligned_elems, stream);
    d_row_ids_scratch.resize(aligned_elems, stream);

    req_bytes     = n_rows * sizeof(DataT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(DataT);
    d_trans.resize(aligned_elems, stream);

    rmm::device_uvector<SPORFDT::NodeWorkItem>         d_work_items(0, stream);
    rmm::device_uvector<ProjectionMatrix<DataT, int>>  d_projection_matrices(0, stream);
    rmm::device_uvector<Split<DataT, int>>             d_splits(0, stream);
    rmm::device_uvector<NodeWorkItemChunk<IdxT>>       d_chunks(0, stream);
    rmm::device_uvector<BlockTask<int>>                d_block_tasks(0, stream);
    rmm::device_uvector<IdxT>                          d_chunk_work_item_ids(0, stream);
    rmm::device_uvector<IdxT>                          d_chunk_nleft(0, stream);
    rmm::device_uvector<IdxT>                          d_chunk_nright(0, stream);
    rmm::device_uvector<IdxT>                          d_chunk_loff(0, stream);
    rmm::device_uvector<IdxT>                          d_chunk_roff(0, stream);

    // TODO: revisit these heuristics
    req_bytes     = n_rows * sizeof(SPORFDT::NodeWorkItem); // heuristic for max work items needed at once
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(SPORFDT::NodeWorkItem);
    d_work_items.resize(aligned_elems, stream);

    req_bytes     = (n_rows / 2) * sizeof(ProjectionMatrix<DataT, int>); // heuristic for max projection matrices needed at once
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(ProjectionMatrix<DataT, int>);
    d_projection_matrices.resize(aligned_elems, stream);

    req_bytes     = (n_rows / 2) * sizeof(Split<DataT, int>); // heuristic for max splits needed at once
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(Split<DataT, int>);
    d_splits.resize(aligned_elems, stream);

    req_bytes     = n_rows * sizeof(NodeWorkItemChunk<IdxT>); // heuristic for max chunks needed at once
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(NodeWorkItemChunk<IdxT>);
    d_chunks.resize(aligned_elems, stream);

    req_bytes     = n_rows * sizeof(IdxT); // temp storage for segmented scans over chunks
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(IdxT);
    d_chunk_work_item_ids.resize(aligned_elems, stream);
    d_chunk_nleft.resize(aligned_elems, stream);
    d_chunk_nright.resize(aligned_elems, stream);
    d_chunk_loff.resize(aligned_elems, stream);
    d_chunk_roff.resize(aligned_elems, stream);

    req_bytes     = (n_rows / 2) * sizeof(BlockTask<int>); // heuristic for max block tasks needed at once
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(BlockTask<int>);
    d_block_tasks.resize(aligned_elems, stream);

    thrust::sequence(thrust::cuda::par.on(stream), d_row_ids.begin(), d_row_ids.begin() + n_rows, 0);
    Dataset<DataT, LabelT, IdxT> dataset = {
      d_trans.data(),         // projected data (single column)
      nullptr,                // labels (unused in predict)
      static_cast<IdxT>(n_rows),
      static_cast<IdxT>(1),   // N = number of projected columns
      static_cast<IdxT>(n_rows),
      static_cast<IdxT>(1),
      d_row_ids.data(),
      n_classes
    };

    MLCommon::TimerCPU timer;
    SPORFPredictNodeQueue<DataT, int, LabelT> queue(tree, n_rows, max_batch_size);
    while (queue.HasWork()) {
      auto [work_items, projection_matrices, splits, chunks, block_tasks] = queue.Pop();
      raft::update_device(d_work_items.data(), work_items.data(), work_items.size(), stream);
      raft::update_device(d_projection_matrices.data(), projection_matrices.data(), projection_matrices.size(), stream);
      raft::update_device(d_splits.data(), splits.data(), splits.size(), stream);
      raft::update_device(d_chunks.data(), chunks.data(), chunks.size(), stream);
      raft::update_device(d_block_tasks.data(), block_tasks.data(), block_tasks.size(), stream);

      if (!block_tasks.empty()) {
        launch_batched_projection_kernel<DataT, LabelT, IdxT>(rows,
                                                              static_cast<IdxT>(n_rows),
                                                              static_cast<IdxT>(n_cols),
                                                              dataset.row_ids,
                                                              d_splits.data(),
                                                              d_projection_matrices.data(),
                                                              static_cast<IdxT>(projection_matrices.size()),
                                                              d_chunks.data(),
                                                              d_block_tasks.data(),
                                                              static_cast<IdxT>(block_tasks.size()),
                                                              d_trans.data(),
                                                              static_cast<IdxT>(dataset.M),
                                                              stream);
      }

      if (!chunks.empty()) {
        constexpr int TPB_SCAN = 256;
        auto n_chunks = static_cast<IdxT>(chunks.size());
        dim3 block(TPB_SCAN);
        dim3 grid((n_chunks + TPB_SCAN - 1) / TPB_SCAN);

        pack_chunk_scan_inputs<IdxT><<<grid, block, 0, stream>>>(d_chunks.data(),
                                                                  n_chunks,
                                                                  d_chunk_work_item_ids.data(),
                                                                  d_chunk_nleft.data(),
                                                                  d_chunk_nright.data());
        RAFT_CUDA_TRY(cudaPeekAtLastError());

        thrust::exclusive_scan_by_key(thrust::cuda::par.on(stream),
                                      d_chunk_work_item_ids.begin(),
                                      d_chunk_work_item_ids.begin() + n_chunks,
                                      d_chunk_nleft.begin(),
                                      d_chunk_loff.begin());
        thrust::exclusive_scan_by_key(thrust::cuda::par.on(stream),
                                      d_chunk_work_item_ids.begin(),
                                      d_chunk_work_item_ids.begin() + n_chunks,
                                      d_chunk_nright.begin(),
                                      d_chunk_roff.begin());

        unpack_chunk_scan_outputs<IdxT><<<grid, block, 0, stream>>>(d_chunks.data(),
                                                                     n_chunks,
                                                                     d_chunk_loff.data(),
                                                                     d_chunk_roff.data());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }

      if (!block_tasks.empty()) {
        launch_batched_partition_kernel<DataT, IdxT>(d_row_ids.data(),
                                                     d_row_ids_scratch.data(),
                                                     d_trans.data(),
                                                     static_cast<IdxT>(dataset.M),
                                                     d_work_items.data(),
                                                     d_splits.data(),
                                                     d_chunks.data(),
                                                     d_block_tasks.data(),
                                                     static_cast<IdxT>(block_tasks.size()),
                                                     static_cast<IdxT>(projection_matrices.size()),
                                                     stream);
        launch_batched_copyback_kernel<IdxT>(d_row_ids_scratch.data(),
                                             d_row_ids.data(),
                                             d_chunks.data(),
                                             d_block_tasks.data(),
                                             static_cast<IdxT>(block_tasks.size()),
                                             static_cast<IdxT>(dataset.M),
                                             stream);
      }

      if (!splits.empty()) {
        raft::update_host(splits.data(), d_splits.data(), splits.size(), stream);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      }

      // Split entries are ordered by the same non-leaf/non-empty visitation order as work_items.
      size_t split_idx = 0;
      for (size_t i = 0; i < work_items.size(); i++) {
        auto& work_item = work_items[i];
        auto node = tree.sparsetree[work_item.idx];
        if (node.IsLeaf() || work_item.instances.count == 0) { continue; }

        work_item.nLeft = static_cast<decltype(work_item.nLeft)>(splits[split_idx].nLeft);
        split_idx++;
      }

      queue.Push(work_items);
    }

    const auto& leaves = queue.GetLeaves();
    auto leaves_size = static_cast<IdxT>(leaves.size());

    rmm::device_uvector<NodeWorkItem> d_leaves(leaves_size, stream);
    raft::update_device(d_leaves.data(), leaves.data(), leaves_size, stream);

    rmm::device_uvector<DataT> d_vector_leaf(tree.vector_leaf.size(), stream);
    raft::update_device(d_vector_leaf.data(), tree.vector_leaf.data(), tree.vector_leaf.size(), stream);

    // RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    rmm::device_uvector<IdxT> d_prediction_leaves(n_rows, stream);

    auto d_leaves_ptr = d_leaves.data();
    auto d_vector_leaf_ptr = d_vector_leaf.data();
    auto d_prediction_leaves_ptr = d_prediction_leaves.data();

    // Pass 1: build row -> leaf mapping.
    thrust::for_each(
      thrust::cuda::par.on(stream),
      thrust::make_counting_iterator<IdxT>(0),
      thrust::make_counting_iterator<IdxT>(leaves_size),
      [=] __device__(IdxT l) {
        auto item = d_leaves_ptr[l];
        for (IdxT i = item.instances.begin; i < item.instances.begin + item.instances.count; i++) {
          IdxT row_id = dataset.row_ids[i];
          d_prediction_leaves_ptr[row_id] = item.idx; // Store the leaf index for each row
        }
      });

    // Pass 2: row-parallel prediction accumulation for coalesced output writes.
    {
      constexpr int TPB = 256;
      dim3 block(TPB);
      dim3 grid((static_cast<IdxT>(n_rows) + TPB - 1) / TPB);
      accumulate_predictions_by_row_kernel<DataT, IdxT><<<grid, block, 0, stream>>>(
        d_prediction_leaves_ptr,
        d_vector_leaf_ptr,
        static_cast<IdxT>(n_rows),
        num_outputs,
        scale,
        predictions);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    // RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

// Explicit instantiations (match the combos you need)
template void SPORFDecisionTree::predict<float, int, int>(const raft::handle_t&, const TreeMetaDataNode<float,int>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<double, int, int>(const raft::handle_t&, const TreeMetaDataNode<double,int>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<float, float, int>(const raft::handle_t&, const TreeMetaDataNode<float,float>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<double, double, int>(const raft::handle_t&, const TreeMetaDataNode<double,double>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum);

template <typename DataT, typename LabelT, typename IdxT>
__global__ void batched_projection_kernel(
  const DataT* d_input_col_major,   // global input X, col-major [n_rows x n_cols]
  IdxT n_rows,
  IdxT n_cols,
  const IdxT* d_row_ids,            // collated row-id map
  Split<DataT, int>* d_out_splits,
  const ProjectionMatrix<DataT, int>* d_proj_mats,
  IdxT n_proj_mats,
  NodeWorkItemChunk<IdxT>* d_chunks,
  const BlockTask<int>* d_block_tasks,
  IdxT n_block_tasks,
  DataT* d_out_col_major,           // projected output buffer
  IdxT out_ld                        // usually n_rows
)
{
  auto block_id = static_cast<IdxT>(blockIdx.x);
  if (block_id >= n_block_tasks) { return; }

  auto tid = static_cast<IdxT>(threadIdx.x);
  auto task = d_block_tasks[block_id];
  if (tid >= task.count) { return; }

  auto chunk = d_chunks[task.work_item_chunk_ids[tid]];
  auto row_pos = chunk.instances_begin + (tid - chunk.thread_local_begin);
  auto payload_id = chunk.payload_idx;
  if (row_pos >= out_ld || payload_id >= n_proj_mats) { return; }

  auto row_id = d_row_ids[row_pos];
  if (row_id >= n_rows) { return; }

  auto pm = d_proj_mats[payload_id];
  for (IdxT comp = 0; comp < pm.n_proj_components; comp++) {
    int start = pm.d_proj_indptr[comp];
    int end = pm.d_proj_indptr[comp + 1];
    DataT acc = DataT(0);

    for (int nz = start; nz < end; nz++) {
      auto feat = static_cast<IdxT>(pm.d_proj_indices[nz]);
      if (feat >= n_cols) { continue; }
      auto coeff = pm.d_proj_coeffs[nz];
      acc += d_input_col_major[feat * n_rows + row_id] * coeff;
    }
    d_out_col_major[comp * out_ld + row_pos] = acc;
  }

  __syncthreads(); // ensure all threads in block have finished writing to output buffer before any thread in block can be reused for next task

  if (tid == chunk.thread_local_begin) { // single thread per chunk to do the counting for that chunk's payload/split
    // Assumes SPORFPredictNodeQueue::Pop() guarantees:
    // 1) payload_ids are batch-local ids aligned with projection_matrices and splits,
    // 2) rows for each payload_id are appended contiguously in block-task order,
    // 3) row_ids_ids are collated positions into row_ids/output buffers.
    // See the Pop() contract comment in sporfdecisiontree.cuh.
    // With those guarantees, run-length counting here is linear in task.count.
    IdxT curr_payload = chunk.payload_idx;
    IdxT curr_left_count = 0;

    if(curr_payload < n_proj_mats) {
      DataT curr_split = d_out_splits[curr_payload].quesval;

      for (IdxT j = 0; j < chunk.instances_count; j++) {
        IdxT row_pos_j = chunk.instances_begin + j;
        if (row_pos_j >= out_ld) { continue; }

        // Predict currently projects one component per work item (column 0).
        DataT projected = d_out_col_major[row_pos_j];
        if (projected <= curr_split) { curr_left_count++; }
      }

      if (curr_left_count > 0) {
        atomicAdd(&d_out_splits[curr_payload].nLeft, curr_left_count);
      }
    }

    auto* p_chunk = &d_chunks[task.work_item_chunk_ids[tid]];
    p_chunk->nLeft = curr_left_count; // also update chunk metadata for use in partitioning
    p_chunk->nRight = chunk.instances_count - curr_left_count;
  }
}

template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_projection_kernel(
  const DataT* d_input_col_major,
  IdxT n_rows,
  IdxT n_cols,
  const IdxT* d_row_ids,
  Split<DataT, int>* d_out_splits,
  const ProjectionMatrix<DataT, int>* d_proj_mats,
  IdxT n_proj_mats,
  NodeWorkItemChunk<IdxT>* d_chunks,
  const BlockTask<int>* d_block_tasks,
  IdxT n_block_tasks,
  DataT* d_out_col_major,
  IdxT out_ld,
  cudaStream_t stream
)
{
  constexpr int TPB = BLOCK_TASK_SIZE;  // or 128
  dim3 block(TPB);
  dim3 grid(n_block_tasks);             // one block per BlockTask for first version
  batched_projection_kernel<DataT, LabelT, IdxT><<<grid, block, 0, stream>>>(
    d_input_col_major,
    n_rows,
    n_cols,
    d_row_ids,
    d_out_splits,
    d_proj_mats,
    n_proj_mats,
    d_chunks,
    d_block_tasks,
    n_block_tasks,
    d_out_col_major,
    out_ld
  );
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Partition the samples to left/right nodes based on the best split
 * @return the position of the left child node in the nodes list. However, this
 *         value is valid only for threadIdx.x == 0.
 * @note this should be called by only one block from all participating blocks
 *       'smem' should be at least of size `sizeof(IdxT) * TPB * 2`
 */
template <typename DataT, typename LabelT, typename IdxT, int TPB>
DI void partition_samples2(
  const DT::Dataset<DataT, LabelT, IdxT>& dataset,
  // const SPORFDT::NodeWorkItem* const d_work_items,
  // IdxT n_work_items,
  DT::Split<DataT, int>* d_out_splits,
  NodeWorkItemChunk<IdxT>* d_chunks,
  BlockTask<IdxT>* d_block_tasks,
  IdxT n_block_tasks,
  char* smem
)
{
  typedef cub::BlockScan<int, TPB> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp1, temp2;
  volatile auto* row_ids = reinterpret_cast<volatile IdxT*>(dataset.row_ids);
  // for compaction
  // size_t smemSize  = sizeof(IdxT) * TPB;
  // auto* lcomp      = reinterpret_cast<IdxT*>(smem);
  // auto* rcomp      = reinterpret_cast<IdxT*>(smem + smemSize);
  // auto range_start = work_item.instances.begin;
  // auto range_len   = work_item.instances.count;
  // auto* col        = dataset.data + split.colid * std::size_t(dataset.M);
  // auto loffset = range_start, part = loffset + split.nLeft, roffset = part;
  // auto end  = range_start + range_len;
  // int lflag = 0, rflag = 0, llen = 0, rlen = 0, minlen = 0;
  // auto tid = threadIdx.x;
  // while (loffset < part && roffset < end) {
  //   // find the samples in the left that belong to right and vice-versa
  //   auto loff = loffset + tid, roff = roffset + tid;
  //   // d_trans (dataset.data) is aligned with the current row ordering,
  //   // so index by position within row_ids, not the raw row_id value.
  //   if (llen == minlen) lflag = loff < part ? col[loff] > split.quesval : 0;
  //   if (rlen == minlen) rflag = roff < end ? col[roff] <= split.quesval : 0;
  //   // scan to compute the locations for each 'misfit' in the two partitions
  //   int lidx, ridx;
  //   BlockScanT(temp1).ExclusiveSum(lflag, lidx, llen);
  //   BlockScanT(temp2).ExclusiveSum(rflag, ridx, rlen);
  //   __syncthreads();
  //   minlen = llen < rlen ? llen : rlen;
  //   // compaction to figure out the right locations to swap
  //   if (lflag) lcomp[lidx] = loff;
  //   if (rflag) rcomp[ridx] = roff;
  //   __syncthreads();
  //   // reset the appropriate flags for the longer of the two
  //   if (lidx < minlen) lflag = 0;
  //   if (ridx < minlen) rflag = 0;
  //   if (llen == minlen) loffset += TPB;
  //   if (rlen == minlen) roffset += TPB;
  //   // swap the 'misfit's
  //   if (tid < minlen) {
  //     auto a              = row_ids[lcomp[tid]];
  //     auto b              = row_ids[rcomp[tid]];
  //     row_ids[lcomp[tid]] = b;
  //     row_ids[rcomp[tid]] = a;
  //   }
  // }
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
void launch_partition_samples2(
  const DT::Dataset<DataT, LabelT, IdxT>& dataset,
  // const SPORFDT::NodeWorkItem* const d_work_items,
  // IdxT n_work_items,
  DT::Split<DataT, int>* d_out_splits,
  NodeWorkItemChunk<IdxT>* d_chunks,
  IdxT n_chunks,
  BlockTask<IdxT>* d_block_tasks,
  IdxT n_block_tasks,
  cudaStream_t stream
)
{
  dim3 block(BLOCK_TASK_SIZE);  // or 128
  dim3 grid(n_chunks); // for first version, one block does the partition
  size_t smem_size = 2 * BLOCK_TASK_SIZE * sizeof(IdxT);
  partition_samples2<DataT, LabelT, IdxT, TPB><<<grid, block, smem_size, stream>>>(
    dataset,
    // d_work_items,
    // n_work_items,
    d_out_splits,
    d_chunks,
    d_block_tasks,
    n_block_tasks,
    nullptr
  );
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace DT
}  // namespace ML
