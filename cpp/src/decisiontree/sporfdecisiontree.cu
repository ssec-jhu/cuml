// sporfdecisiontree.cu
#include "sporfdecisiontree.cuh"               // class/decl
#include <cub/cub.cuh>
#include <cuml/tree/sporfdecisiontree.hpp>     // public API
#include <chrono>
#include <tuple>
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
__global__ void batched_projection_kernel(PredictWorkspacePointers<DataT, LabelT, IdxT> pointers,
                                          PredictWorkspaceMeta<DataT, LabelT, IdxT> meta);

template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_projection_kernel(const PredictWorkspacePointers<DataT, LabelT, IdxT>& pointers,
                                      const PredictWorkspaceMeta<DataT, LabelT, IdxT>& meta,
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

template <typename DataT, typename LabelT, typename IdxT>
__global__ void batched_partition_kernel(PredictWorkspacePointers<DataT, LabelT, IdxT> pointers,
                                         PredictWorkspaceMeta<DataT, LabelT, IdxT> meta)
{
  auto block_id = static_cast<IdxT>(blockIdx.x);
  if (block_id >= meta.n_block_tasks) { return; }

  auto tid = static_cast<IdxT>(threadIdx.x);
  auto task = pointers.d_block_tasks[block_id];
  bool active = tid < task.count;

  __shared__ int s_is_left[BLOCK_TASK_SIZE];
  __shared__ IdxT s_chunk_begin[BLOCK_TASK_SIZE];
  __shared__ IdxT s_local_left_rank[BLOCK_TASK_SIZE];

  NodeWorkItemChunk<IdxT> chunk{};
  IdxT slot_local = 0;
  IdxT node_begin = 0;
  IdxT node_count = 0;
  IdxT split_nleft = 0;
  IdxT row_pos = 0;
  bool lane_valid = false;
  bool is_left = false;
  DataT split_quesval = DataT(0);

  if (active) {
    chunk = pointers.d_chunks[task.work_item_chunk_ids[tid]];
    slot_local = tid - chunk.thread_local_begin;
    s_chunk_begin[tid] = chunk.thread_local_begin;

    if (slot_local < chunk.instances_count && chunk.payload_idx < meta.n_payloads) {
      auto work_item = pointers.d_work_items[chunk.work_item_idx];
      node_begin = static_cast<IdxT>(work_item.instances.begin);
      node_count = static_cast<IdxT>(work_item.instances.count);
      row_pos = chunk.instances_begin + slot_local;

      if (row_pos < node_begin + node_count && row_pos < meta.n_rows) {
        auto split = pointers.d_splits[chunk.payload_idx];
        split_nleft = static_cast<IdxT>(split.nLeft);
        split_quesval = split.quesval;
        DataT projected = pointers.d_trans[row_pos];
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

  pointers.d_row_ids_scratch[dst_pos] = pointers.d_row_ids[row_pos];
}

template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_partition_kernel(const PredictWorkspacePointers<DataT, LabelT, IdxT>& pointers,
                                     const PredictWorkspaceMeta<DataT, LabelT, IdxT>& meta,
                                     cudaStream_t stream)
{
  dim3 block(BLOCK_TASK_SIZE);
  dim3 grid(meta.n_block_tasks);
  batched_partition_kernel<DataT, LabelT, IdxT><<<grid, block, 0, stream>>>(pointers, meta);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename DataT, typename LabelT, typename IdxT>
__global__ void batched_copyback_kernel(PredictWorkspacePointers<DataT, LabelT, IdxT> pointers,
                                        PredictWorkspaceMeta<DataT, LabelT, IdxT> meta)
{
  auto block_id = static_cast<IdxT>(blockIdx.x);
  if (block_id >= meta.n_block_tasks) { return; }

  auto tid = static_cast<IdxT>(threadIdx.x);
  auto task = pointers.d_block_tasks[block_id];
  if (tid >= task.count) { return; }

  auto chunk = pointers.d_chunks[task.work_item_chunk_ids[tid]];
  auto slot_local = tid - chunk.thread_local_begin;
  if (slot_local >= chunk.instances_count) { return; }

  auto row_pos = chunk.instances_begin + slot_local;
  if (row_pos >= meta.n_rows) { return; }

  pointers.d_row_ids[row_pos] = pointers.d_row_ids_scratch[row_pos];
}

template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_copyback_kernel(const PredictWorkspacePointers<DataT, LabelT, IdxT>& pointers,
                                    const PredictWorkspaceMeta<DataT, LabelT, IdxT>& meta,
                                    cudaStream_t stream)
{
  dim3 block(BLOCK_TASK_SIZE);
  dim3 grid(meta.n_block_tasks);
  batched_copyback_kernel<DataT, LabelT, IdxT><<<grid, block, 0, stream>>>(pointers, meta);
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

template <typename IdxT>
__global__ void compute_chunk_offsets_kernel(NodeWorkItemChunk<IdxT>* d_chunks, IdxT n_chunks)
{
  // Predict batches typically have modest chunk counts; one-thread linear pass avoids
  // multi-kernel + scan-by-key launch overhead and writes offsets directly in-place.
  if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
  if (n_chunks <= 0) { return; }

  IdxT curr_work_item = d_chunks[0].work_item_idx;
  IdxT loff = 0;
  IdxT roff = 0;

  for (IdxT i = 0; i < n_chunks; i++) {
    auto c = d_chunks[i];
    if (i == 0 || c.work_item_idx != curr_work_item) {
      curr_work_item = c.work_item_idx;
      loff = 0;
      roff = 0;
    }
    d_chunks[i].loff = loff;
    d_chunks[i].roff = roff;
    loff += c.nLeft;
    roff += c.nRight;
  }
}

// Template definition moved from the header
template <typename DataT, typename LabelT, typename IndexT>
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
  predict(handle,
          tree,
          max_batch_size,
          rows,
          n_rows,
          n_cols,
          scale,
          predictions,
          num_outputs,
          verbosity,
          handle.get_stream());
}

// Template definition moved from the header
template <typename DataT, typename LabelT, typename IndexT>
void SPORFDecisionTree::predict(const raft::handle_t& handle,
                                const TreeMetaDataNode<DataT, LabelT>& tree,
                                std::size_t max_batch_size,
                                const DataT* rows, // input data in column-major format, device pointer
                                std::size_t n_rows,
                                std::size_t n_cols,
                                double scale,
                                DataT* predictions,
                                int num_outputs,
                                rapids_logger::level_enum verbosity,
                                cudaStream_t stream)
{
  SPORFDecisionTreeWorkspace<DataT, LabelT, IndexT> ws(n_rows, max_batch_size, stream);
  predict(handle,
          tree,
          max_batch_size,
          rows,
          n_rows,
          n_cols,
          scale,
          predictions,
          num_outputs,
          verbosity,
          ws,
          stream);
}

// Template definition moved from the header
template <typename DataT, typename LabelT, typename IndexT>
void SPORFDecisionTree::predict(const raft::handle_t& handle,
                                const TreeMetaDataNode<DataT, LabelT>& tree,
                                std::size_t max_batch_size,
                                const DataT* rows, // input data in column-major format, device pointer
                                std::size_t n_rows,
                                std::size_t n_cols,
                                double scale,
                                DataT* predictions,
                                int num_outputs,
                                rapids_logger::level_enum verbosity,
                                SPORFDecisionTreeWorkspace<DataT, LabelT, IndexT>& ws,
                                cudaStream_t stream)
  {
    RAFT_CUDA_TRY(cudaSetDevice(handle.get_device()));
    using IdxT = IndexT;
    const bool do_timing = ML::default_logger().should_log(rapids_logger::level_enum::debug);
    using clock_t = std::chrono::steady_clock;
    auto to_ms = [](clock_t::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };

    cudaEvent_t ev_stage_start{}, ev_stage_stop{};
    if (do_timing) {
      RAFT_CUDA_TRY(cudaEventCreate(&ev_stage_start));
      RAFT_CUDA_TRY(cudaEventCreate(&ev_stage_stop));
    }

    double ms_setup = 0.0;
    double ms_pop = 0.0;
    double ms_h2d_meta = 0.0;
    double ms_projection = 0.0;
    double ms_scan = 0.0;
    double ms_partition = 0.0;
    double ms_splits_d2h = 0.0;
    double ms_push = 0.0;
    double ms_leaf_finalize = 0.0;
    size_t n_batches = 0;

    auto measure_gpu = [&](double& accum_ms, const auto& fn) {
      if (!do_timing) {
        fn();
        return;
      }
      RAFT_CUDA_TRY(cudaEventRecord(ev_stage_start, stream));
      fn();
      RAFT_CUDA_TRY(cudaEventRecord(ev_stage_stop, stream));
      RAFT_CUDA_TRY(cudaEventSynchronize(ev_stage_stop));
      float stage_ms = 0.0f;
      RAFT_CUDA_TRY(cudaEventElapsedTime(&stage_ms, ev_stage_start, ev_stage_stop));
      accum_ms += stage_ms;
    };

    auto t_setup_start = clock_t::now();

    IndexT n_classes = 0; // Dummy variable, not used in prediction
    ASSERT(static_cast<IndexT>(n_rows) <= ws.meta.n_rows, "Predict workspace insufficient n_rows");
    ASSERT(static_cast<IndexT>(max_batch_size) <= ws.meta.max_batch_size,
           "Predict workspace insufficient max_batch_size");
    ws.reset(tree, stream);
    ws.meta.n_cols = static_cast<IdxT>(n_cols);
    ws.pointers.d_input_col_major = rows;
    thrust::sequence(
      thrust::cuda::par.on(stream), ws.pointers.d_row_ids, ws.pointers.d_row_ids + n_rows, 0);
    Dataset<DataT, LabelT, IdxT> dataset = {
      ws.pointers.d_trans,             // projected data (single column)
      nullptr,                // labels (unused in predict)
      static_cast<IndexT>(n_rows),
      static_cast<IndexT>(1),   // N = number of projected columns
      static_cast<IndexT>(n_rows),
      static_cast<IndexT>(1),
      ws.pointers.d_row_ids,
      n_classes
    };
    if (do_timing) { ms_setup += to_ms(clock_t::now() - t_setup_start); }

    SPORFPredictNodeQueue<DataT, int, LabelT> queue(tree, n_rows, max_batch_size);
    while (queue.HasWork()) {
      n_batches++;
      auto t_pop_start = clock_t::now();
      auto popped_batch = queue.Pop();
      auto& work_items = std::get<0>(popped_batch);
      auto& projection_matrices = std::get<1>(popped_batch);
      auto& splits = std::get<2>(popped_batch);
      auto& chunks = std::get<3>(popped_batch);
      auto& block_tasks = std::get<4>(popped_batch);
      ws.meta.n_work_items = static_cast<IdxT>(work_items.size());
      ws.meta.n_payloads = static_cast<IdxT>(projection_matrices.size());
      ws.meta.n_chunks = static_cast<IdxT>(chunks.size());
      ws.meta.n_block_tasks = static_cast<IdxT>(block_tasks.size());
      if (do_timing) { ms_pop += to_ms(clock_t::now() - t_pop_start); }

      measure_gpu(ms_h2d_meta, [&]() {
        ASSERT(ws.meta.n_work_items <= ws.meta.cap_work_items, "Predict workspace overflow: work_items");
        ASSERT(ws.meta.n_payloads <= ws.meta.cap_payloads, "Predict workspace overflow: payloads");
        ASSERT(ws.meta.n_chunks <= ws.meta.cap_chunks, "Predict workspace overflow: chunks");
        ASSERT(ws.meta.n_block_tasks <= ws.meta.cap_block_tasks, "Predict workspace overflow: block_tasks");
        raft::update_device(ws.pointers.d_work_items, work_items.data(), ws.meta.n_work_items, stream);
        raft::update_device(
          ws.pointers.d_projection_matrices, projection_matrices.data(), ws.meta.n_payloads, stream);
        raft::update_device(ws.pointers.d_splits, splits.data(), ws.meta.n_payloads, stream);
        raft::update_device(ws.pointers.d_chunks, chunks.data(), ws.meta.n_chunks, stream);
        raft::update_device(ws.pointers.d_block_tasks, block_tasks.data(), ws.meta.n_block_tasks, stream);
      });

      if (!block_tasks.empty()) {
        measure_gpu(ms_projection, [&]() {
          launch_batched_projection_kernel<DataT, LabelT, IdxT>(ws.pointers, ws.meta, stream);
        });
      }

      if (!chunks.empty()) {
        measure_gpu(ms_scan, [&]() {
          auto n_chunks = static_cast<IdxT>(chunks.size());
          compute_chunk_offsets_kernel<IdxT><<<1, 1, 0, stream>>>(ws.pointers.d_chunks, n_chunks);
          RAFT_CUDA_TRY(cudaPeekAtLastError());
        });
      }

      if (!block_tasks.empty()) {
        measure_gpu(ms_partition, [&]() {
          launch_batched_partition_kernel<DataT, LabelT, IdxT>(ws.pointers, ws.meta, stream);
          launch_batched_copyback_kernel<DataT, LabelT, IdxT>(ws.pointers, ws.meta, stream);
        });
      }

      if (!splits.empty()) {
        auto t_d2h_start = clock_t::now();
        raft::update_host(splits.data(), ws.pointers.d_splits, splits.size(), stream);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
        if (do_timing) { ms_splits_d2h += to_ms(clock_t::now() - t_d2h_start); }
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

      auto t_push_start = clock_t::now();
      queue.Push(work_items);
      if (do_timing) { ms_push += to_ms(clock_t::now() - t_push_start); }
    }

    auto t_leaf_start = clock_t::now();
    const auto& leaves = queue.GetLeaves();
    auto leaves_size = static_cast<IdxT>(leaves.size());

    ASSERT(leaves_size <= static_cast<IdxT>(ws.d_leaves.size()), "Predict workspace overflow: leaves");
    ASSERT(static_cast<IdxT>(tree.vector_leaf.size()) <= static_cast<IdxT>(ws.d_vector_leaf.size()),
           "Predict workspace overflow: vector_leaf");
    raft::update_device(ws.d_leaves.data(), leaves.data(), leaves_size, stream);
    raft::update_device(
      ws.d_vector_leaf.data(), tree.vector_leaf.data(), tree.vector_leaf.size(), stream);

    // RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    auto d_leaves_ptr = ws.d_leaves.data();
    auto d_vector_leaf_ptr = ws.d_vector_leaf.data();
    auto d_prediction_leaves_ptr = ws.pointers.d_prediction_leaves;

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
    if (do_timing) { ms_leaf_finalize += to_ms(clock_t::now() - t_leaf_start); }

    if (do_timing) {
      CUML_LOG_DEBUG(
        "SPORFDecisionTree::predict timings (ms): setup=%f batches=%zu pop=%f h2d_meta=%f proj=%f scan=%f partition=%f splits_d2h=%f push=%f leaf_finalize=%f",
        ms_setup,
        n_batches,
        ms_pop,
        ms_h2d_meta,
        ms_projection,
        ms_scan,
        ms_partition,
        ms_splits_d2h,
        ms_push,
        ms_leaf_finalize);
      RAFT_CUDA_TRY(cudaEventDestroy(ev_stage_start));
      RAFT_CUDA_TRY(cudaEventDestroy(ev_stage_stop));
    }

    // RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

// Explicit instantiations (match the combos you need)
template void SPORFDecisionTree::predict<float, int, int>(const raft::handle_t&, const TreeMetaDataNode<float,int>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<double, int, int>(const raft::handle_t&, const TreeMetaDataNode<double,int>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<float, float, int>(const raft::handle_t&, const TreeMetaDataNode<float,float>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<double, double, int>(const raft::handle_t&, const TreeMetaDataNode<double,double>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<float, int, int>(const raft::handle_t&, const TreeMetaDataNode<float,int>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum, cudaStream_t);
template void SPORFDecisionTree::predict<double, int, int>(const raft::handle_t&, const TreeMetaDataNode<double,int>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum, cudaStream_t);
template void SPORFDecisionTree::predict<float, float, int>(const raft::handle_t&, const TreeMetaDataNode<float,float>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum, cudaStream_t);
template void SPORFDecisionTree::predict<double, double, int>(const raft::handle_t&, const TreeMetaDataNode<double,double>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum, cudaStream_t);
template void SPORFDecisionTree::predict<float, int, int>(const raft::handle_t&, const TreeMetaDataNode<float, int>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum, SPORFDecisionTreeWorkspace<float, int, int>&, cudaStream_t);
template void SPORFDecisionTree::predict<double, int, int>(const raft::handle_t&, const TreeMetaDataNode<double, int>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum, SPORFDecisionTreeWorkspace<double, int, int>&, cudaStream_t);
template void SPORFDecisionTree::predict<float, float, int>(const raft::handle_t&, const TreeMetaDataNode<float, float>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum, SPORFDecisionTreeWorkspace<float, float, int>&, cudaStream_t);
template void SPORFDecisionTree::predict<double, double, int>(const raft::handle_t&, const TreeMetaDataNode<double, double>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum, SPORFDecisionTreeWorkspace<double, double, int>&, cudaStream_t);

template <typename DataT, typename LabelT, typename IdxT>
__global__ void batched_projection_kernel(
  PredictWorkspacePointers<DataT, LabelT, IdxT> pointers,
  PredictWorkspaceMeta<DataT, LabelT, IdxT> meta
)
{
  auto block_id = static_cast<IdxT>(blockIdx.x);
  if (block_id >= meta.n_block_tasks) { return; }

  auto tid = static_cast<IdxT>(threadIdx.x);
  auto task = pointers.d_block_tasks[block_id];
  if (tid >= task.count) { return; }

  auto chunk = pointers.d_chunks[task.work_item_chunk_ids[tid]];
  auto row_pos = chunk.instances_begin + (tid - chunk.thread_local_begin);
  auto payload_id = chunk.payload_idx;
  if (row_pos >= meta.n_rows || payload_id >= meta.n_payloads) { return; }

  auto row_id = pointers.d_row_ids[row_pos];
  if (row_id >= meta.n_rows) { return; }

  auto pm = pointers.d_projection_matrices[payload_id];
  for (IdxT comp = 0; comp < pm.n_proj_components; comp++) {
    int start = pm.d_proj_indptr[comp];
    int end = pm.d_proj_indptr[comp + 1];
    DataT acc = DataT(0);

    for (int nz = start; nz < end; nz++) {
      auto feat = static_cast<IdxT>(pm.d_proj_indices[nz]);
      if (feat >= meta.n_cols) { continue; }
      auto coeff = pm.d_proj_coeffs[nz];
      acc += pointers.d_input_col_major[feat * meta.n_rows + row_id] * coeff;
    }
    pointers.d_trans[comp * meta.n_rows + row_pos] = acc;
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

    if(curr_payload < meta.n_payloads) {
      DataT curr_split = pointers.d_splits[curr_payload].quesval;

      for (IdxT j = 0; j < chunk.instances_count; j++) {
        IdxT row_pos_j = chunk.instances_begin + j;
        if (row_pos_j >= meta.n_rows) { continue; }

        // Predict currently projects one component per work item (column 0).
        DataT projected = pointers.d_trans[row_pos_j];
        if (projected <= curr_split) { curr_left_count++; }
      }

      if (curr_left_count > 0) {
        atomicAdd(&pointers.d_splits[curr_payload].nLeft, curr_left_count);
      }
    }

    auto* p_chunk = &pointers.d_chunks[task.work_item_chunk_ids[tid]];
    p_chunk->nLeft = curr_left_count; // also update chunk metadata for use in partitioning
    p_chunk->nRight = chunk.instances_count - curr_left_count;
  }
}

template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_projection_kernel(
  const PredictWorkspacePointers<DataT, LabelT, IdxT>& pointers,
  const PredictWorkspaceMeta<DataT, LabelT, IdxT>& meta,
  cudaStream_t stream
)
{
  constexpr int TPB = BLOCK_TASK_SIZE;  // or 128
  dim3 block(TPB);
  dim3 grid(meta.n_block_tasks);             // one block per BlockTask for first version
  batched_projection_kernel<DataT, LabelT, IdxT><<<grid, block, 0, stream>>>(pointers, meta);
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
