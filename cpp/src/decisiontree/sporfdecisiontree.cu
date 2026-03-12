// sporfdecisiontree.cu
#include "sporfdecisiontree.cuh"               // class/decl
#include <cuml/tree/sporfdecisiontree.hpp>     // public API
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
                                          const SparseTreeNode<DataT, LabelT>* const d_nodes,
                                          const NodeWorkItem* const d_work_items,
                                          IdxT n_work_items,
                                          IdxT* d_out_nLeft,
                                          const ProjectionMatrix<DataT, int>* d_proj_mats,
                                          IdxT n_proj_mats,
                                          const BlockTask<IdxT>* d_block_tasks,
                                          IdxT n_block_tasks,
                                          DataT* d_out_col_major,
                                          IdxT out_ld);

template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_projection_kernel(const DataT* d_input_col_major,
                                      IdxT n_rows,
                                      IdxT n_cols,
                                      const IdxT* d_row_ids,
                                      const SparseTreeNode<DataT, LabelT>* const d_nodes,
                                      const NodeWorkItem* const d_work_items,
                                      IdxT n_work_items,
                                      IdxT* d_out_nLeft,
                                      const ProjectionMatrix<DataT, int>* d_proj_mats,
                                      IdxT n_proj_mats,
                                      const BlockTask<IdxT>* d_block_tasks,
                                      IdxT n_block_tasks,
                                      DataT* d_out_col_major,
                                      IdxT out_ld,
                                      cudaStream_t stream);

template <typename DataT, typename LabelT, typename IdxT>
__global__ void partition_kernel(const Dataset<DataT, LabelT, IdxT> dataset,
                                 DataT split_quesval, IdxT split_colid, DataT split_best_metric_val,
                                 const SPORFDT::NodeWorkItem work_item)
{
    Split<DataT, IdxT> split{split_quesval, split_colid, split_best_metric_val, work_item.nLeft};
    // DT::InstanceRange instances{work_item.instances.begin, work_item.instances.count};
    // DT::NodeWorkItem local_work_item{work_item.idx, work_item.depth, instances};
    extern __shared__ char smem[];
    SPORFDT::partitionSamples<DataT, LabelT, IdxT, TPB_DEFAULT>(dataset, split, work_item, smem);
}

// Template definition moved from the header
template <class DataT, class LabelT>
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
    rmm::device_uvector<IdxT> row_ids(0, stream);
    rmm::device_uvector<DataT> d_trans(0, stream);
    rmm::device_uvector<SparseTreeNode<DataT, LabelT>> d_sparsetree(0, stream);

    size_t req_bytes     = n_rows * sizeof(IdxT);
    size_t aligned_bytes = calculateAlignedBytes(req_bytes);
    size_t aligned_elems = aligned_bytes / sizeof(IdxT);
    row_ids.resize(aligned_elems, stream);

    req_bytes     = n_rows * sizeof(DataT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(DataT);
    d_trans.resize(aligned_elems, stream);

    req_bytes     = tree.sparsetree.size() * sizeof(SparseTreeNode<DataT, LabelT>);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(SparseTreeNode<DataT, LabelT>);
    d_sparsetree.resize(aligned_elems, stream);
    raft::update_device(d_sparsetree.data(), tree.sparsetree.data(), tree.sparsetree.size(), stream);

    rmm::device_uvector<ProjectionMatrix<DataT, int>> d_projection_matrices(0, stream);
    rmm::device_uvector<BlockTask<int>> d_block_tasks(0, stream);
    rmm::device_uvector<SparseTreeNode<DataT, LabelT>> d_nodes(0, stream);
    rmm::device_uvector<SPORFDT::NodeWorkItem> d_work_items(0, stream);
    rmm::device_uvector<IdxT> d_nLeft(0, stream);

    // TODO: revisit these heuristics
    req_bytes     = (n_rows / 2) * sizeof(ProjectionMatrix<DataT, int>); // heuristic for max projection matrices needed at once
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(ProjectionMatrix<DataT, int>);
    d_projection_matrices.resize(aligned_elems, stream);

    req_bytes     = (n_rows / 2) * sizeof(BlockTask<IdxT>); // heuristic for max block tasks needed at once
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(BlockTask<IdxT>);
    d_block_tasks.resize(aligned_elems, stream);

    req_bytes     = tree.sparsetree.size() * sizeof(SparseTreeNode<DataT, LabelT>);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(SparseTreeNode<DataT, LabelT>);
    d_nodes.resize(aligned_elems, stream);
    raft::update_device(d_nodes.data(), tree.sparsetree.data(), tree.sparsetree.size(), stream);

    req_bytes     = n_rows * sizeof(SPORFDT::NodeWorkItem); // heuristic for max work items needed at once
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(SPORFDT::NodeWorkItem);
    d_work_items.resize(aligned_elems, stream);

    req_bytes     = n_rows * sizeof(IdxT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(IdxT);
    d_nLeft.resize(aligned_elems, stream);

    thrust::sequence(thrust::cuda::par.on(stream), row_ids.begin(), row_ids.begin() + n_rows, 0);
    Dataset<DataT, LabelT, IdxT> dataset = {
      d_trans.data(),         // projected data (single column)
      nullptr,                // labels (unused in predict)
      static_cast<IdxT>(n_rows),
      static_cast<IdxT>(1),   // N = number of projected columns
      static_cast<IdxT>(n_rows),
      static_cast<IdxT>(1),
      row_ids.data(),
      n_classes
    };

    MLCommon::TimerCPU timer;
    SPORFPredictNodeQueue<DataT, int, LabelT> queue(tree, n_rows, max_batch_size);
    while (queue.HasWork()) {
      auto [work_items, projection_matrices, block_tasks] = queue.Pop();
      raft::update_device(d_projection_matrices.data(), projection_matrices.data(), projection_matrices.size(), stream);
      raft::update_device(d_block_tasks.data(), block_tasks.data(), block_tasks.size(), stream);
      raft::update_device(d_work_items.data(), work_items.data(), work_items.size(), stream);
      RAFT_CUDA_TRY(cudaMemsetAsync(d_nLeft.data(), 0, work_items.size() * sizeof(IdxT), stream));

      if (!block_tasks.empty()) {
        launch_batched_projection_kernel<DataT, LabelT, IdxT>(rows,
                                                              static_cast<IdxT>(n_rows),
                                                              static_cast<IdxT>(n_cols),
                                                              dataset.row_ids,
                                                              d_nodes.data(),
                                                              d_work_items.data(),
                                                              static_cast<IdxT>(work_items.size()),
                                                              d_nLeft.data(),
                                                              d_projection_matrices.data(),
                                                              static_cast<IdxT>(projection_matrices.size()),
                                                              d_block_tasks.data(),
                                                              static_cast<IdxT>(block_tasks.size()),
                                                              d_trans.data(),
                                                              static_cast<IdxT>(dataset.M),
                                                              stream);
      }

      std::vector<IdxT> h_nLeft(work_items.size(), 0);
      if (!work_items.empty()) {
        raft::update_host(h_nLeft.data(), d_nLeft.data(), h_nLeft.size(), stream);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      }

      for(unsigned long i = 0; i < work_items.size(); i++) {
        auto& work_item = work_items[i];
        auto& begin = work_item.instances.begin;
        auto& count = work_item.instances.count;
        auto node = tree.sparsetree[work_item.idx];
        if (node.IsLeaf() || count == 0) { continue; }

        IdxT colid = 0;
        work_item.nLeft = h_nLeft[i];

        partition_kernel<DataT, LabelT, IdxT> <<<1, TPB_DEFAULT, 2 * TPB_DEFAULT * sizeof(IdxT), stream>>>(
          dataset, node.QueryValue(), colid, node.BestMetric(), work_item
        );
        RAFT_CUDA_TRY(cudaPeekAtLastError());
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

    // TODO: change this to iterate over rows instead of leaves
    thrust::for_each(
      thrust::cuda::par.on(stream),
      thrust::make_counting_iterator<IdxT>(0),
      thrust::make_counting_iterator<IdxT>(leaves_size),
      [=] __device__(IdxT l) {
        auto item = d_leaves_ptr[l];
        for (IdxT i = item.instances.begin; i < item.instances.begin + item.instances.count; i++) {
          IdxT row_id = dataset.row_ids[i];
          d_prediction_leaves_ptr[row_id] = item.idx; // Store the leaf index for each row
          for (int j = 0; j < num_outputs; j++) {
            predictions[row_id * num_outputs + j] += d_vector_leaf_ptr[item.idx * num_outputs + j] * scale; // Aggregate predictions (e.g., sum for regression)
          }
        }
      });

    // RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

// Explicit instantiations (match the combos you need)
template void SPORFDecisionTree::predict<float, int>(const raft::handle_t&, const TreeMetaDataNode<float,int>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<double, int>(const raft::handle_t&, const TreeMetaDataNode<double,int>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<float, float>(const raft::handle_t&, const TreeMetaDataNode<float,float>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<double, double>(const raft::handle_t&, const TreeMetaDataNode<double,double>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum);

template <typename DataT, typename LabelT, typename IdxT>
__global__ void batched_projection_kernel(
  const DataT* d_input_col_major,   // global input X, col-major [n_rows x n_cols]
  IdxT n_rows,
  IdxT n_cols,
  const IdxT* d_row_ids,            // collated row-id map
  const SparseTreeNode<DataT, LabelT>* const d_nodes,       // global tree nodes
  const SPORFDT::NodeWorkItem* const d_work_items,       // work items for this batch
  IdxT n_work_items,
  IdxT* d_out_nLeft,
  const ProjectionMatrix<DataT, int>* d_proj_mats,
  IdxT n_proj_mats,
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

  auto row_pos = task.row_ids_ids[tid];
  auto proj_id = task.proj_ids[tid];
  if (row_pos >= out_ld || proj_id >= n_proj_mats) { return; }

  auto row_id = d_row_ids[row_pos];
  if (row_id >= n_rows) { return; }

  auto pm = d_proj_mats[proj_id];
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

  if (tid == 0) {
    // Assumes SPORFPredictNodeQueue::Pop() guarantees:
    // 1) work_item_ids are batch-local ids into `popped`,
    // 2) rows for each work_item_id are appended contiguously in block-task order,
    // 3) row_ids_ids are collated positions into row_ids/output buffers.
    // See the Pop() contract comment in sporfdecisiontree.cuh.
    // With those guarantees, run-length counting here is linear in task.count.
    IdxT curr_work_item = static_cast<IdxT>(-1);
    DataT curr_split = DataT(0);
    IdxT curr_left_count = 0;

    for (IdxT j = 0; j < task.count; j++) {
      IdxT work_item_id = static_cast<IdxT>(task.work_item_ids[j]);
      if (work_item_id >= n_work_items) { continue; }

      if (work_item_id != curr_work_item) {
        if (curr_work_item != static_cast<IdxT>(-1) && curr_left_count > 0) {
          atomicAdd(d_out_nLeft + curr_work_item, curr_left_count);
        }
        curr_work_item = work_item_id;
        curr_left_count = 0;

        auto node_idx = static_cast<IdxT>(d_work_items[curr_work_item].idx);
        curr_split = d_nodes[node_idx].QueryValue();
      }

      IdxT row_pos_j = static_cast<IdxT>(task.row_ids_ids[j]);
      if (row_pos_j >= out_ld) { continue; }

      // Predict currently projects one component per work item (column 0).
      DataT projected = d_out_col_major[row_pos_j];
      if (projected <= curr_split) { curr_left_count++; }
    }

    if (curr_work_item != static_cast<IdxT>(-1) && curr_left_count > 0) {
      atomicAdd(d_out_nLeft + curr_work_item, curr_left_count);
    }
  }
}

template <typename DataT, typename LabelT, typename IdxT>
void launch_batched_projection_kernel(
  const DataT* d_input_col_major,
  IdxT n_rows,
  IdxT n_cols,
  const IdxT* d_row_ids,
  const SparseTreeNode<DataT, LabelT>* const d_nodes,
  const SPORFDT::NodeWorkItem* const d_work_items,
  IdxT n_work_items,
  IdxT* d_out_nLeft,
  const ProjectionMatrix<DataT, int>* d_proj_mats,
  IdxT n_proj_mats,
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
    d_nodes,
    d_work_items,
    n_work_items,
    d_out_nLeft,
    d_proj_mats,
    n_proj_mats,
    d_block_tasks,
    n_block_tasks,
    d_out_col_major,
    out_ld
  );
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace DT
}  // namespace ML
