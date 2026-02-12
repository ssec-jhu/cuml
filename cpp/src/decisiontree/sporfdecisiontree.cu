// sporfdecisiontree.cu
#include "sporfdecisiontree.cuh"               // class/decl
#include <cuml/tree/sporfdecisiontree.hpp>     // public API
#include "batched-levelalgo/kernels/sporf_builder_kernels.cuh"
// pull in the device definitions but skip the explicit instantiations to avoid
// clashes with the fixed _DataT/_LabelT aliases in this TU.
#define ML_SPORF_BUILDER_SKIP_EXPLICIT_INSTANTIATIONS
#include "batched-levelalgo/kernels/sporf_builder_kernels_impl.cuh"
#undef ML_SPORF_BUILDER_SKIP_EXPLICIT_INSTANTIATIONS

namespace ML {
namespace DT {

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
                                const DataT* rows,
                                std::size_t n_rows,
                                std::size_t n_cols,
                                DataT* predictions,
                                int num_outputs,
                                rapids_logger::level_enum verbosity)
  {
    std::cout << "*** SPORFDecisionTree::predict() called with max_batch_size=" << max_batch_size << " ***" << std::endl;
    RAFT_CUDA_TRY(cudaSetDevice(handle.get_device()));
    auto stream = handle.get_stream();

    //RAFT_CUDA_TRY(cudaSetDevice(handle.get_device()));

    IdxT n_classes = 0; // Dummy variable, not used in prediction
    rmm::device_uvector<DataT> d_rows(0, stream);
    rmm::device_uvector<IdxT> row_ids(0, stream);
    rmm::device_uvector<DataT> d_contiguous(0, stream);
    rmm::device_uvector<DataT> d_trans(0, stream);
    // rmm::device_uvector<IdxT> smem(0, handle.get_stream());

    size_t req_bytes     = n_rows * n_cols * sizeof(DataT);
    size_t aligned_bytes = calculateAlignedBytes(req_bytes);
    size_t aligned_elems = aligned_bytes / sizeof(DataT);
    d_rows.resize(aligned_elems, stream);

    req_bytes     = n_rows * sizeof(IdxT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(IdxT);
    row_ids.resize(aligned_elems, stream);

    req_bytes     = n_rows * n_cols * sizeof(DataT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(DataT);
    d_contiguous.resize(aligned_elems, stream);

    req_bytes     = n_rows * sizeof(DataT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(DataT);
    d_trans.resize(aligned_elems, stream);

    // req_bytes     = max_batch_size * 2 * TPB_DEFAULT * sizeof(IdxT);
    // aligned_bytes = calculateAlignedBytes(req_bytes);
    // aligned_elems = aligned_bytes / sizeof(IdxT);
    // smem.resize(aligned_elems, handle.get_stream());

    std::cout << "Copying input data to device buffer..." << std::endl;
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_rows.data(),
                                  rows,
                                  n_rows * n_cols * sizeof(DataT),
                                  cudaMemcpyHostToDevice,
                                  stream));

    std::cout << "Preparing row IDs..." << std::endl;
    thrust::sequence(thrust::cuda::par.on(stream), row_ids.begin(), row_ids.begin() + n_rows, 0);

    Dataset<DataT, LabelT, IdxT> dataset = {
      d_trans.data(),         // projected data (single column)
      nullptr,                // labels (unused in predict)
      static_cast<IdxT>(n_rows),
      static_cast<IdxT>(1),   // N = number of projected columns
      static_cast<IdxT>(n_rows),
      static_cast<IdxT>(1),
      row_ids.data(),
      n_classes};

    raft::common::nvtx::range fun_scope("SPORFBuilder::train @sporfbuilder.cuh [batched-levelalgo]");
    MLCommon::TimerCPU timer;
    SPORFPredictNodeQueue<DataT, LabelT> queue(tree, n_rows, max_batch_size);
    while (queue.HasWork()) {
      auto work_items                      = queue.Pop();

      for (unsigned long i = 0; i < work_items.size(); i++) {
        auto& work_item = work_items[i];

        if (tree.sparsetree[work_item.idx].LeftChildId() == -1) continue;

        IdxT colid = 0;
        auto node = tree.sparsetree[work_item.idx];
        auto* random_matrix = tree.projection_vectors[work_item.idx].get();
        auto& begin = work_item.instances.begin;
        auto& count = work_item.instances.count;

        std::cout << "Processing node " << work_item.idx << ", begin=" << begin << ", count=" << count << ", n_cols=" << n_cols << std::endl;
        std::cout << "random_matrix:" << std::endl;
        print_rand_mat(*random_matrix, stream);

        raft::matrix::copyRows<DataT, IdxT, size_t>(
          d_rows.data(),                  // in (device)
          count,                          // number of rows of output matrix
          n_cols,                         // number of columns of output matrix
          d_contiguous.data() + begin,    // out
          dataset.row_ids + begin,        // row indices to copy
          count,
          stream,
          false                           // do-row-major
        );

        std::cout << "Data copied to contiguous buffer for node " << work_item.idx << std::endl;

        paramsRPROJ rproj_params{
          static_cast<int>(count), // number of samples
          static_cast<int>(n_cols),               // number of features
          1,                       // number of components
          -1.0f,                   // error tolerance (not used)
          false,                   // gaussian or sparse method
          1.0,                     // density (ignored when gaussian_method==true)
          false,                   // not used
          0                        // random seed
        };

        std::cout << "Fitting random projection for node " << work_item.idx << std::endl;
        RPROJtransform<DataT>(
          handle,
          d_contiguous.data() + begin,
          random_matrix,
          d_trans.data() + begin,
          &rproj_params
        );

        std::cout << "Data transformed for node " << work_item.idx << std::endl;

        auto first = thrust::make_counting_iterator<IdxT>(begin);
        auto last  = first + count;
        work_item.nLeft = thrust::count_if(
          thrust::cuda::par.on(stream),
          first,
          last,
          [=] __device__(IdxT row_id) {
            return dataset.data[row_id * dataset.N + colid] <= node.QueryValue();
          });
        // auto split = Split(node.QueryValue(), colid, node.BestMetric(), work_item.nLeft);

        // char* smem_base = reinterpret_cast<char*>(smem.data());
        // char* buf = smem_base + (i * 2 * TPB_DEFAULT * sizeof(IdxT));
        partition_kernel<DataT, LabelT, IdxT>
          <<<1, TPB_DEFAULT, 2 * TPB_DEFAULT * sizeof(IdxT), stream>>>(
            dataset, node.QueryValue(), colid, node.BestMetric(), work_item);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
        // partitionSamples<DataT, LabelT, IdxT, TPB_DEFAULT>(dataset, split, work_item, buf);
      }

      queue.Push(work_items);
        std::cout << std::endl;
    }

    const auto& leaves = queue.GetLeaves();
    rmm::device_uvector<NodeWorkItem> d_leaves(leaves.size(), stream);
    raft::update_device(d_leaves.data(), leaves.data(), leaves.size(), stream);

    rmm::device_uvector<DataT> d_vector_leaf(tree.vector_leaf.size(), stream);
    raft::update_device(d_vector_leaf.data(), tree.vector_leaf.data(), tree.vector_leaf.size(), stream);
    auto leaves_size = static_cast<IdxT>(leaves.size());

    auto d_leaves_ptr = d_leaves.data();
    auto d_vector_leaf_ptr = d_vector_leaf.data();
    thrust::for_each(
      thrust::cuda::par.on(stream),
      thrust::make_counting_iterator<IdxT>(0),
      thrust::make_counting_iterator<IdxT>(leaves_size),
      [=] __device__(IdxT l) {
        auto item = d_leaves_ptr[l];
        for (IdxT i = item.instances.begin; i < item.instances.begin + item.instances.count; i++) {
          IdxT row_id = dataset.row_ids[i];
          for (int j = 0; j < num_outputs; j++) {
            predictions[row_id * num_outputs + j] += d_vector_leaf_ptr[item.idx * num_outputs + j];
          }
        }
      });
  }

// Explicit instantiations (match the combos you need)
template void SPORFDecisionTree::predict<float, int>(const raft::handle_t&, const TreeMetaDataNode<float,int>&, std::size_t, const float*, std::size_t, std::size_t, float*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<double, int>(const raft::handle_t&, const TreeMetaDataNode<double,int>&, std::size_t, const double*, std::size_t, std::size_t, double*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<float, float>(const raft::handle_t&, const TreeMetaDataNode<float,float>&, std::size_t, const float*, std::size_t, std::size_t, float*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<double, double>(const raft::handle_t&, const TreeMetaDataNode<double,double>&, std::size_t, const double*, std::size_t, std::size_t, double*, int, rapids_logger::level_enum);

}  // namespace DT
}  // namespace ML
