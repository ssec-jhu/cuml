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
                                double scale,
                                DataT* predictions,
                                int num_outputs,
                                rapids_logger::level_enum verbosity)
  {
    std::cout << "*** SPORFDecisionTree::predict() called with max_batch_size=" << max_batch_size << ", n_rows=" << n_rows << ", n_cols=" << n_cols << ", num_outputs=" << num_outputs << " ***" << std::endl;
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
    d_contiguous.resize(aligned_elems, stream);

    req_bytes     = n_rows * sizeof(IdxT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(IdxT);
    row_ids.resize(aligned_elems, stream);

    // req_bytes     = n_rows * n_cols * sizeof(DataT);
    // aligned_bytes = calculateAlignedBytes(req_bytes);
    // aligned_elems = aligned_bytes / sizeof(DataT);

    req_bytes     = n_rows * sizeof(DataT);
    aligned_bytes = calculateAlignedBytes(req_bytes);
    aligned_elems = aligned_bytes / sizeof(DataT);
    d_trans.resize(aligned_elems, stream);

    // req_bytes     = max_batch_size * 2 * TPB_DEFAULT * sizeof(IdxT);
    // aligned_bytes = calculateAlignedBytes(req_bytes);
    // aligned_elems = aligned_bytes / sizeof(IdxT);
    // smem.resize(aligned_elems, handle.get_stream());

    // for some inexplicable reason, the input data is passed in row-major format,
    // but we copy it to the device in column-major format to work with absolutely everything else in this entire codebase.
    std::cout << "Copying input data to device buffer..." << std::endl;
    // Copy host row-major input to device staging buffer
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_contiguous.data(),
                                  rows,
                                  n_rows * n_cols * sizeof(DataT),
                                  cudaMemcpyHostToDevice,
                                  stream));

    // Materialize a column-major view (d_rows) from the row-major staging buffer
    auto total = n_rows * n_cols;
    thrust::for_each(thrust::cuda::par.on(stream),
                     thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(total),
                     [in = d_contiguous.data(), out = d_rows.data(), n_rows, n_cols] __device__(size_t idx) {
                       size_t r = idx / n_cols;
                       size_t c = idx % n_cols;
                       out[c * n_rows + r] = in[r * n_cols + c];
                     });

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

    raft::common::nvtx::range fun_scope("SPORFBuilder::predict @sporfdecisiontree.cu [batched-levelalgo]");

    std::cout << "HERE'S THE TREE PASSED TO PREDICT:" << std::endl;
    for(size_t i = 0; i < tree.sparsetree.size(); i++) {
      auto& node = tree.sparsetree[i];
      std::cout << "Node " << i << ": is_leaf=" << node.IsLeaf() << ", quesval=" << node.QueryValue() << ", best_metric_val=" << node.BestMetric() << std::endl;
      if(!node.IsLeaf()) {
        auto& random_matrix = tree.projection_vectors[i];
        std::cout << "  Projection vector for node " << i << ":" << std::endl;
        print_rand_mat(*random_matrix, stream);
      }
    }
    std::cout << std::endl;
    std::cout << std::endl;


    MLCommon::TimerCPU timer;
    SPORFPredictNodeQueue<DataT, LabelT> queue(tree, n_rows, max_batch_size);
    while (queue.HasWork()) {
      auto work_items                      = queue.Pop();

      for (unsigned long i = 0; i < work_items.size(); i++) {
        auto& work_item = work_items[i];

        // if (tree.sparsetree[work_item.idx].LeftChildId() == -1) continue;

        IdxT colid = 0;
        auto node = tree.sparsetree[work_item.idx];
        auto& begin = work_item.instances.begin;
        auto& count = work_item.instances.count;

        if (node.IsLeaf() || count == 0) {
          continue; // leaves and empty nodes don't need to be processed further
        }

        auto* random_matrix = tree.projection_vectors[work_item.idx].get();

        std::cout << "Processing node " << work_item.idx << ", begin=" << begin << ", count=" << count << ", n_cols=" << n_cols << std::endl;
        std::cout << "random_matrix:" << std::endl;
        print_rand_mat(*random_matrix, stream);

        raft::matrix::copyRows<DataT, IdxT, size_t>(
          d_rows.data(),               // in (device, column-major, lda = n_rows)
          n_rows,                      // input leading dimension / total rows
          n_cols,                      // number of columns of input/output
          d_contiguous.data() + begin, // out (contiguous block for this node)
          dataset.row_ids + begin,     // row indices to copy
          count,                       // number of rows to copy for this node
          stream,
          false);                      // input is column-major

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

        std::cout << "Transforming data for node " << work_item.idx << std::endl;
        RPROJtransform<DataT>(
          handle,
          d_contiguous.data() + begin,
          random_matrix,
          d_trans.data() + begin,
          &rproj_params
        );

        std::cout << "Transformed data for node " << work_item.idx << ":" << std::endl;
        raft::print_device_vector("d_trans", d_trans.data() + begin, count, std::cout);
        // std::vector<DataT> h_trans(count);
        // RAFT_CUDA_TRY(cudaMemcpyAsync(h_trans.data(), d_trans.data() + begin, count * sizeof(DataT), cudaMemcpyDeviceToHost, stream));
        // RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
        // for (size_t j = 0; j < std::min(count, static_cast<size_t>(10)); j++) {
        //   std::cout << h_trans[j] << " ";
        // }
        // std::cout << std::endl;

        auto first = thrust::make_counting_iterator<IdxT>(begin);
        auto last  = first + count;
        work_item.nLeft = thrust::count_if(
          thrust::cuda::par.on(stream),
          first,
          last,
          [=] __device__(IdxT pos) {
            // dataset.data is column-major and aligned with the current row ordering; index by position
            return dataset.data[(colid * dataset.M) + pos] <= node.QueryValue();
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

    // std::vector<IdxT> h_row_ids(n_rows);
    // RAFT_CUDA_TRY(cudaMemcpyAsync(h_row_ids.data(),
    //                               dataset.row_ids,
    //                               n_rows * sizeof(IdxT),
    //                               cudaMemcpyDeviceToHost,
    //                               stream));
    // RAFT_CUDA_TRY(cudaStreamSynchronize(stream));


    const auto& leaves = queue.GetLeaves();
    auto leaves_size = static_cast<IdxT>(leaves.size());

    std::cout << "Leaves collected: " << leaves_size << std::endl;
    for(IdxT i = 0; i < leaves_size; i++) {
      auto item = leaves[i];
      std::cout << "Leaf " << i << ": idx=" << item.idx << ", depth=" << item.depth
                << ", instances=[" << item.instances.begin << ", " << item.instances.count << "]" << std::endl;
    }

    rmm::device_uvector<NodeWorkItem> d_leaves(leaves_size, stream);
    raft::update_device(d_leaves.data(), leaves.data(), leaves_size, stream);

    rmm::device_uvector<DataT> d_vector_leaf(tree.vector_leaf.size(), stream);
    raft::update_device(d_vector_leaf.data(), tree.vector_leaf.data(), tree.vector_leaf.size(), stream);

    // rmm::device_uvector<DataT> d_predictions(n_rows * num_outputs, stream);
    // RAFT_CUDA_TRY(cudaMemsetAsync(d_predictions.data(), 0, n_rows * num_outputs * sizeof(DataT), stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    rmm::device_uvector<IdxT> d_prediction_leaves(n_rows, stream);

    auto d_leaves_ptr = d_leaves.data();
    auto d_vector_leaf_ptr = d_vector_leaf.data();
    // auto d_predictions_ptr = d_predictions.data();
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

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    raft::print_device_vector("prediction_leaves", d_prediction_leaves.data(), n_rows, std::cout);
    raft::print_device_vector("predictions", predictions, n_rows * num_outputs, std::cout);

    std::cout << "HERE'S THE RAW TEST DATA:" << std::endl;
    for(size_t i = 0; i < n_rows; i++) {
      std::cout << "Row " << i << ": ";
      for (size_t j = 0; j < n_cols; j++) {
        std::cout << rows[i * n_cols + j] << " "; // remember the host input data is in row-major format
      }
      std::cout << std::endl;
    }

    std::cout << "FROM THE DEVICE:" << std::endl;
    std::vector<DataT> h_buf(n_rows * n_cols);
    RAFT_CUDA_TRY(cudaMemcpyAsync(h_buf.data(), d_rows.data(), n_rows * n_cols * sizeof(DataT), cudaMemcpyDeviceToHost, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    for(size_t i = 0; i < n_rows; i++) {
      std::cout << "Row " << i << ": ";
      for (size_t j = 0; j < n_cols; j++) {
        std::cout << h_buf[i + j * n_rows] << " "; // device data is in column-major format
      }
      std::cout << std::endl;
    }
    // std::cout << "Copying prediction_leaves to host..." << std::endl;
    // std::vector<IdxT> h_prediction_leaves(n_rows);
    // RAFT_CUDA_TRY(cudaMemcpyAsync(h_prediction_leaves.data(),
    //                               d_prediction_leaves.data(),
    //                               n_rows * sizeof(IdxT),
    //                               cudaMemcpyDeviceToHost,
    //                               stream));

    // std::cout << "Copying predictions to host..." << std::endl;
    // RAFT_CUDA_TRY(cudaMemcpyAsync(predictions,
    //                               d_predictions.data(),
    //                               n_rows * num_outputs * sizeof(DataT),
    //                               cudaMemcpyDeviceToHost,
    //                               stream));
    // RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    std::cout << "*** SPORFDecisionTree::predict() completed ***" << std::endl;
    std::cout << std::endl;
  }

// Explicit instantiations (match the combos you need)
template void SPORFDecisionTree::predict<float, int>(const raft::handle_t&, const TreeMetaDataNode<float,int>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<double, int>(const raft::handle_t&, const TreeMetaDataNode<double,int>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<float, float>(const raft::handle_t&, const TreeMetaDataNode<float,float>&, std::size_t, const float*, std::size_t, std::size_t, double, float*, int, rapids_logger::level_enum);
template void SPORFDecisionTree::predict<double, double>(const raft::handle_t&, const TreeMetaDataNode<double,double>&, std::size_t, const double*, std::size_t, std::size_t, double, double*, int, rapids_logger::level_enum);

}  // namespace DT
}  // namespace ML
