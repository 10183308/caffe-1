#include <vector>
#include "cudnn.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"

#include "caffe/layers/DenseBlock_layer.hpp"

namespace caffe {

__global__ void DenseBlockForward(const int n, const float* in, float* out) {
  CUDA_KERNEL_LOOP(index, n){
    out[index] = sin(in[index]);
  } 
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
    
  const float* bottom_data_me;
  cudaMalloc(&bottom_data_me,count * sizeof(float)); 
  // NOLINT_NEXT_LINE(whitespace/operators)
  DenseBlockForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_FL_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void DenseBlockBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype sinx = out_data[index];
    out_diff[index] = in_diff[index] * cos(sinx);
  }
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    DenseBlockBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DenseBlockLayer);

}  // namespace caffe
