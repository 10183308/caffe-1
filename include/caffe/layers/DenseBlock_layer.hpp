#ifndef CAFFE_DENSEBLOCK_LAYER_HPP_
#define CAFFE_DENSEBLOCK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class DenseBlockLayer : public Layer<Dtype> {
 public:
  explicit DenseBlockLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top); 
  virtual inline const char* type() const { return "DenseBlock"; } 

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //common Blobs for both CPU & GPU mode
  Blob<Dtype> BN_Scaler, BN_Bias;
  vector<Blob<Dtype>*> filter_Vec;
  
  //GPU ptr for efficient space usage only, these pointers not allocated when CPU_ONLY, this are not Blob because Descriptor is not traditional 
  float* postConv_data_gpu;
  float* postConv_grad_gpu;
  float* postBN_data_gpu;
  float* postBN_grad_gpu;
  float* postReLU_data_gpu;
  float* postReLU_grad_gpu;
  

  
};

}  // namespace caffe

#endif  // CAFFE_DENSEBLOCK_LAYER_HPP_

