#ifndef CAFFE_DENSEBLOCK_LAYER_HPP_
#define CAFFE_DENSEBLOCK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

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
    
  //start GPU specific data section
  //GPU ptr for efficient space usage only, these pointers not allocated when CPU_ONLY, this are not Blob because Descriptor is not traditional 
  float* postConv_data_gpu;
  float* postConv_grad_gpu;
  float* postBN_data_gpu;
  float* postBN_grad_gpu;
  float* postReLU_data_gpu;
  float* postReLU_grad_gpu;
  float* workspace;
  float* ResultRunningMean_gpu;
  float* ResultRunningVariance_gpu;
  float* ResultSaveMean_gpu;
  float* ResultSaveInvVariance_gpu;
  
  int initChannel, growthRate, numTransition; 
  int N,H,W; //N,H,W of the input tensor, inited in reshape phase
  int trainCycleIdx; //used in BN train phase for EMA Mean/Var estimation
  //convolution Related
  int pad_h, pad_w, conv_verticalStride, conv_horizentalStride; 
  int filter_H, filter_W;
  //gpu workspace size
  int workspace_size_bytes;
  //gpu handles and descriptors
  cudnnHandle_t* cudnnHandlePtr;
  vector<cudnnTensorDescriptor_t *> tensorDescriptorVec_narrow;//for BN & ReLU
  vector<cudnnTensorDescriptor_t *> tensorDescriptorVec_conv_x;//local Conv X
  cudnnTensorDescriptor_t * tensorDescriptor_conv_y;//local Conv Y
  cudnnTensorDescriptor_t * tensorDescriptor_BN_initChannel;//BN when transitionIdx = 0
  cudnnTensorDescriptor_t * tensorDescriptor_BN_growthRate;//BN when transitionIdx > 0
  //filter descriptor for conv
  vector<cudnnFilterDescriptor_t *> filterDescriptorVec;
  //conv descriptor for conv
  cudnnConvolutionDescriptor_t* conv_Descriptor;

  //end GPU specific data setion
};

}  // namespace caffe

#endif  // CAFFE_DENSEBLOCK_LAYER_HPP_

