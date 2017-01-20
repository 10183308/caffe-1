#ifndef CAFFE_DENSEBLOCK_LAYER_HPP_
#define CAFFE_DENSEBLOCK_LAYER_HPP_

#include <vector>
#include <string>

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

  virtual void Forward_cpu_public(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu_public(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void Forward_gpu_public(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  void Backward_gpu_public(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  virtual void syncBlobs(DenseBlockLayer<Dtype>* originLayer);

  virtual void setLogId(int uid);

  virtual void logInternal_cpu(string dir);

  void logInternal_gpu(string dir,int transitionIdx,bool logDynamic,bool logDiff);

 protected:
  
  virtual void CPU_Initialization();

  void GPU_Initialization();

  virtual void LoopEndCleanup_cpu();

  void LoopEndCleanup_gpu();

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  //start logging specific data: for debugging
  int logId;
  //end logging specific data

  //common Blobs for both CPU & GPU mode
  //in this->blobs_, containing all filters for Convolution, scalers and bias for BN
  
  //start CPU specific data section
  bool cpuInited;
  //at T has shape (1,initC+T*growth,1,1)
  vector<Blob<Dtype>*> batch_Mean; 
  vector<Blob<Dtype>*> batch_Var;

  vector<Blob<Dtype>*> merged_conv;//at T has shape (N,initC+T*growth,H,W), but this vector has T+1 elements

  vector<Blob<Dtype>*> BN_XhatVec;//at T has shape (N,initC+T*growth,H,W)
  vector<Blob<Dtype>*> postBN_blobVec;
  vector<Blob<Dtype>*> postReLU_blobVec;
  vector<Blob<Dtype>*> postConv_blobVec;//at T has shape(N,growth,H,W)
  //end CPU specific data section

  //start GPU specific data section
  //GPU ptr for efficient space usage only, these pointers not allocated when CPU_ONLY, these are not Blobs because Descriptor is not traditional 
  bool gpuInited;
  Dtype* postConv_data_gpu;
  Dtype* postConv_grad_gpu;
  Dtype* postBN_data_gpu;
  Dtype* postBN_grad_gpu;
  Dtype* postReLU_data_gpu;
  Dtype* postReLU_grad_gpu;
  Dtype* workspace;
  vector<Dtype*> ResultSaveMean_gpu;
  vector<Dtype*> ResultSaveInvVariance_gpu;
    
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
  vector<cudnnTensorDescriptor_t *> tensorDescriptorVec_narrow;//for BN
  vector<cudnnTensorDescriptor_t *> tensorDescriptorVec_conv_x;//local Conv X
  cudnnTensorDescriptor_t * tensorDescriptor_conv_y;//local Conv Y
  cudnnTensorDescriptor_t * tensorDescriptor_BN_initChannel;//<channelwise>
  cudnnTensorDescriptor_t * tensorDescriptor_BN_growthRate;//<channelwise>
  vector<cudnnTensorDescriptor_t *> tensorDescriptor_BN_wide;//<channelwise>
  //filter descriptor for conv
  vector<cudnnFilterDescriptor_t *> filterDescriptorVec;
  //conv descriptor for conv
  cudnnConvolutionDescriptor_t* conv_Descriptor;

  //end GPU specific data setion
};

}  // namespace caffe

#endif  // CAFFE_DENSEBLOCK_LAYER_HPP_

