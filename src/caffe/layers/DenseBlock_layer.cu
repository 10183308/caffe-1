#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>

#include <sys/types.h>
#include <dirent.h>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "caffe/layers/DenseBlock_layer.hpp"

namespace caffe {

  bool dirExists_cu(string dirStr){
    const char* dirCStr = dirStr.c_str();
    DIR* dir = opendir(dirCStr);
    if (ENOENT == errno){
      return false;
    }
    closedir(dir);
    return true;
  }

  void tryCreateDirectory_cu(string fileName){
    vector<string> strVec;
    boost::split(strVec,fileName,boost::is_any_of("/"));
    string newStr="";
    for (int i=0;i<strVec.size()-1;++i){
      newStr += strVec[i] + (i==strVec.size()-2?"":"/");
    }
    boost::filesystem::path dirToCreate(newStr);
    if (!dirExists_cu(newStr)){
      boost::filesystem::create_directories(dirToCreate);
    }
  }


string itos_cu(int i){
  string output = boost::lexical_cast<string>(i);
  return output; 
}

template <typename Dtype>
void gpu_copy_one_to_many(const Dtype* inPtr_gpu,Dtype* outPtr_gpu,int numChunks,int chunkSize_input,int chunkStride_output){
    for (int chunkIdx=0;chunkIdx<numChunks;++chunkIdx){
	const Dtype* inPtr_local = inPtr_gpu + chunkIdx*chunkSize_input; 
	Dtype* outPtr_local = outPtr_gpu + chunkIdx*chunkStride_output;
        CUDA_CHECK(cudaMemcpy(outPtr_local,inPtr_local,chunkSize_input * sizeof(Dtype),cudaMemcpyDeviceToDevice));
    }
}

template <typename Dtype>
void gpu_copy_many_to_one(Dtype* inPtr_gpu,Dtype* outPtr_gpu,int numChunks,int chunkSize_output,int chunkStride_input){
    for (int chunkIdx=0;chunkIdx<numChunks;++chunkIdx){
        Dtype* inPtr_local = inPtr_gpu + chunkIdx*chunkStride_input;
	Dtype* outPtr_local = outPtr_gpu + chunkIdx*chunkSize_output;
	CUDA_CHECK(cudaMemcpy(inPtr_local,outPtr_local,chunkSize_output * sizeof(Dtype),cudaMemcpyDeviceToDevice));
    }
}

template <typename Dtype>
void log_gpuPtr(Dtype* gpuPtr,int numValues,string fileName){
    Dtype* cpuPtr = new Dtype[numValues];
    cudaMemcpy(cpuPtr,gpuPtr,numValues*sizeof(Dtype),cudaMemcpyDeviceToHost);
    const char* fileName_cstr = fileName.c_str();
    tryCreateDirectory_cu(fileName_cstr);
    std::ofstream outWriter(fileName_cstr,std::ofstream::out);
    for (int i=0;i<numValues;++i){
      outWriter<<cpuPtr[i]<<",";
    }
    outWriter<<std::endl;
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::logInternal_gpu(string dir){
    string localDir = dir+"/gpu_"+itos_cu(this->logId)+"/";
    int postBufferSize = this->N * (this->initChannel + this->growthRate * this->numTransition) * this->H * this->W;
    //postConv_data_gpu
    log_gpuPtr(this->postConv_data_gpu,postBufferSize,localDir+"postConv_data_gpu");
    //postConv_grad_gpu
    log_gpuPtr(this->postConv_grad_gpu,postBufferSize,localDir+"postConv_grad_gpu");
    //postBN_data_gpu
    log_gpuPtr(this->postBN_data_gpu,postBufferSize,localDir+"postBN_data_gpu");
    //postBN_grad_gpu
    log_gpuPtr(this->postBN_grad_gpu,postBufferSize,localDir+"postBN_grad_gpu");
    //postReLU_data_gpu
    log_gpuPtr(this->postReLU_data_gpu,postBufferSize,localDir+"postReLU_data_gpu");
    //postReLU_grad_gpu
    log_gpuPtr(this->postReLU_grad_gpu,postBufferSize,localDir+"postReLU_grad_gpu");
    //ResultRunningMean_gpu
    int numChannelsTotal = this->initChannel + this->growthRate * this->numTransition;
    log_gpuPtr(this->ResultRunningMean_gpu,numChannelsTotal,localDir+"ResultRunningMean_gpu");
    //ResultRunningVariance_gpu
    log_gpuPtr(this->ResultRunningVariance_gpu,numChannelsTotal,localDir+"ResultRunningVariance_gpu");
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::GPU_Initialization(){
    //GPU intermediate ptrs
    int bufferSize_byte = this->N*(this->initChannel+this->growthRate*this->numTransition)*this->H*this->W*sizeof(Dtype);
    CUDA_CHECK(cudaMalloc(&this->postConv_data_gpu,bufferSize_byte));
    CUDA_CHECK(cudaMalloc(&this->postBN_data_gpu,bufferSize_byte));
    CUDA_CHECK(cudaMalloc(&this->postReLU_data_gpu,bufferSize_byte));
    CUDA_CHECK(cudaMalloc(&this->postConv_grad_gpu,bufferSize_byte));
    CUDA_CHECK(cudaMalloc(&this->postBN_grad_gpu,bufferSize_byte));
    CUDA_CHECK(cudaMalloc(&this->postReLU_grad_gpu,bufferSize_byte));

    cudaMemset(this->postConv_data_gpu,0,bufferSize_byte);
    cudaMemset(this->postBN_data_gpu,0,bufferSize_byte);
    cudaMemset(this->postReLU_data_gpu,0,bufferSize_byte);
    cudaMemset(this->postConv_grad_gpu,0,bufferSize_byte);
    cudaMemset(this->postBN_grad_gpu,0,bufferSize_byte);
    cudaMemset(this->postReLU_grad_gpu,0,bufferSize_byte);
    //workspace
    CUDA_CHECK(cudaMalloc(&this->workspace,this->workspace_size_bytes));
    cudaMemset(this->workspace,0,this->workspace_size_bytes);
    //Result Running/Saving Mean/Variance/InvVariance
    int totalChannel = this->initChannel + this->growthRate*this->numTransition;
    CUDA_CHECK(cudaMalloc(&this->ResultRunningMean_gpu,totalChannel*sizeof(Dtype)));
    CUDA_CHECK(cudaMalloc(&this->ResultRunningVariance_gpu,totalChannel*sizeof(Dtype)));
    CUDA_CHECK(cudaMalloc(&this->ResultSaveMean_gpu,totalChannel*sizeof(Dtype)));
    CUDA_CHECK(cudaMalloc(&this->ResultSaveInvVariance_gpu,totalChannel*sizeof(Dtype)));
		
    cudaMemset(this->ResultRunningMean_gpu,0,totalChannel*sizeof(Dtype));
    cudaMemset(this->ResultRunningVariance_gpu,0,totalChannel*sizeof(Dtype));
    cudaMemset(this->ResultSaveMean_gpu,0,totalChannel*sizeof(Dtype));
    cudaMemset(this->ResultSaveInvVariance_gpu,0,totalChannel*sizeof(Dtype));

    //handles and descriptors
    //cudnn handle
    this->cudnnHandlePtr = new cudnnHandle_t;
    CUDNN_CHECK(cudnnCreate(this->cudnnHandlePtr));
    //global Activation Descriptor:ReLU
    this->activationDesc = new cudnnActivationDescriptor_t;
    cudnn::createActivationDescriptor<Dtype>(this->activationDesc,CUDNN_ACTIVATION_RELU);
    //conv_y global tensor descriptor
    this->tensorDescriptor_conv_y = new cudnnTensorDescriptor_t;
    cudnn::createTensor4dDesc<Dtype>(this->tensorDescriptor_conv_y);
    cudnn::setTensor4dDesc<Dtype>(this->tensorDescriptor_conv_y,this->N,this->growthRate,this->H,this->W,(this->numTransition*this->growthRate+this->initChannel)*this->H*this->W,this->H*this->W,this->W,1);	
    //BN&ReLU narrow descriptor, conv_x local tensor descriptor
    for (int i=0;i<this->numTransition;++i){
	//narrow descriptor
	int narrowChannelNum = (i==0?this->initChannel:this->growthRate);
	cudnnTensorDescriptor_t * narrow_Desc_local = new cudnnTensorDescriptor_t;
	cudnn::createTensor4dDesc<Dtype>(narrow_Desc_local);
	cudnn::setTensor4dDesc<Dtype>(narrow_Desc_local,this->N,narrowChannelNum,this->H,this->W,(this->numTransition*this->growthRate+this->initChannel)*this->H*this->W,this->H*this->W,this->W,1);
	this->tensorDescriptorVec_narrow.push_back(narrow_Desc_local);
	//conv_x descriptor
	int conv_x_channels = this->initChannel + this->growthRate * i;
	cudnnTensorDescriptor_t * wide_Desc_local_x = new cudnnTensorDescriptor_t;
	cudnn::createTensor4dDesc<Dtype>(wide_Desc_local_x);
	cudnn::setTensor4dDesc<Dtype>(wide_Desc_local_x,this->N,conv_x_channels,this->H,this->W,(this->numTransition*this->growthRate+this->initChannel)*this->H*this->W,this->H*this->W,this->W,1);
	this->tensorDescriptorVec_conv_x.push_back(wide_Desc_local_x); 
	//filter Descriptor for Convolution
	cudnnFilterDescriptor_t * localFilterDesc = new cudnnFilterDescriptor_t;
	cudnn::createFilterDesc<Dtype>(localFilterDesc,growthRate,conv_x_channels,this->filter_H,this->filter_W);
	this->filterDescriptorVec.push_back(localFilterDesc);
    }
    //BN parameter (Scale,Bias) Descriptor
    this->tensorDescriptor_BN_initChannel = new cudnnTensorDescriptor_t;
    cudnn::createTensor4dDesc<Dtype>(this->tensorDescriptor_BN_initChannel);
    cudnn::setTensor4dDesc<Dtype>(this->tensorDescriptor_BN_initChannel,1,this->initChannel,1,1);
    this->tensorDescriptor_BN_growthRate = new cudnnTensorDescriptor_t;
    cudnn::createTensor4dDesc<Dtype>(this->tensorDescriptor_BN_growthRate);
    cudnn::setTensor4dDesc<Dtype>(this->tensorDescriptor_BN_growthRate,1,this->growthRate,1,1);
    //Conv Descriptor
    this->conv_Descriptor = new cudnnConvolutionDescriptor_t;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(this->conv_Descriptor));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*this->conv_Descriptor,this->pad_h,this->pad_w,this->conv_verticalStride,this->conv_horizentalStride,1,1,CUDNN_CONVOLUTION));

}

template <typename Dtype>
void cleanupBuffer(Dtype* ptr_gpu,int count){
    cudaMemset(ptr_gpu,0,count*sizeof(Dtype));
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::LoopEndCleanup_gpu(){
    int valsBuffer = this->N * (this->initChannel + this->growthRate * this->numTransition) * this->H * this->W;
    cleanupBuffer(this->postConv_data_gpu,valsBuffer);
    cleanupBuffer(this->postConv_grad_gpu,valsBuffer);
    cleanupBuffer(this->postBN_data_gpu,valsBuffer);
    cleanupBuffer(this->postBN_grad_gpu,valsBuffer);
    cleanupBuffer(this->postReLU_data_gpu,valsBuffer);
    cleanupBuffer(this->postReLU_grad_gpu,valsBuffer);
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (!this->gpuInited){
      this->GPU_Initialization();
      this->gpuInited = true;
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  //copy to bottom_data to buffer with stride
  int chunkSize_copy_init = this->initChannel * this->H * this->W;
  int chunkStride_copy = (this->initChannel + this->growthRate * this->numTransition) * this->H * this->W;
  gpu_copy_one_to_many<Dtype>(bottom_data,this->postConv_data_gpu,this->N,chunkSize_copy_init,chunkStride_copy);
  //work in the buffer, transition by transition
  for (int transitionIdx=0;transitionIdx < this->numTransition;++transitionIdx){
      //BN and ReLU
      int channelsBefore_noself = (transitionIdx==0?0:(this->initChannel + (transitionIdx - 1)*this->growthRate));
      Dtype* BN_x_ptr = this->postConv_data_gpu + channelsBefore_noself * this->H * this->W;  
      Dtype* BN_y_ptr = this->postBN_data_gpu + channelsBefore_noself * this->H * this->W;
      Dtype* ReLU_y_ptr = this->postReLU_data_gpu + channelsBefore_noself * this->H * this->W;
      //BN
      Dtype* BN_mean_local = this->ResultRunningMean_gpu + channelsBefore_noself;
      Dtype* BN_var_local = this->ResultRunningVariance_gpu + channelsBefore_noself;
      cudnnTensorDescriptor_t * localBN_paramDesc = (transitionIdx==0?tensorDescriptor_BN_initChannel:tensorDescriptor_BN_growthRate);
      if (this->phase_ == TEST){
          CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
	    *(this->cudnnHandlePtr),CUDNN_BATCHNORM_SPATIAL,
	    cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	    *(this->tensorDescriptorVec_narrow[transitionIdx]),BN_x_ptr,
	    *(this->tensorDescriptorVec_narrow[transitionIdx]),BN_y_ptr,
	    *localBN_paramDesc,
	    this->blobs_[this->numTransition + transitionIdx]->gpu_data(),
            this->blobs_[2 * this->numTransition + transitionIdx]->gpu_data(),
	    BN_mean_local,BN_var_local,CUDNN_BN_MIN_EPSILON)
	  );
      }
      else{
          Dtype* resultSaveMean_local = this->ResultSaveMean_gpu + channelsBefore_noself;
          Dtype* resultSaveInvVariance_local =  this->ResultSaveInvVariance_gpu + channelsBefore_noself;
	  double EMA_factor = 1.0/(1+this->trainCycleIdx);	  
	  CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
	    *(this->cudnnHandlePtr),CUDNN_BATCHNORM_SPATIAL,
	    cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	    *(this->tensorDescriptorVec_narrow[transitionIdx]),BN_x_ptr,
	    *(this->tensorDescriptorVec_narrow[transitionIdx]),BN_y_ptr,
	    *localBN_paramDesc,
	    this->blobs_[this->numTransition + transitionIdx]->gpu_data(),
	    this->blobs_[2 * this->numTransition + transitionIdx]->gpu_data(),
	    EMA_factor,BN_mean_local,BN_var_local,CUDNN_BN_MIN_EPSILON,
	    resultSaveMean_local,resultSaveInvVariance_local)
	  );
	  this->trainCycleIdx += 1;
      } 
      //ReLU
      CUDNN_CHECK(cudnnActivationForward(*(this->cudnnHandlePtr),
	*(this->activationDesc), cudnn::dataType<Dtype>::one, 
	*(this->tensorDescriptorVec_narrow[transitionIdx]),BN_y_ptr,
	cudnn::dataType<Dtype>::zero,
	*(this->tensorDescriptorVec_narrow[transitionIdx]),ReLU_y_ptr)
      );
      //Convolution
      int delayChannel = this->initChannel + this->growthRate * transitionIdx;
      Dtype* conv_x_local = this->postReLU_data_gpu;
      Dtype* conv_y_local = this->postConv_data_gpu + delayChannel * this->H * this->W;
      CUDNN_CHECK(cudnnConvolutionForward(*(this->cudnnHandlePtr),
	cudnn::dataType<Dtype>::one,
	*(this->tensorDescriptorVec_conv_x[transitionIdx]),conv_x_local,
	*(this->filterDescriptorVec[transitionIdx]),
	this->blobs_[transitionIdx]->gpu_data(),
	*(this->conv_Descriptor),CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
	this->workspace,this->workspace_size_bytes,cudnn::dataType<Dtype>::zero,
	*(this->tensorDescriptor_conv_y),conv_y_local	
	)		      
      ); 
  } 
  //change top data
  int chunkSize_copy_end = this->growthRate * this->H * this->W;
  int resultChannelGap = this->initChannel + this->growthRate * (this->numTransition - 1);
  Dtype* resultBuffer_ptr = postConv_data_gpu + resultChannelGap * this->H * this->W;
  gpu_copy_many_to_one<Dtype>(resultBuffer_ptr,top_data,this->N,chunkSize_copy_end,chunkStride_copy);
  this->logInternal_gpu("TClog");
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (!this->gpuInited){
	this->GPU_Initialization();
    	this->gpuInited = true;
    }

    //assuming buffers store already computed value, always propagate down
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    //deploy top diff to buffer
    int chunkSize_copy_init = this->initChannel * this->H * this->W;
    int chunkSize_copy_end = this->growthRate * this->H * this->W;
    int chunkStride_copy = (this->initChannel + this->growthRate * this->numTransition) * this->H * this->W;
    int resultChannelGap = this->initChannel + this->growthRate * (this->numTransition - 1);
    Dtype* targetDeploy_ptr = this->postConv_grad_gpu + resultChannelGap * this->H * this->W; 
    gpu_copy_one_to_many(top_diff,targetDeploy_ptr,this->N,chunkSize_copy_end,chunkStride_copy);    
    //Backward, transition by transition
    for (int transitionIdx=this->numTransition-1;transitionIdx>=0;--transitionIdx){
        int channelsBefore_noself = this->initChannel + transitionIdx * this->growthRate;
        int channelsBefore_self = transitionIdx>0?(this->initChannel + (transitionIdx - 1) * this->growthRate):0;
	//Conv
        Dtype* filterGrad_local = this->blobs_[transitionIdx]->mutable_gpu_diff();
	const Dtype* filterData_local =this->blobs_[transitionIdx]->gpu_data();
	Dtype* conv_x_local = postReLU_data_gpu;
	Dtype* conv_dy_local = postConv_grad_gpu + channelsBefore_self * this->H * this->W;
	//Conv w.r.t. filter
	CUDNN_CHECK(cudnnConvolutionBackwardFilter(*(this->cudnnHandlePtr),
	  cudnn::dataType<Dtype>::one, 
	  *(this->tensorDescriptorVec_conv_x[transitionIdx]),conv_x_local,
	  *(this->tensorDescriptor_conv_y),conv_dy_local,
	  *(this->conv_Descriptor),CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
	  this->workspace,this->workspace_size_bytes,
	  cudnn::dataType<Dtype>::zero,
	  *(this->filterDescriptorVec[transitionIdx]),filterGrad_local	  
	  )		
	);
	//Conv w.r.t. x
	CUDNN_CHECK(cudnnConvolutionBackwardData(*(this->cudnnHandlePtr),
	  cudnn::dataType<Dtype>::one,
	  *(this->filterDescriptorVec[transitionIdx]),filterData_local,
	  *(this->tensorDescriptor_conv_y),conv_dy_local,
	  *(this->conv_Descriptor),CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
	  this->workspace,this->workspace_size_bytes,
	  cudnn::dataType<Dtype>::one,
	  *(this->tensorDescriptorVec_conv_x[transitionIdx]),postReLU_grad_gpu
	  )		
	);	
	//ReLU
	Dtype* ReLU_y_local = postReLU_data_gpu + channelsBefore_noself*this->H*this->W;
	Dtype* ReLU_x_local = postBN_data_gpu + channelsBefore_noself*this->H*this->W;
	Dtype* ReLU_dy_local = postReLU_grad_gpu + channelsBefore_noself*this->H*this->W;
        Dtype* ReLU_dx_local = postBN_grad_gpu + channelsBefore_noself*this->H*this->W;	
	CUDNN_CHECK(cudnnActivationBackward(*(this->cudnnHandlePtr),
	  *(this->activationDesc),cudnn::dataType<Dtype>::one,
	  *(this->tensorDescriptorVec_narrow[transitionIdx]),ReLU_y_local,
	  *(this->tensorDescriptorVec_narrow[transitionIdx]),ReLU_dy_local,
	  *(this->tensorDescriptorVec_narrow[transitionIdx]),ReLU_x_local,
	  cudnn::dataType<Dtype>::zero,
	  *(this->tensorDescriptorVec_narrow[transitionIdx]),ReLU_dx_local  
	  )
	);
	//BN
	Dtype* BN_x_local = postConv_data_gpu + channelsBefore_noself*this->H*this->W;
	Dtype* BN_dx_local = postConv_grad_gpu + channelsBefore_noself*this->H*this->W;
	Dtype* saveMean_local = this->ResultSaveMean_gpu + channelsBefore_noself; 
	Dtype* saveInvVar_local = this->ResultSaveInvVariance_gpu + channelsBefore_noself;
	cudnnTensorDescriptor_t * BNparam_desc = (transitionIdx==0?this->tensorDescriptor_BN_initChannel:this->tensorDescriptor_BN_growthRate);
	CUDNN_CHECK(cudnnBatchNormalizationBackward(*(this->cudnnHandlePtr),
	  CUDNN_BATCHNORM_SPATIAL,
	  cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	  cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	  *(this->tensorDescriptorVec_narrow[transitionIdx]),BN_x_local,
	  *(this->tensorDescriptorVec_narrow[transitionIdx]),ReLU_dx_local,
	  *(this->tensorDescriptorVec_narrow[transitionIdx]),BN_dx_local,
	  *BNparam_desc,
	  this->blobs_[this->numTransition + transitionIdx]->gpu_data(),
	  this->blobs_[this->numTransition + transitionIdx]->mutable_gpu_diff(),
	  this->blobs_[2*this->numTransition + transitionIdx]->mutable_gpu_diff(),
	  CUDNN_BN_MIN_EPSILON,saveMean_local,saveInvVar_local
	  )		
	);
    }
    //deploy buffer to bottom diff 
    gpu_copy_many_to_one(postConv_grad_gpu,bottom_diff,this->N,chunkSize_copy_init,chunkStride_copy); 
    this->LoopEndCleanup_gpu();
}

INSTANTIATE_LAYER_GPU_FUNCS(DenseBlockLayer);

}  // namespace caffe
