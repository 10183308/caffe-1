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

#include "caffe/util/gpu_util.cuh"
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
       
	//printf("inpointer %p\n",inPtr_gpu);
	//printf("outpointer %p\n",outPtr_gpu);
	CUDA_CHECK(cudaMemcpy(outPtr_local,inPtr_local,chunkSize_input * sizeof(Dtype),cudaMemcpyDeviceToDevice));
    }
}

template <typename Dtype>
void gpu_copy_many_to_one(Dtype* inPtr_gpu,Dtype* outPtr_gpu,int numChunks,int chunkSize_output,int chunkStride_input){
    for (int chunkIdx=0;chunkIdx<numChunks;++chunkIdx){
        Dtype* inPtr_local = inPtr_gpu + chunkIdx*chunkStride_input;
	Dtype* outPtr_local = outPtr_gpu + chunkIdx*chunkSize_output;
	CUDA_CHECK(cudaMemcpy(outPtr_local,inPtr_local,chunkSize_output * sizeof(Dtype),cudaMemcpyDeviceToDevice));
    }
}

template <typename Dtype>
void print_gpuPtr(Dtype* gpuPtr,int numValues){
    Dtype* cpuPtr = new Dtype[numValues];
    cudaMemcpy(cpuPtr,gpuPtr,numValues*sizeof(Dtype),cudaMemcpyDeviceToHost);
    for (int i=0;i<numValues;++i){
      std::cout<< cpuPtr[i] <<",";
    }
    std::cout<<std::endl;
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
    for (int transitionIdx=0;transitionIdx<this->numTransition;++transitionIdx){
      //Filter_grad_gpu
      int filterSize = (this->initChannel+this->growthRate*transitionIdx) * this->growthRate * this->filter_H * this->filter_W;
      log_gpuPtr(this->blobs_[transitionIdx]->mutable_gpu_diff(),filterSize,localDir+"Filter_grad_gpu_"+itos_cu(transitionIdx));
      //Scaler_grad_gpu
      int numChannelLocal = transitionIdx==0?this->initChannel:this->growthRate;
      log_gpuPtr(this->blobs_[transitionIdx+this->numTransition]->mutable_gpu_diff(),numChannelLocal,localDir+"Scaler_grad_gpu_"+itos_cu(transitionIdx));
      log_gpuPtr(this->blobs_[transitionIdx+this->numTransition]->mutable_gpu_data(),numChannelLocal,localDir+"Scaler_data_gpu_"+itos_cu(transitionIdx));
      //Bias_grad_gpu
      log_gpuPtr(this->blobs_[transitionIdx+2*this->numTransition]->mutable_gpu_diff(),numChannelLocal,localDir+"Bias_grad_gpu_"+itos_cu(transitionIdx));
      log_gpuPtr(this->blobs_[transitionIdx+2*this->numTransition]->mutable_gpu_data(),numChannelLocal,localDir+"Bias_data_gpu_"+itos_cu(transitionIdx));
    }
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
        
    //handles and descriptors
    //cudnn handle
    this->cudnnHandlePtr = new cudnnHandle_t;
    CUDNN_CHECK(cudnnCreate(this->cudnnHandlePtr));
    //conv_y global tensor descriptor
    this->tensorDescriptor_conv_y = new cudnnTensorDescriptor_t;
    cudnn::createTensor4dDesc<Dtype>(this->tensorDescriptor_conv_y);
    cudnn::setTensor4dDesc<Dtype>(this->tensorDescriptor_conv_y,this->N,this->growthRate,this->H,this->W,(this->numTransition*this->growthRate+this->initChannel)*this->H*this->W,this->H*this->W,this->W,1);	
    //per transition variables
    for (int i=0;i<this->numTransition;++i){
	//Result Running/Saving Mean/Variance/InvVariance
    	int localChannel = this->initChannel + i * this->growthRate;
    	Dtype* local_ResultMean;
	Dtype* local_ResultVar;
	Dtype* local_SaveMean;
	Dtype* local_SaveInvVar;
	
	CUDA_CHECK(cudaMalloc(&local_ResultMean,localChannel*sizeof(Dtype)));
    	CUDA_CHECK(cudaMalloc(&local_ResultVar,localChannel*sizeof(Dtype)));
    	CUDA_CHECK(cudaMalloc(&local_SaveMean,localChannel*sizeof(Dtype)));
    	CUDA_CHECK(cudaMalloc(&local_SaveInvVar,localChannel*sizeof(Dtype)));
		
    	cudaMemset(local_ResultMean,0,localChannel*sizeof(Dtype));
    	cudaMemset(local_ResultVar,0,localChannel*sizeof(Dtype));
    	cudaMemset(local_SaveMean,0,localChannel*sizeof(Dtype));
    	cudaMemset(local_SaveInvVar,0,totalChannel*sizeof(Dtype));
   
	this->ResultRunningMean_gpu.push_back(local_ResultMean);
	this->ResultRunningVariance_gpu.push_back(local_ResultVar);
	this->ResultSaveMean_gpu.push_back(local_SaveMean);
	this->ResultSaveInvVariance_gpu.push_back(local_SaveInvVar);
	//Result Save Variance[i] for i-th transition
	int reversibleChannels=(i==0)?0:initChannel+(i-1)*growthRate;
	Dtype* local_SaveVar;
	CUDA_CHECK(cudaMalloc(&local_SaveVar,reversibleChannels*sizeof(Dtype)));
	cudaMemset(local_SaveVar,0,reversibleChannels*sizeof(Dtype));
	this->ResultSaveVariance_gpu.push_back(local_SaveVar);
	
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
	//wide channelwise descriptor for BN type2
	int channelsBefore_noself = i==0?0:initChannel + (i-1) * growthRate;
	cudnnTensorDescriptor_t * wide_BNparam = new cudnnTensorDescriptor_t;
	cudnn::createTensor4dDesc<Dtype>(wide_BNparam);
	cudnn::setTensor4dDesc<Dtype>(1,channelsBefore_noself,1,1);
	this->tensorDescriptor_BN_wide.push_back(wide_BNparam);
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
__global__ void helper_computeBatchVariance(int n,Dtype* xPtr,Dtype* batchMeanPtr,Dtype* batchVarPtr,int transitionIdx,int numTransition,int N,int initChannel,int growthRate,int H,int W,int channelLimit){
  CUDA_KERNEL_LOOP(index, n){
    int localChannelIdx =  (index / (H * W)) % (initChannel + growthRate * numTransition);
    if (localChannelIdx < channelLimit){
      caffe_gpu_atomic_add((xPtr[index]-batchMeanPtr[localChannelIdx]) * (xPtr[index]-batchMeanPtr[localChannelIdx]),batchVarPtr + localChannelIdx);
    }
  }
}

//variance is only used in the reverse BN process
template <typename Dtype>
void computeBatchVariance(int n,Dtype* xPtr,Dtype* batchMeanPtr,Dtype* batchVarPtr,int transitionIdx,int numTransition,int N,int initChannel,int growthRate,int H,int W){ 
  int channelLimit = transitionIdx==0?0:initChannel+(transitionIdx-1)*growthRate; 
  helper_computeBatchVariance<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>((n,xPtr,batchMeanPtr,batchVarPtr,transitionIdx,numTransition,N,initChannel,growthRate,H,W,channelLimit);
  int M = N * H * W;
  caffe_gpu_scal<Dtype>(channelLimit,1.0/(M-1),batchVarPtr);
}

//ReLU: Negative_slope = 0.5
template <typename Dtype>
__global__ void ReLUForward(int n,Dtype* xPtr,Dtype* yPtr,int transitionIdx,int numTransition,int N,int initChannel,int growthRate,int H,int W){
  int channelLimit = initChannel + transitionIdx * growthRate;
  CUDA_KERNEL_LOOP(index, n){
    int localChannelIdx = (index / (H * W)) % (initChannel + growthRate * numTransition);
    //i.e. for transitionIdx==1, fwd both region 0 and 1
    if (localChannelIdx < channelLimit){
      yPtr[index] = xPtr[index] > 0? xPtr[index]: 0.5 * xPtr[index]; 
    }
  }
}

template <typename Dtype>
__global__ void ReLUBackward(int n,Dtype* xPtr,Dtype* dxPtr,Dtype* dyPtr,int transitionIdx,int numTransition,int N,int initChannel,int growthRate,int H,int W){
  int channelLimit = initChannel + transitionIdx * growthRate;
  CUDA_KERNEL_LOOP(index, n){
    int localChannelIdx = (index/(H*W)) % (initChannel + growthRate * numTransition);
    //i.e. for transitionIdx==1, bwd both region 0 and 1
    if (localChannelIdx < channelLimit){
      dxPtr[index] = xPtr[index]>0?dyPtr[index]:0.5*dyPtr[index];
    }
  }
}

template <typename Dtype>
__global__ void ReLUReverse(int n,Dtype* yPtr,Dtype* xPtr,int transitionIdx,int numTransition,int N,int initChannel,int growthRate,int H,int W){
  int channelLimit = transitionIdx==0?0:initChannel+(transitionIdx-1)*growthRate;
  CUDA_KERNEL_LOOP(index, n){
    int localChannelIdx = (index/(H*W)) % (initChannel + growthRate * numTransition); 
    //i.e. for transitionIdx==1, only reverse transform region 0
    if (localChannelIdx < channelLimit){
      xPtr[index] = yPtr[index]>=0?yPtr[index]:2*yPtr[index];
    }
  }
}

template <typename Dtype>
__global__ void BNReverse(int n,Dtype* yPtr,Dtype* xPtr,Dtype* scalerPtr,Dtype* biasPtr,Dtype* batchMeanPtr,Dtype* batchVarPtr,double epsilon,int N,int initChannel,int growthRate,int H,int W){
  int channelLimit = transitionIdx==0?0:initChannel+(transitionIdx-1)*growthRate;
  CUDA_KERNEL_LOOP(index, n){
    int localChannelIdx = (index/(H*W)) % (initChannel + growthRate * numTransition); 
    //i.e. for transitionIdx==1, only reverse transform region 0
    if (localChannelIdx < channelLimit){
      //x = a * y + b :: affine transform, find out a and b
      double a = sqrt(batchVarPtr[localChannelIdx] + epsilon) / (scalerPtr[localChannelIdx]);
      double b = batchMeanPtr[localChannelIdx] - (biasPtr[localChannelIdx]*sqrt(batchVarPtr[localChannelIdx]+epsilon))/(scalerPtr[localChannelIdx]);
      xPtr[index] = a * yPtr[index] + b;
    }
  }
}

//acts on cpu pointer
template <typename Dtype>
void ScalerProtector(Dtype* scaler_mutable_data,int numValues){
  for (int i=0;i<numValues;++i){
    Dtype localVal = scaler_mutable_data[i];
    if ((localVal < 1e-3) && (localVal > -1e-3)){
        if (localVal>=0){scaler_mutable_data[i] = 1e-3;}
	else {scaler_mutable_data[i] = -1e-3;}
    }
  }
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (!this->gpuInited){
      //std::cout<< "start gpu init"<<std::endl;
      this->GPU_Initialization();
      this->gpuInited = true;
      //std::cout<< "end gpu init"<<std::endl;
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
      //use scaler protector before forward
      int totalLocalChannel = initChannel + growthRate * transitionIdx;
      ScalerProtector(this->blobs_[numTransition + transitionIdx]->mutable_cpu_data(),totalLocalChannel);
      //BN::type1 normal narrow channels::postConv -> postBN 
      int channelsBefore_noself = (transitionIdx==0?0:(this->initChannel + (transitionIdx - 1)*this->growthRate));
      Dtype* BN_narrow_x_ptr = this->postConv_data_gpu + channelsBefore_noself * this->H * this->W;  
      Dtype* BN_narrow_y_ptr = this->postBN_data_gpu + channelsBefore_noself * this->H * this->W;
      Dtype* BN_narrow_globalMean= this->ResultRunningMean_gpu[transitionIdx] + channelsBefore_noself;
      Dtype* BN_narrow_globalVar = this->ResultRunningVariance_gpu[transitionIdx] + channelsBefore_noself;
      cudnnTensorDescriptor_t * narrowBN_paramDesc = (transitionIdx==0?tensorDescriptor_BN_initChannel:tensorDescriptor_BN_growthRate);
      
      if (this->phase_ == TEST){
          CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
	    *(this->cudnnHandlePtr),CUDNN_BATCHNORM_SPATIAL,
	    cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	    *(this->tensorDescriptorVec_narrow[transitionIdx]),BN_narrow_x_ptr,
	    *(this->tensorDescriptorVec_narrow[transitionIdx]),BN_narrow_y_ptr,
	    *narrowBN_paramDesc,
	    this->blobs_[this->numTransition+transitionIdx]->gpu_data()+channelsBefore_noself,
            this->blobs_[2*this->numTransition+transitionIdx]->gpu_data()+channelsBefore_noself,
	    BN_narrow_globalMean,BN_narrow_globalVar,CUDNN_BN_MIN_EPSILON)
	  );
      }
      else{
          Dtype* batchMean = this->ResultSaveMean_gpu[transitionIdx] + channelsBefore_noself;
          Dtype* batchInvVar =  this->ResultSaveInvVariance_gpu[transitionIdx] + channelsBefore_noself;
	  double EMA_factor = 1.0/(1+this->trainCycleIdx);
	  CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
	    *(this->cudnnHandlePtr),CUDNN_BATCHNORM_SPATIAL,
	    cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	    *(this->tensorDescriptorVec_narrow[transitionIdx]),BN_narrow_x_ptr,
	    *(this->tensorDescriptorVec_narrow[transitionIdx]),BN_narrow_y_ptr,
	    *narrowBN_paramDesc,
	    this->blobs_[this->numTransition + transitionIdx]->gpu_data() + channelsBefore_noself,
	    this->blobs_[2 * this->numTransition + transitionIdx]->gpu_data() + channelsBefore_noself,
	    EMA_factor,BN_narrow_globalMean,BN_narrow_globalVar,CUDNN_BN_MIN_EPSILON,
	    batchMean,batchInvVar)
	  );
      }
      //BN :: type2: wide channels, for anything prior to channels for
      //type1 BN
      if (transitionIdx > 0){
        cudnnTensorDescriptor_t wideBN_paramDesc = this->tensorDescriptor_BN_wide[transitionIdx]; 
	Dtype* BN_wide_x_ptr = this->postReLU_data_gpu;
	Dtype* BN_wide_y_ptr = this->postBN_data_gpu;
	Dtype* BN_wide_globalMean = this->ResultRunningMean_gpu;
	Dtype* BN_wide_globalVar = this->ResultRunningVariance_gpu;
	
	if (this->phase_ == TEST){
          CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
	    *(this->cudnnHandlePtr),CUDNN_BATCHNORM_SPATIAL,
	    cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	    *(this->tensorDescriptorVec_conv_x[transitionIdx-1]),BN_wide_x_ptr,
	    *(this->tensorDescriptorVec_conv_x[transitionIdx-1]),BN_wide_y_ptr,
	    *wideBN_paramDesc,
	    this->blobs_[this->numTransition + transitionIdx]->gpu_data(),
            this->blobs_[2 * this->numTransition + transitionIdx]->gpu_data(),
	    BN_wide_globalMean,BN_wide_globalVar,CUDNN_BN_MIN_EPSILON)
	  );
	}
	else {
          Dtype* batchMean = this->ResultSaveMean_gpu[transitionIdx];
	  Dtype* batchInvVar = this->ResultSaveInvVariance_gpu[transitionIdx];
          double EMA_factor = 1.0/(1+this->trainCycleIdx);
          CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
	    *(this->cudnnHandlePtr),CUDNN_BATCHNORM_SPATIAL,
	    cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	    *(this->tensorDescriptorVec_conv_x[transitionIdx-1]),BN_wide_x_ptr,
	    *(this->tensorDescriptorVec_conv_x[transitionIdx-1]),BN_wide_y_ptr,
	    *wideBN_paramDesc,
	    this->blobs_[this->numTransition + transitionIdx]->gpu_data(),
	    this->blobs_[2 * this->numTransition + transitionIdx]->gpu_data(),
	    EMA_factor,BN_wide_globalMean,BN_wide_globalVar,CUDNN_BN_MIN_EPSILON,
	    batchMean,batchInvVar)
	  );
	}
        //compute per channel variance after BN
        int work_n = this->N * (this->initChannel + this->numTransition * this->growthRate) * this->H * this->W; 
        computeBatchVariance<Dtype><<<CAFFE_GET_BLOCKS(work_n), CAFFE_CUDA_NUM_THREADS>>>(work_n,this->postReLU_data_gpu,this->ResultSaveMean_gpu[transitionIdx],this->ResultSaveVariance_gpu[transitionIdx],transitionIdx,this->numTransition,this->N,this->initChannel,this->growthRate,this->H,this->W);
      }

      //ReLU
      Dtype* ReLU_x_ptr = this->postBN_data_gpu;
      Dtype* ReLU_y_ptr = this->postReLU_data_gpu;
      ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(work_n), CAFFE_CUDA_NUM_THREADS>>>(work_n,ReLU_x_ptr,ReLU_y_ptr,transitionIdx,this->numTransition,this->N,this->initChannel,this->growthRate,this->H,this->W);
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
  this->trainCycleIdx += 1;
  //change top data
  int numValues = this->N * (this->initChannel+this->growthRate*this->numTransition) * this->H * this->W; 
  CUDA_CHECK(cudaMemcpy(top[0]->mutable_gpu_data(),this->postConv_data_gpu,numValues * sizeof(Dtype),cudaMemcpyDeviceToDevice));
  //this->logInternal_gpu("TClog");
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
    int numValues = this->N * (this->initChannel + this->growthRate * this->numTransition) * this->H * this->W;
    CUDA_CHECK(cudaMemcpy(this->postConv_grad_gpu,top_diff,numValues * sizeof(Dtype),cudaMemcpyDeviceToDevice));
    //Backward, transition by transition
    for (int transitionIdx=this->numTransition-1;transitionIdx>=0;--transitionIdx){
        int channelsBefore_self = this->initChannel + transitionIdx * this->growthRate;
        int channelsBefore_noself = transitionIdx>0?(this->initChannel + (transitionIdx - 1) * this->growthRate):0;
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
	//ReLU Bwd, for any j <= transitionIdx
	int work_n = this->N * (this->initChannel + this->numTransition * this->growthRate) * this->H * this->W;
	Dtype* ReLU_x_local = postBN_data_gpu;
	Dtype* ReLU_dy_local = postReLU_grad_gpu;
        Dtype* ReLU_dx_local = postBN_grad_gpu;	
	ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(work_n),CAFFE_CUDA_NUM_THREADS>>>(work_n,ReLU_x_local,ReLU_dx_local,ReLU_dy_local,transitionIdx,this->numTransition,this->N,this->initChannel,this->growthRate,this->H,this->W);
        //ReLU Reverse, for any j < transitionIdx
        Dtype* ReLU_reverse_y_local = postBN_data_gpu;
	Dtype* ReLU_reverse_x_local = postReLU_data_gpu;
	ReLUReverse<Dtype><<<CAFFE_GET_BLOCKS(work_n),CAFFE_CUDA_NUM_THREADS>>>(work_n,ReLU_reverse_y_local,ReLU_reverse_x_local,transitionIdx,this->numTransition,this->N,this->initChannel,this->growthRate,this->H,this->W);
	//BN Bwd, type2, wide
	if (transitionIdx > 0){
	  Dtype* BNwide_x_local = this->postReLU_data_gpu;
	  Dtype* BNwide_dx_local = this->postReLU_grad_gpu;
	  Dtype* BNwide_dy_local = this->postBN_grad_gpu;
	  Dtype* saveMeanwide_local = this->ResultSaveMean_gpu[transitionIdx]; 
	  Dtype* saveInvVarwide_local = this->ResultSaveInvVariance_gpu[transitionIdx];
	  cudnnTensorDescriptor_t * BNwideparam_desc = this->tensorDescriptor_BN_wide[transitionIdx];
	
	  CUDNN_CHECK(cudnnBatchNormalizationBackward(*(this->cudnnHandlePtr),
	    CUDNN_BATCHNORM_SPATIAL,
	    cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	    cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	    *(this->tensorDescriptorVec_conv_x[transitionIdx-1]),BNwide_x_local,
	    *(this->tensorDescriptorVec_conv_x[transitionIdx-1]),BNwide_dy_local,
	    *(this->tensorDescriptorVec_conv_x[transitionIdx-1]),BNwide_dx_local,
	    *BNwideparam_desc,
	    this->blobs_[this->numTransition + transitionIdx]->gpu_data(),
	    this->blobs_[this->numTransition + transitionIdx]->mutable_gpu_diff(),
	    this->blobs_[2*this->numTransition + transitionIdx]->mutable_gpu_diff(),
	    CUDNN_BN_MIN_EPSILON,saveMeanwide_local,saveInvVarwide_local
	    )		
	  );
	}
	//BN Bwd, type1, narrow
        Dtype* BNnarrow_x_local = this->postConv_data_gpu + channelsBefore_noself * this->H * this->W;
	Dtype* BNnarrow_dx_local = this->postConv_grad_gpu + channelsBefore_noself * this->H * this->W;
	Dtype* BNnarrow_dy_local = this->postBN_grad_gpu + channelsBefore_noself * this->H * this->W;
	Dtype* saveMeannarrow_local = this->ResultSaveMean_gpu + channelsBefore_noself;
	Dtype* saveInvVarnarrow_local = this->ResultSaveInvVariance_gpu + channelsBefore_noself;
        cudnnTensorDescriptor_t * BNnarrowparam_desc = (transitionIdx==0)?tensorDescriptor_BN_initChannel : tensorDescriptor_BN_growthRate;
        CUDNN_CHECK(cudnnBatchNormalizationBackward(*(this->cudnnHandlePtr),
	  CUDNN_BATCHNORM_SPATIAL,
	  cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	  cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
	  *(this->tensorDescriptorVec_narrow[transitionIdx]),BNnarrow_x_local,
	  *(this->tensorDescriptorVec_narrow[transitionIdx]),BNnarrow_dy_local,
	  *(this->tensorDescriptorVec_narrow[transitionIdx]),BNnarrow_dx_local,
	  *BNnarrowparam_desc,
	  this->blobs_[this->numTransition + transitionIdx]->gpu_data(),
	  this->blobs_[this->numTransition + transitionIdx]->mutable_gpu_diff(),
	  this->blobs_[2*this->numTransition + transitionIdx]->mutable_gpu_diff(),
	  CUDNN_BN_MIN_EPSILON,saveMeannarrow_local,saveInvVarnarrow_local
	  )		
	);	
	//BN reverse
        Dtype* BN_reverse_y_local = this->postReLU_data_gpu;
	Dtype* BN_reverse_x_local = this->postBN_data_gpu;
	Dtype* scalerPtr = this->blobs_[this->numTransition+transitionIdx]->gpu_data() + channelsBefore_noself;
	Dtype* biasPtr = this->blobs_[2*this->numTransition+transitionIdx]->gpu_data() + channelsBefore_noself;
	Dtype* batchMeanPtr = this->ResultSaveMean_gpu[transitionIdx] + channelsBefore_noself;
	Dtpye* batchVarPtr = this->ResultSaveVariance_gpu[transitionIdx] + channelsBefore_noself;
	BNReverse<Dtype><<<CAFFE_GET_BLOCKS(work_n),CAFFE_CUDA_NUM_THREADS>>>(work_n,BN_reverse_y_local,BN_reverse_x_local,scalerPtr,biasPtr,batchMeanPtr,batchVarPtr,CUDNN_BN_MIN_EPSILON,this->N,this->initChannel,this->growthRate,this->H,this->W);
    }
    //deploy buffer to bottom diff 
    int chunkSize_copy_init = this->initChannel * this->H * this->W;
    int chunkStride_copy = (this->initChannel + this->numTransition * this->growthRate) * this->H * this->W;
    gpu_copy_many_to_one(postConv_grad_gpu,bottom_diff,this->N,chunkSize_copy_init,chunkStride_copy);
    //this->logInternal_gpu("TClog");
    this->LoopEndCleanup_gpu();
}

INSTANTIATE_LAYER_GPU_FUNCS(DenseBlockLayer);

}  // namespace caffe
