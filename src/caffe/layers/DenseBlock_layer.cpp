#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/DenseBlock_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void DenseBlockLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	DenseBlockParameter dbParam = this->layer_param_.denseblock_param();
        this->numTransition = dbParam.numtransition();
        this->initChannel = dbParam.initchannel();
        this->growthRate = dbParam.growthrate();
        this->trainCycleIdx = 0; //initially, trainCycleIdx = 0
        this->pad_h = dbParam.pad_h();
        this->pad_w = dbParam.pad_w();
        this->conv_verticalStride = dbParam.conv_verticalstride();
        this->conv_horizentalStride = dbParam.conv_horizentalstride();
        this->filter_H = dbParam.filter_h();
        this->filter_W = dbParam.filter_w();
        this->workspace_size_bytes = 10000000;
        //Parameter Blobs
        //blobs_[0] is BN scaler weights, blobs_[1] is BN bias weights
        //blobs_[2:2+numTransition] is filter weights for convolution of each transition
        this->blobs_.resize(2+this->numTransition);
	int numChannelsTotal = this->initChannel + this->growthRate * this->numTransition;
	int channelShape_Arr[] = {1,numChannelsTotal,1,1};
        vector<int> channelShape(channelShape_Arr, channelShape_Arr+4);
	this->blobs_[0].reset(new Blob<Dtype>(channelShape));
	this->blobs_[1].reset(new Blob<Dtype>(channelShape));
	shared_ptr<Filler<Dtype> > weight_filler0(GetFiller<Dtype>(dbParam.bn_scaler_filler()));
        weight_filler0->Fill(this->blobs_[0].get());
        shared_ptr<Filler<Dtype> > weight_filler1(GetFiller<Dtype>(dbParam.bn_bias_filler()));
        weight_filler1->Fill(this->blobs_[1].get());
        for (int transitionIdx=0;transitionIdx < this->numTransition;++transitionIdx){
	    int inChannels = initChannel + transitionIdx * growthRate;
	    int filterShape_Arr[] = {growthRate,inChannels,filter_H,filter_W};
	    vector<int> filterShape (filterShape_Arr,filterShape_Arr+4);
	    this->blobs_[2+transitionIdx].reset(new Blob<Dtype>(filterShape));
	    shared_ptr<Filler<Dtype> > filter_Filler(GetFiller<Dtype>(dbParam.filter_filler()));
	    filter_Filler->Fill(this->blobs_[2+transitionIdx].get());
	}
	
	
}

  template <typename Dtype>
  void DenseBlockLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){ 
        this->N = bottom[0]->shape()[0]; 
        this->H = bottom[0]->shape()[2];
        this->W = bottom[0]->shape()[3];
//GPU intermediate ptrs
#ifndef CPU_ONLY
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
	for (int i=0;i<this->numTransition;++i){
	    int narrowChannelNum = (i==0?this->initChannel:this->growthRate);
	    cudnnTensorDescriptor_t * narrow_Desc_local = new cudnnTensorDescriptor_t;
	    cudnn::createTensor4dDesc<Dtype>(narrow_Desc_local);
	    cudnn::setTensor4dDesc<Dtype>(narrow_Desc_local,this->N,narrowChannelNum,this->H,this->W,(this->numTransition*this->growthRate+this->initChannel)*this->H*this->W,this->H*this->W,this->W,1);
	    this->tensorDescriptorVec_narrow.push_back(narrow_Desc_local);
	    int conv_x_channels = this->initChannel + this->growthRate * i;
	    cudnnTensorDescriptor_t * wide_Desc_local_x = new cudnnTensorDescriptor_t;
	    cudnn::createTensor4dDesc<Dtype>(wide_Desc_local_x);
	    cudnn::setTensor4dDesc<Dtype>(wide_Desc_local_x,this->N,conv_x_channels,this->H,this->W,(this->numTransition*this->growthRate+this->initChannel)*this->H*this->W,this->H*this->W,this->W,1);
	    this->tensorDescriptorVec_conv_x.push_back(wide_Desc_local_x); 
	}

#endif 
  }

  template <typename Dtype>
  void DenseBlockLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) 
  { 
    
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      top_data[i] = sin(bottom_data[i]);
    }
  }

  template <typename Dtype>
  void DenseBlockLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<Dtype>*>& bottom) 
  { 
    if (propagate_down[0]) {
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const int count = bottom[0]->count();
      Dtype bottom_datum;
      for (int i = 0; i < count; ++i) {
        bottom_datum = bottom_data[i];
        bottom_diff[i] = top_diff[i] * cos(bottom_datum);
      }
    }
  }

#ifdef CPU_ONLY
STUB_GPU(DenseBlockLayer);
#endif

INSTANTIATE_CLASS(DenseBlockLayer);
REGISTER_LAYER_CLASS(DenseBlock);

}  // namespace caffe  
