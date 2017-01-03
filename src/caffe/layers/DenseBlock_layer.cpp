// Sin neuron activation function layer.
// Adapted from TanH layer which was adapted from the ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/DenseBlock_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void DenseBlockLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	DenseBlockParameter dbParam = this->layer_param_.denseblock_param();
        this->numTransition = dbParam.numTransition();
        this->initChannel = dbParam.initChannel();
        this->growthRate = dbParam.growthRate();
        this->trainCycleIdx = 0; //initially, trainCycleIdx = 0
        this->pad_h = dbParam.pad_h();
        this->pad_w = dbParam.pad_w();
        this->conv_verticalStride = dbParam.conv_verticalStride();
        this->conv_horizentalStride = dbParam.conv_horizentalStride();
        this->filter_H = dbParam.filter_H();
        this->filter_W = dbParam.filter_W();
        this->workspace_size_bytes = 10000000;
        //Parameter Blobs
        //blobs_[0] is BN scaler weights, blobs_[1] is BN bias weights
        //blobs_[2:2+numTransition] is filter weights for convolution of each transition
        this->blobs_.resize(2+this->numTransition);
	int numChannelsTotal = this->initChannel + this->growthRate * this->numTransition;
	vector<int> channelShape = {1,numChannelsTotal,1,1};
	this->blobs_[0].reset(new Blob<Dtype>(channelShape));
	this->blobs_[1].reset(new Blob<Dtype>(channelShape));
	shared_ptr<Filler<Dtype>> weight_filler0(GetFiller<Dtype>(dbParam.BN_Scaler_Filler()));
        weight_filler0->Fill(this->blobs_[0].get());
        shared_ptr<Filler<Dtype>> weight_filler1(GetFiller<Dtype>(dbParam.BN_Bias_Filler()));
        weight_filler1->Fill(this->blobs_[1].get());
        for (int transitionIdx=0;transitionIdx < this->numTransition;++transitionIdx){
	    int inChannels = initChannel + transitionIdx * growthRate;
	    vector<int> filterShape = {growthRate,inChannels,filter_H,filter_W};
	    this->blobs_[2+transitionIdx].reset(new Blob<Dtype>(filterShape));
	    shared_ptr<Filler<Dtype>> filter_Filler(GetFiller<Dtype>(dbParam.Filter_Filler()));
	    filter_Filler->Fill(this->blobs_[2+transitionIdx].get());
	}
	
	
}

  template <typename Dtype>
  void DenseBlockLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){ 
        this->N = bottom[0]->shape[0]; 
        this->H = bottom[0]->shape[2];
        this->W = bottom[0]->shape[3];
//GPU intermediate ptrs
#ifndef CPU_ONLY
        int bufferSize_byte = this->N*(this->initChannel+this->growthRate*this->numTransition)*this->H*this->W*sizeof(Dtype);
        cudaMalloc(&this->postConv_data_gpu,bufferSize_byte);
        cudaMalloc(&this->postBN_data_gpu,bufferSize_byte);
        cudaMalloc(&this->postReLU_data_gpu,bufferSize_byte);
        cudaMalloc(&this->postConv_grad_gpu,bufferSize_byte);
        cudaMalloc(&this->postBN_grad_gpu,bufferSize_byte);
        cudaMalloc(&this->postReLU_grad_gpu,bufferSize_byte);

        cudaMemset(this->postConv_data_gpu,0,bufferSize_byte);
	cudaMemset(this->postBN_data_gpu,0,bufferSize_byte);
	cudaMemset(this->postReLU_data_gpu,0,bufferSize_byte);
	cudaMemset(this->postConv_grad_gpu,0,bufferSize_byte);
	cudaMemset(this->postBN_grad_gpu,0,bufferSize_byte);
	cudaMemset(this->postReLU_grad_gpu,0,bufferSize_byte);
	//workspace
	cudaMalloc(&this->workspace,this->workspace_size_bytes);
	cudaMemset(this->workspace,0,this->workspace_size_bytes);
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
