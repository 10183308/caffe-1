#include <time.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "gtest/gtest.h"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/DenseBlock_layer.hpp"

namespace caffe {

int global_id = 1;
int big_initC = 160;
int big_growthRate = 12;
int big_numTransition = 12;

template <typename Dtype>
void printBlobDiff(Blob<Dtype>* B){
  for (int n=0;n<B->shape(0);++n){
    for (int c=0;c<B->shape(1);++c){
      for (int h=0;h<B->shape(2);++h){
        for (int w=0;w<B->shape(3);++w){
	  std::cout<< B->diff_at(n,c,h,w)<<",";
	}
      }
    }
  }
  std::cout<<std::endl;
}

template <typename TypeParam>
class DenseBlockLayerTest : public GPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DenseBlockLayerTest()
      : blob_bottom_cpu(new Blob<Dtype>(2,3,5,5)),
        blob_top_cpu(new Blob<Dtype>(2,7,5,5)),
	blob_bottom_gpu(new Blob<Dtype>(2,3,5,5)),
	blob_top_gpu(new Blob<Dtype>(2,7,5,5)),
	bigBlob_bottom_cpu(new Blob<Dtype>(64,big_initC,32,32)),
	bigBlob_top_cpu(new Blob<Dtype>(64,big_initC+big_growthRate*big_numTransition,32,32)),
	bigBlob_bottom_gpu(new Blob<Dtype>(64,big_initC,32,32)),
	bigBlob_top_gpu(new Blob<Dtype>(64,big_initC+big_growthRate*big_numTransition,32,32))
  {
    Caffe::set_random_seed(1704);
    DenseBlockParameter* db_param = this->layer_param.mutable_denseblock_param();
    db_param->set_numtransition(2);
    db_param->set_initchannel(3);
    db_param->set_growthrate(2);
    db_param->set_pad_h(1);
    db_param->set_pad_w(1);
    db_param->set_conv_verticalstride(1);
    db_param->set_conv_horizentalstride(1);
    db_param->set_filter_h(3);
    db_param->set_filter_w(3);
    db_param->mutable_filter_filler()->set_type("gaussian");
    db_param->mutable_bn_scaler_filler()->set_type("constant");
    db_param->mutable_bn_scaler_filler()->set_value(1);
    db_param->mutable_bn_bias_filler()->set_type("constant");
    db_param->mutable_bn_bias_filler()->set_value(0);
    //For comparison with existing Caffe layer    
    BatchNormParameter* bn_param = this->layer_param.mutable_batch_norm_param();
    bn_param->set_moving_average_fraction(0.999);
    bn_param->set_eps(1e-5);

    ScaleParameter* scale_param = this->layer_param.mutable_scale_param();
    scale_param->mutable_filler()->set_type("constant");
    scale_param->mutable_filler()->set_value(1);
    scale_param->set_bias_term(true);
    scale_param->mutable_bias_filler()->set_type("constant");
    scale_param->mutable_bias_filler()->set_value(0);

    ReLUParameter* relu_param = this->layer_param.mutable_relu_param();
    relu_param->set_negative_slope(0.5);
    
    ConvolutionParameter* conv_param = this->layer_param.mutable_convolution_param();
    conv_param->set_num_output(2);
    conv_param->set_bias_term(false);
    if (conv_param->pad_size()<1){
      conv_param->add_pad(1);
    }
    else {
      conv_param->set_pad(0,1);
    }
    if (conv_param->kernel_size_size()<1){
      conv_param->add_kernel_size(3);
    }
    else {
      conv_param->set_kernel_size(0,3);
    }
    if (conv_param->stride_size()<1){
      conv_param->add_stride(1);
    }
    else {
      conv_param->set_stride(0,1);
    }
    if (conv_param->dilation_size()<1){
      conv_param->add_dilation(1);
    }
    else {
      conv_param->set_dilation(0,1);
    }
    conv_param->mutable_weight_filler()->set_type("gaussian");
    
    ConcatParameter* concat_param = this->layer_param.mutable_concat_param();
    concat_param->set_axis(1);
    
    //big for speed test
    DenseBlockParameter* bigDB_param = this->bigLayer_param.mutable_denseblock_param();
    bigDB_param->set_numtransition(big_numTransition);
    bigDB_param->set_initchannel(big_initC);
    bigDB_param->set_growthrate(big_growthRate);
    bigDB_param->set_pad_h(1);
    bigDB_param->set_pad_w(1);
    bigDB_param->set_conv_verticalstride(1);
    bigDB_param->set_conv_horizentalstride(1);
    bigDB_param->set_filter_h(3);
    bigDB_param->set_filter_w(3);
    bigDB_param->mutable_filter_filler()->set_type("gaussian");
    bigDB_param->mutable_bn_scaler_filler()->set_type("constant");
    bigDB_param->mutable_bn_scaler_filler()->set_value(1);
    bigDB_param->mutable_bn_bias_filler()->set_type("constant");
    bigDB_param->mutable_bn_bias_filler()->set_value(0);
   
    this->bottomVec_gpu.push_back(this->blob_bottom_gpu);
    this->bottomVec_cpu.push_back(this->blob_bottom_cpu);
    this->topVec_gpu.push_back(this->blob_top_gpu);
    this->topVec_cpu.push_back(this->blob_top_cpu);

    this->bigBottomVec_cpu.push_back(this->bigBlob_bottom_cpu);
    this->bigTopVec_cpu.push_back(this->bigBlob_top_cpu);
    this->bigBottomVec_gpu.push_back(this->bigBlob_bottom_gpu);
    this->bigTopVec_gpu.push_back(this->bigBlob_top_gpu);
  }
  virtual ~DenseBlockLayerTest() {}

  void FillDiff(Blob<Dtype>* B){
    caffe_rng_gaussian<Dtype>(B->count(),Dtype(0.0),Dtype(1.0),B->mutable_cpu_diff());
  }
 
  LayerParameter layer_param;
  LayerParameter bigLayer_param;

  Blob<Dtype>* blob_bottom_cpu;
  Blob<Dtype>* blob_top_cpu;
  Blob<Dtype>* blob_bottom_gpu;
  Blob<Dtype>* blob_top_gpu;

  Blob<Dtype>* bigBlob_bottom_cpu;
  Blob<Dtype>* bigBlob_top_cpu;
  Blob<Dtype>* bigBlob_bottom_gpu;
  Blob<Dtype>* bigBlob_top_gpu;

  vector<Blob<Dtype>*> bottomVec_cpu;
  vector<Blob<Dtype>*> topVec_cpu;
  vector<Blob<Dtype>*> bottomVec_gpu;
  vector<Blob<Dtype>*> topVec_gpu;
  
  vector<Blob<Dtype>*> bigBottomVec_cpu;
  vector<Blob<Dtype>*> bigTopVec_cpu;
  vector<Blob<Dtype>*> bigBottomVec_gpu;
  vector<Blob<Dtype>*> bigTopVec_gpu;
};

TYPED_TEST_CASE(DenseBlockLayerTest, TestDtypesAndDevices);

TYPED_TEST(DenseBlockLayerTest, TestDenseBlockFwd) {
  typedef typename TypeParam::Dtype Dtype;
  DenseBlockParameter* db_param = this->layer_param.mutable_denseblock_param();
  //this->layer_param.set_phase(TRAIN);//To be disabled
  DenseBlockLayer<Dtype>* layer=new DenseBlockLayer<Dtype>(this->layer_param);
  //this->layer_param.set_phase(TEST);
  DenseBlockLayer<Dtype>* layer2=new DenseBlockLayer<Dtype>(this->layer_param);
  
  shared_ptr<Filler<Dtype> > gaussianFiller(GetFiller<Dtype>(db_param->filter_filler()));
  gaussianFiller->Fill(this->blob_bottom_cpu);
  this->blob_bottom_gpu->CopyFrom(*this->blob_bottom_cpu);
 

  layer->SetUp(this->bottomVec_cpu,this->topVec_cpu);
  layer2->SetUp(this->bottomVec_gpu,this->topVec_gpu);

  layer->setLogId(global_id);
  global_id += 1;
  layer2->setLogId(global_id);
  global_id += 1;

  //synchronize the random filled parameters of layer and layers
  layer->Forward_cpu_public(this->bottomVec_cpu,this->topVec_cpu);
  
  layer2->syncBlobs(layer);
  layer2->Forward(this->bottomVec_gpu,this->topVec_gpu);

  for (int n=0;n<2;++n){
    for (int c=0;c<7;++c){
      for (int h=0;h<5;++h){
        for (int w=0;w<5;++w){
	  EXPECT_NEAR(this->blob_top_cpu->data_at(n,c,h,w),this->blob_top_gpu->data_at(n,c,h,w),0.4);
	}
      }
    }
  }
  delete layer;
  delete layer2;
}


TYPED_TEST(DenseBlockLayerTest, TestDenseBlockBwd) {
  typedef typename TypeParam::Dtype Dtype;
  DenseBlockParameter* db_param = this->layer_param.mutable_denseblock_param();
  DenseBlockLayer<Dtype>* layer3=new DenseBlockLayer<Dtype>(this->layer_param);
  DenseBlockLayer<Dtype>* layer4=new DenseBlockLayer<Dtype>(this->layer_param);
  shared_ptr<Filler<Dtype> > gaussianFiller(GetFiller<Dtype>(db_param->filter_filler()));
  //bottom fill
  gaussianFiller->Fill(this->blob_bottom_cpu);
  this->blob_bottom_gpu->CopyFrom(*this->blob_bottom_cpu);
  layer3->SetUp(this->bottomVec_cpu,this->topVec_cpu);
  layer4->SetUp(this->bottomVec_gpu,this->topVec_gpu);
  
  layer3->setLogId(global_id);
  global_id += 1;
  layer4->setLogId(global_id);
  global_id += 1;
  //synchronize the random filled parameters of layer and layers
  layer4->syncBlobs(layer3);
  //forward
  layer3->Forward_cpu_public(this->bottomVec_cpu,this->topVec_cpu);
  layer4->Forward(this->bottomVec_gpu,this->topVec_gpu);
  //top fill
  this->FillDiff(this->blob_top_cpu);
  this->blob_top_gpu->CopyFrom(*this->blob_top_cpu,true);
  //backward
  vector<bool> propagate_down(1,true);
  layer3->Backward_cpu_public(this->topVec_cpu,propagate_down,this->bottomVec_cpu);
  layer4->Backward(this->topVec_gpu,propagate_down,this->bottomVec_gpu);
  //Bottom Grad
  for (int n=0;n<2;++n){
    for (int c=0;c<3;++c){
      for (int h=0;h<5;++h){
        for (int w=0;w<5;++w){
	  EXPECT_NEAR(this->blob_bottom_cpu->diff_at(n,c,h,w),this->blob_bottom_gpu->diff_at(n,c,h,w),0.4);
	}
      }
    }
  }
  //Filter Grad
  //Filter_1
  Blob<Dtype>* filter1layer3 = layer3->blobs()[1].get();
  Blob<Dtype>* filter1layer4 = layer4->blobs()[1].get();
  for (int outCIdx=0;outCIdx<2;++outCIdx){
    for (int inCIdx=0;inCIdx<5;++inCIdx){
      for (int filterHIdx=0;filterHIdx<3;++filterHIdx){
        for (int filterWIdx=0;filterWIdx<3;++filterWIdx){
	  EXPECT_NEAR(filter1layer3->diff_at(outCIdx,inCIdx,filterHIdx,filterWIdx),filter1layer4->diff_at(outCIdx,inCIdx,filterHIdx,filterWIdx),0.4);
	  //std::cout<<(filter1layer3->diff_at(outCIdx,inCIdx,filterHIdx,filterWIdx) - filter1layer4->diff_at(outCIdx,inCIdx,filterHIdx,filterWIdx))<<",";
	}
      }
    }
  }
  //Filter_0
  Blob<Dtype>* filter0layer3 = layer3->blobs()[0].get();
  Blob<Dtype>* filter0layer4 = layer4->blobs()[0].get();
  for (int outCIdx=0;outCIdx<2;++outCIdx){
    for (int inCIdx=0;inCIdx<3;++inCIdx){
      for (int filterHIdx=0;filterHIdx<3;++filterHIdx){
        for (int filterWIdx=0;filterWIdx<3;++filterWIdx){
	  EXPECT_NEAR(filter0layer3->diff_at(outCIdx,inCIdx,filterHIdx,filterWIdx),filter0layer4->diff_at(outCIdx,inCIdx,filterHIdx,filterWIdx),0.6); //slightly relax
          std::cout<<(filter0layer3->diff_at(outCIdx,inCIdx,filterHIdx,filterWIdx) - filter0layer4->diff_at(outCIdx,inCIdx,filterHIdx,filterWIdx))<<",";
	}
      }
    }
  }

  //Scaler Grad
  for (int transitionIdx=0;transitionIdx<layer3->numTransition;++transitionIdx){
    Blob<Dtype>* layer3localScaler = layer3->blobs()[layer3->numTransition+transitionIdx].get();
    Blob<Dtype>* layer4localScaler = layer4->blobs()[layer4->numTransition+transitionIdx].get();
    int localNumChannel = transitionIdx==0?3:2;
    for (int channelIdx=0;channelIdx < localNumChannel;++channelIdx){
      EXPECT_NEAR(layer3localScaler->diff_at(0,channelIdx,0,0),layer4localScaler->diff_at(0,channelIdx,0,0),1.2); 
      /*std::cout<<"CPU scaler"<<std::endl;
      std::cout<<layer3localScaler->diff_at(0,channelIdx,0,0)<<std::endl;
      std::cout<<"GPU scaler"<<std::endl;
      std::cout<<layer4localScaler->diff_at(0,channelIdx,0,0)<<std::endl;*/
    }
  } 
  //Bias Grad
  for (int transitionIdx=0;transitionIdx<layer3->numTransition;++transitionIdx){
    Blob<Dtype>* layer3localBias = layer3->blobs()[2*layer3->numTransition+transitionIdx].get();
    Blob<Dtype>* layer4localBias = layer4->blobs()[2*layer4->numTransition+transitionIdx].get();
    int localNumChannel = transitionIdx==0?3:2;
    for (int channelIdx=0;channelIdx < localNumChannel;++channelIdx){
      EXPECT_NEAR(layer3localBias->diff_at(0,channelIdx,0,0),layer4localBias->diff_at(0,channelIdx,0,0),0.4);
      /*std::cout<<"CPU Bias"<<std::endl;
      std::cout<<layer3localBias->diff_at(0,channelIdx,0,0)<<std::endl;
      std::cout<<"GPU Bias"<<std::endl;
      std::cout<<layer4localBias->diff_at(0,channelIdx,0,0)<<std::endl;*/
    }
  } 
  //GlobalMean/Var should have no Grad
  for (int i=0;i<2*layer3->numTransition;++i){
    Blob<Dtype>* layer3B = layer3->blobs()[3*layer3->numTransition+i].get();
    Blob<Dtype>* layer4B = layer4->blobs()[3*layer4->numTransition+i].get();
    for (int c=0;c<layer3B->shape(1);++c){
      EXPECT_NEAR(layer3B->diff_at(0,c,0,0),0,1e-3);
      EXPECT_NEAR(layer4B->diff_at(0,c,0,0),0,1e-3);
    }
  }

}

template <typename Dtype>
void BlobDataMemcpy(Blob<Dtype>* dest,Blob<Dtype>* src,int numValues){
  memcpy(dest->mutable_cpu_data(),src->cpu_data(),numValues*sizeof(Dtype));
}

void tryCreateDirectory(string fileName);

template <typename Dtype>
void logBlob(Blob<Dtype>* B,string filename);

//Fwd propagate in the orthodox way, also synchronize parameters to DenseBlock layer
template <typename Dtype>
void Simulate_Fwd(vector<Blob<Dtype>*>& bottom,vector<Blob<Dtype>*>& top,DenseBlockLayer<Dtype>* DBLayerPtr,LayerParameter* layerParamPtr){
  Blob<Dtype>* postBN1 = new Blob<Dtype>(2,3,5,5);
  vector<Blob<Dtype>*> postBN1Vec;
  postBN1Vec.push_back(postBN1);

  Blob<Dtype>* postScale1 = new Blob<Dtype>(2,3,5,5);
  vector<Blob<Dtype>*> postScale1Vec;
  postScale1Vec.push_back(postScale1);
  
  Blob<Dtype>* postReLU1 = new Blob<Dtype>(2,3,5,5);
  vector<Blob<Dtype>*> postReLU1Vec;
  postReLU1Vec.push_back(postReLU1); 
  
  Blob<Dtype>* postConv1 = new Blob<Dtype>(2,2,5,5);
  vector<Blob<Dtype>*> postConv1Vec;
  postConv1Vec.push_back(postConv1); 
  
  Blob<Dtype>* postConcat1 = new Blob<Dtype>(2,5,5,5);
  vector<Blob<Dtype>*> preConcat1Vec;
  preConcat1Vec.push_back(postReLU1);
  preConcat1Vec.push_back(postConv1);
  vector<Blob<Dtype>*> postConcat1Vec;
  postConcat1Vec.push_back(postConcat1); 
  
  Blob<Dtype>* postBN2 = new Blob<Dtype>(2,5,5,5);
  vector<Blob<Dtype>*> postBN2Vec;
  postBN2Vec.push_back(postBN2);

  Blob<Dtype>* postScale2 = new Blob<Dtype>(2,3,5,5);
  vector<Blob<Dtype>*> postScale2Vec;
  postScale2Vec.push_back(postScale2);
  
  Blob<Dtype>* postReLU2 = new Blob<Dtype>(2,5,5,5);
  vector<Blob<Dtype>*> postReLU2Vec;
  postReLU2Vec.push_back(postReLU2);  
  
  Blob<Dtype>* postConv2 = new Blob<Dtype>(2,2,5,5);
  vector<Blob<Dtype>*> postConv2Vec;
  postConv2Vec.push_back(postConv2);  
  
  vector<Blob<Dtype>*> preConcat2Vec;
  preConcat2Vec.push_back(postReLU2);
  preConcat2Vec.push_back(postConv2);
  //BN1
  BatchNormLayer<Dtype>* BNlayer1 = new BatchNormLayer<Dtype>(*layerParamPtr);
  BNlayer1->SetUp(bottom,postBN1Vec);
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[3*2+0].get(),BNlayer1->blobs()[0].get(),3); 
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[4*2+0].get(),BNlayer1->blobs()[1].get(),3); 
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[5*2].get(),BNlayer1->blobs()[2].get(),1); 
  //Scale1
  ScaleLayer<Dtype>* Scalelayer1 = new ScaleLayer<Dtype>(*layerParamPtr);
  Scalelayer1->SetUp(postBN1Vec,postScale1Vec);
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[1*2+0].get(),Scalelayer1->blobs()[0].get(),3);
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[2*2+0].get(),Scalelayer1->blobs()[1].get(),3);
  //ReLU1
  ReLULayer<Dtype>* ReLUlayer1 = new ReLULayer<Dtype>(*layerParamPtr);
  ReLUlayer1->SetUp(postScale1Vec,postReLU1Vec);
  //Conv1
  ConvolutionLayer<Dtype>* Convlayer1 = new ConvolutionLayer<Dtype>(*layerParamPtr);  
  Convlayer1->SetUp(postReLU1Vec,postConv1Vec);
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[0*2+0].get(),Convlayer1->blobs()[0].get(),54); 
  //Concat1  
  ConcatLayer<Dtype>* Concatlayer1 = new ConcatLayer<Dtype>(*layerParamPtr);
  Concatlayer1->SetUp(preConcat1Vec,postConcat1Vec);
  //BN2
  BatchNormLayer<Dtype>* BNlayer2 = new BatchNormLayer<Dtype>(*layerParamPtr);
  BNlayer2->SetUp(postConcat1Vec,postBN2Vec);
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[3*2+1].get(),BNlayer2->blobs()[0].get(),5); 
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[4*2+1].get(),BNlayer2->blobs()[1].get(),5); 
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[5*2].get(),BNlayer2->blobs()[2].get(),1); 
  //Scale2
  ScaleLayer<Dtype>* Scalelayer2 = new ScaleLayer<Dtype>(*layerParamPtr);
  Scalelayer2->SetUp(postBN2Vec,postScale2Vec);
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[1*2+1].get(),Scalelayer2->blobs()[0].get(),5);
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[2*2+1].get(),Scalelayer2->blobs()[1].get(),5);
  //ReLU2
  ReLULayer<Dtype>* ReLUlayer2 = new ReLULayer<Dtype>(*layerParamPtr); 
  ReLUlayer2->SetUp(postScale2Vec,postReLU2Vec);
  //Conv2
  ConvolutionLayer<Dtype>* Convlayer2 = new ConvolutionLayer<Dtype>(*layerParamPtr);
  Convlayer2->SetUp(postReLU2Vec,postConv2Vec); 
  BlobDataMemcpy<Dtype>(DBLayerPtr->blobs()[0*2+1].get(),Convlayer2->blobs()[0].get(),90); 
  //Concat1  
  //Concat2
  ConcatLayer<Dtype>* Concatlayer2 = new ConcatLayer<Dtype>(*layerParamPtr);
  Concatlayer2->SetUp(preConcat2Vec,top); 
  //Forward
  string dir_root = "TC_TrueFwdlog";
  BNlayer1->Forward(bottom,postBN1Vec);
  string postBN1_dir = dir_root+"/postBN1";
  logBlob(postBN1Vec[0],postBN1_dir);
  Scalelayer1->Forward(postBN1Vec,postScale1Vec);
  string postScale1_dir = dir_root+"/postScale1";
  logBlob(postScale1Vec[0],postScale1_dir);
  ReLUlayer1->Forward(postScale1Vec,postReLU1Vec);
  string postReLU1_dir = dir_root+"/postReLU1";
  logBlob(postReLU1Vec[0],postReLU1_dir);
  Convlayer1->Forward(postReLU1Vec,postConv1Vec); 
  string postConv1_dir = dir_root+"/postConv1";
  logBlob(postConv1Vec[0],postConv1_dir);
  Concatlayer1->Forward(preConcat1Vec,postConcat1Vec);
  string postConcat1_dir = dir_root+"/postConcat1";
  logBlob(postConcat1Vec[0],postConcat1_dir);

  BNlayer2->Forward(postConcat1Vec,postBN2Vec);
  string postBN2_dir = dir_root+"/postBN2";
  logBlob(postBN2Vec[0],postBN2_dir); 
  Scalelayer2->Forward(postBN2Vec,postScale2Vec); 
  string postScale2_dir = dir_root+"/postScale2";
  logBlob(postScale2Vec[0],postScale2_dir); 
  ReLUlayer2->Forward(postScale2Vec,postReLU2Vec);
  string postReLU2_dir = dir_root+"/postReLU2";
  logBlob(postReLU2Vec[0],postReLU2_dir); 
  Convlayer2->Forward(postReLU2Vec,postConv2Vec);
  string postConv2_dir = dir_root+"/postConv2";
  logBlob(postConv2Vec[0],postConv2_dir); 
  Concatlayer2->Forward(preConcat2Vec,top); 
  string postConcat2_dir = dir_root+"/postConcat2";
  logBlob(top[0],postConcat2_dir);
}

template <typename Dtype>
void Simulate_Bwd(vector<Blob<Dtype>*> top,vector<Blob<Dtype>*> bottom){
 
}

TYPED_TEST(DenseBlockLayerTest, TestTrueFwd){
  typedef typename TypeParam::Dtype Dtype;
  DenseBlockParameter* db_param = this->layer_param.mutable_denseblock_param();
  DenseBlockLayer<Dtype>* dbLayer = new DenseBlockLayer<Dtype>(this->layer_param);
  
  shared_ptr<Filler<Dtype> > gaussianFiller(GetFiller<Dtype>(db_param->filter_filler()));
  gaussianFiller->Fill(this->blob_bottom_cpu);
  this->blob_bottom_gpu->CopyFrom(*this->blob_bottom_cpu);
  dbLayer->SetUp(this->bottomVec_gpu,this->topVec_gpu);
  
  Simulate_Fwd<Dtype>(this->bottomVec_cpu,this->topVec_cpu,dbLayer,&(this->layer_param));
  dbLayer->Forward(this->bottomVec_gpu,this->topVec_gpu);

  for (int n=0;n<2;++n){
    for (int c=0;c<7;++c){
      for (int h=0;h<5;++h){
        for (int w=0;w<5;++w){
          EXPECT_NEAR(this->blob_top_cpu->data_at(n,c,h,w),this->blob_top_gpu->data_at(n,c,h,w),0.2);	
	}
      }
    }
  }

  delete dbLayer;
}

/*
TYPED_TEST(DenseBlockLayerTest, TestSpeed){
  typedef typename TypeParam::Dtype Dtype;
  DenseBlockParameter* bigDB_param = this->bigLayer_param.mutable_denseblock_param();
  DenseBlockLayer<Dtype>* layer5=new DenseBlockLayer<Dtype>(this->bigLayer_param);
  shared_ptr<Filler<Dtype> > gaussianFiller(GetFiller<Dtype>(bigDB_param->filter_filler()));
  //bottom fill
  gaussianFiller->Fill(this->bigBlob_bottom_cpu);
  this->bigBlob_bottom_gpu->CopyFrom(*this->bigBlob_bottom_cpu);
  vector<bool> propagate_down(1,true);
  layer5->SetUp(this->bigBottomVec_gpu,this->bigTopVec_gpu);
  layer5->Forward_gpu_public(this->bigBottomVec_gpu,this->bigTopVec_gpu);
  layer5->Backward_gpu_public(this->bigTopVec_gpu,propagate_down,this->bigBottomVec_gpu);
}
*/
}  // namespace caffe
