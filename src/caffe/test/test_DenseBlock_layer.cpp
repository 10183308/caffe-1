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
class DenseBlockLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DenseBlockLayerTest()
      : blob_bottom_cpu(new Blob<Dtype>(2,3,5,5)),
        blob_top_cpu(new Blob<Dtype>(2,7,5,5)),
	blob_bottom_gpu(new Blob<Dtype>(2,3,5,5)),
	blob_top_gpu(new Blob<Dtype>(2,7,5,5)),
	bigBlob_bottom_cpu(new Blob<Dtype>(64,big_initC,100,100)),
	bigBlob_top_cpu(new Blob<Dtype>(64,big_initC+big_growthRate*big_numTransition,100,100)),
	bigBlob_bottom_gpu(new Blob<Dtype>(64,big_initC,100,100)),
	bigBlob_top_gpu(new Blob<Dtype>(64,big_initC+big_growthRate*big_numTransition,100,100))
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
    db_param->mutable_bn_scaler_filler()->set_type("gaussian");
    db_param->mutable_bn_bias_filler()->set_type("gaussian");
  
   DenseBlockParameter* bigDB_param = this->layer_param.mutable_denseblock_param();
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
    bigDB_param->mutable_bn_scaler_filler()->set_type("gaussian");
    bigDB_param->mutable_bn_bias_filler()->set_type("gaussian");
   
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
  DenseBlockLayer<Dtype>* layer=new DenseBlockLayer<Dtype>(this->layer_param);
  DenseBlockLayer<Dtype>* layer2=new DenseBlockLayer<Dtype>(this->layer_param);
  
  shared_ptr<Filler<Dtype> > gaussianFiller(GetFiller<Dtype>(db_param->bn_scaler_filler()));
  gaussianFiller->Fill(this->blob_bottom_cpu);
  this->blob_bottom_gpu->CopyFrom(*this->blob_bottom_cpu);
  
  layer->SetUp(this->bottomVec_cpu,this->topVec_cpu);
  layer2->SetUp(this->bottomVec_gpu,this->topVec_gpu);

  layer->setLogId(global_id);
  global_id += 1;
  layer2->setLogId(global_id);
  global_id += 1;

  //synchronize the random filled parameters of layer and layers
  layer2->syncBlobs(layer);

  layer->Forward_cpu_public(this->bottomVec_cpu,this->topVec_cpu);
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
  shared_ptr<Filler<Dtype> > gaussianFiller(GetFiller<Dtype>(db_param->bn_scaler_filler()));
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
  for (int n=0;n<2;++n){
    for (int c=0;c<3;++c){
      for (int h=0;h<5;++h){
        for (int w=0;w<5;++w){
	  EXPECT_NEAR(this->blob_bottom_cpu->diff_at(n,c,h,w),this->blob_bottom_gpu->diff_at(n,c,h,w),0.4);
	}
      }
    }
  }

}
/*
TYPED_TEST(DenseBlockLayerTest, TestSpeed){
  typedef typename TypeParam::Dtype Dtype;
  DenseBlockParameter* bigDB_param = this->bigLayer_param.mutable_denseblock_param();
  DenseBlockLayer<Dtype>* layer5=new DenseBlockLayer<Dtype>(this->bigLayer_param);
  shared_ptr<Filler<Dtype> > gaussianFiller(GetFiller<Dtype>(bigDB_param->bn_scaler_filler()));
  //bottom fill
  gaussianFiller->Fill(this->bigBlob_bottom_cpu);
  this->bigBlob_bottom_gpu->CopyFrom(*this->bigBlob_bottom_cpu);
  vector<bool> propagate_down(1,true);
  std::cout<<"preSetup"<<std::endl;
  layer5->SetUp(this->bigBottomVec_gpu,this->bigTopVec_gpu);
  std::cout<<"postSetup"<<std::endl;
  clock_t begin1 = std::clock();
  layer5->Forward(this->bigBottomVec_gpu,this->bigTopVec_gpu);
  clock_t end1 = std::clock();
  double elapsed_sec1 = double(end1-begin1) / CLOCKS_PER_SEC;
  std::cout<<"elapsed time 1:"<<elapsed_sec1<<std::endl;
  clock_t begin2 = std::clock();
  layer5->Backward(this->bigTopVec_gpu,propagate_down,this->bigBottomVec_gpu);
  clock_t end2 = std::clock();
  double elapsed_sec2 = double(end2-begin2) / CLOCKS_PER_SEC;
  std::cout<<"elapsed time 2:"<<elapsed_sec2<<std::endl;
}
*/
}  // namespace caffe
