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
#include "caffe/layers/DenseBlock_layer.hpp"

namespace caffe {

template <typename TypeParam>
class DenseBlockLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DenseBlockLayerTest()
      : blob_bottom_cpu(new Blob<Dtype>(2,3,5,5)),
        blob_top_cpu(new Blob<Dtype>(2,2,5,5)),
	blob_bottom_gpu(new Blob<Dtype>(2,3,5,5)),
	blob_top_gpu(new Blob<Dtype>(2,2,5,5))
  {
    Caffe::set_random_seed(1701);
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
    this->idIdx = 1;
    this->bottomVec_gpu.push_back(this->blob_bottom_gpu);
    this->bottomVec_cpu.push_back(this->blob_bottom_cpu);
    this->topVec_gpu.push_back(this->blob_top_gpu);
    this->topVec_cpu.push_back(this->blob_top_cpu);
  }
  virtual ~DenseBlockLayerTest() {}
  int idIdx;
  LayerParameter layer_param;
  Blob<Dtype>* blob_bottom_cpu;
  Blob<Dtype>* blob_top_cpu;
  Blob<Dtype>* blob_bottom_gpu;
  Blob<Dtype>* blob_top_gpu;

  vector<Blob<Dtype>*> bottomVec_cpu;
  vector<Blob<Dtype>*> topVec_cpu;
  vector<Blob<Dtype>*> bottomVec_gpu;
  vector<Blob<Dtype>*> topVec_gpu;
};

void writeHelloWorld(){
  //std::ofstream testOut("HelloWorld.txt",std::ofstream::out);
  //testOut<< "Hello WOrld"<<endl;
  std::cout<< boost::filesystem::exists("hello") <<std::endl;
  boost::filesystem::path dir("hello/world/hahaha");
  boost::filesystem::create_directories(dir);
}

TYPED_TEST_CASE(DenseBlockLayerTest, TestDtypesAndDevices);

TYPED_TEST(DenseBlockLayerTest, TestDenseBlock) {
  typedef typename TypeParam::Dtype Dtype;
  //test
  writeHelloWorld();
  DenseBlockParameter* db_param = this->layer_param.mutable_denseblock_param();
  shared_ptr<DenseBlockLayer<Dtype> > layer(new DenseBlockLayer<Dtype>(this->layer_param));
  shared_ptr<DenseBlockLayer<Dtype> > layer2(new DenseBlockLayer<Dtype>(this->layer_param));
  
  shared_ptr<Filler<Dtype> > gaussianFiller(GetFiller<Dtype>(db_param->bn_scaler_filler()));
  gaussianFiller->Fill(this->blob_bottom_cpu);
  this->blob_bottom_gpu->CopyFrom(*this->blob_bottom_cpu);
  
  layer->SetUp(this->bottomVec_cpu,this->topVec_cpu);
  layer2->SetUp(this->bottomVec_gpu,this->topVec_gpu);

  layer->setLogId(this->idIdx);
  this->idIdx += 1;
  layer2->setLogId(this->idIdx);
  this->idIdx += 1;

  std::cout<< "HelloWorld" << std::endl;
  //synchronize the random filled parameters of layer and layers
  layer2->syncBlobs(layer.get());

  layer->Forward_cpu_public(this->bottomVec_cpu,this->topVec_cpu);
  layer2->Forward(this->bottomVec_gpu,this->topVec_gpu);

  for (int n=0;n<2;++n){
    for (int c=0;c<2;++c){
      for (int h=0;h<5;++h){
        for (int w=0;w<5;++w){
	  EXPECT_NEAR(this->blob_top_cpu->data_at(n,c,h,w),this->blob_top_gpu->data_at(n,c,h,w),0.1);
	}
      }
    }
  }
}

//TYPED_TEST(DenseBlockLayerTest, TestDenseBlockGradient) {
// 
//}

}  // namespace caffe
