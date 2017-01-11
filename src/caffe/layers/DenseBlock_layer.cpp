#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
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

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/DenseBlock_layer.hpp"

namespace caffe {

  bool dirExists(string dirStr){
    const char* dirCStr = dirStr.c_str();
    DIR* dir = opendir(dirCStr);
    if (ENOENT == errno){
      return false;
    }
    closedir(dir);
    return true;
  }

  void tryCreateDirectory(string fileName){
    vector<string> strVec;
    boost::split(strVec,fileName,boost::is_any_of("/"));
    string newStr="";
    for (int i=0;i<strVec.size()-1;++i){
      newStr += strVec[i] + (i==strVec.size()-2?"":"/");
    }
    boost::filesystem::path dirToCreate(newStr);
    if (!dirExists(newStr)){
      boost::filesystem::create_directories(dirToCreate);
    }
  }

  template <typename Dtype>
  void DenseBlockLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	this->cpuInited = false;
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
	//for transition i, 
	//blobs_[i] is its filter blob
	//blobs_[numTransition + i] is its scaler blob
	//blobs_[2*numTransition + i] is its bias blob
        this->blobs_.resize(3*this->numTransition);
	for (int transitionIdx=0;transitionIdx < this->numTransition;++transitionIdx){
	    //filter
	    int inChannels = initChannel + transitionIdx * growthRate;
	    int filterShape_Arr[] = {growthRate,inChannels,filter_H,filter_W};
	    vector<int> filterShape (filterShape_Arr,filterShape_Arr+4);
	    this->blobs_[transitionIdx].reset(new Blob<Dtype>(filterShape));
	    shared_ptr<Filler<Dtype> > filter_Filler(GetFiller<Dtype>(dbParam.filter_filler()));
	    filter_Filler->Fill(this->blobs_[transitionIdx].get());
	    //scaler & bias
	    int numChannel_local = (transitionIdx==0?this->initChannel:this->growthRate); 
	    int BNparamShape_Arr [] = {1,numChannel_local,1,1};
	    vector<int> BNparamShape (BNparamShape_Arr,BNparamShape_Arr+4);
	    //scaler
	    this->blobs_[numTransition + transitionIdx].reset(new Blob<Dtype>(BNparamShape));
	    shared_ptr<Filler<Dtype> > weight_filler0(GetFiller<Dtype>(dbParam.bn_scaler_filler()));
	    weight_filler0->Fill(this->blobs_[numTransition+transitionIdx].get());
	    //bias
	    this->blobs_[2*numTransition + transitionIdx].reset(new Blob<Dtype>(BNparamShape));
	    shared_ptr<Filler<Dtype> > weight_filler1(GetFiller<Dtype>(dbParam.bn_bias_filler()));
	    weight_filler1->Fill(this->blobs_[2*numTransition+transitionIdx].get()); 
	}
	
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){ 
        this->N = bottom[0]->shape()[0]; 
        this->H = bottom[0]->shape()[2];
        this->W = bottom[0]->shape()[3];
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::syncBlobs(DenseBlockLayer<Dtype>* originLayer){
    vector<shared_ptr<Blob<Dtype> > >& originBlobs = originLayer->blobs();
    for (int blobIdx=0;blobIdx < originBlobs.size();++blobIdx){
      shared_ptr<Blob<Dtype> > localBlob = originBlobs[blobIdx];
      Blob<Dtype> * newBlob = new Blob<Dtype>(localBlob->shape());
      newBlob->CopyFrom(*(localBlob.get()),true);
      shared_ptr<Blob<Dtype> > sharedPtrBlob(newBlob);
      this->blobs_[blobIdx] = sharedPtrBlob;
    }
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::setLogId(int uid){
    this->logId = uid;
}

template <typename Dtype>
void logBlob(Blob<Dtype>* B,string fileName){
    string dataNameStr = fileName + "_data";
    string gradNameStr = fileName + "_grad";
    std::cout<<dataNameStr;
    std::cout<<gradNameStr;
    const char* dataName = (dataNameStr).c_str();
    const char* gradName = (gradNameStr).c_str();
    
    tryCreateDirectory(dataName);
    tryCreateDirectory(gradName);
    std::ofstream outWriter_data(dataName,std::ofstream::out);
    std::ofstream outWriter_grad(gradName,std::ofstream::out); 
    for (int n=0;n<B->shape(0);++n){
      for (int c=0;c<B->shape(1);++c){
        for (int h=0;h<B->shape(2);++h){
	  for (int w=0;w<B->shape(3);++w){
	    outWriter_data<<B->data_at(n,c,h,w)<<",";
	    outWriter_grad<<B->diff_at(n,c,h,w)<<",";
	  }
	}
      }
    }
    outWriter_data<<std::endl;
    outWriter_grad<<std::endl;
}

string itos(int i){
  string output = boost::lexical_cast<string>(i);
  return output;  
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::logInternal_cpu(string dir){
    string localDir = dir+"/cpu_"+itos(this->logId)+"/"; 
    //global_Mean
    for (int i=0;i<this->global_Mean.size();++i){
      string blobStr = localDir+"global_Mean_"+itos(i);
      logBlob(this->global_Mean[i],blobStr);
    }
    //batch_Mean
    for (int i=0;i<this->batch_Mean.size();++i){
      string blobStr = localDir+"batch_Mean_"+itos(i);
      logBlob(this->batch_Mean[i],blobStr);
    }
    //global_Var
    for (int i=0;i<this->global_Var.size();++i){
      string blobStr = localDir+"global_Var_"+itos(i);
      logBlob(this->global_Var[i],blobStr);
    }
    //batch_Var
    for (int i=0;i<this->batch_Var.size();++i){
      string blobStr = localDir+"batch_Var_"+itos(i);
      logBlob(this->batch_Var[i],blobStr);
    }
    //merged_conv
    for (int i=0;i<this->merged_conv.size();++i){
      string blobStr = localDir+"merged_conv_"+itos(i);
      logBlob(this->merged_conv[i],blobStr);
    }
    //BN_XhatVec
    for (int i=0;i<this->BN_XhatVec.size();++i){
      string blobStr = localDir+"BN_XhatVec_"+itos(i);
      logBlob(this->BN_XhatVec[i],blobStr);
    }
    //postBN_blobVec
    for (int i=0;i<this->postBN_blobVec.size();++i){
      string blobStr = localDir+"postBN_blobVec_"+itos(i);
      logBlob(this->postBN_blobVec[i],blobStr);
    }
    //postReLU_blobVec
    for (int i=0;i<this->postReLU_blobVec.size();++i){
      string blobStr = localDir+"postReLU_blobVec_"+itos(i);
      logBlob(this->postReLU_blobVec[i],blobStr);
    }
    //postConv_blobVec
    for (int i=0;i<this->postConv_blobVec.size();++i){
      string blobStr = localDir+"postConv_blobVec_"+itos(i);
      logBlob(this->postConv_blobVec[i],blobStr);
    }
    //filter
    for (int i=0;i<this->numTransition;++i){
      string blobStr = localDir+"filter_"+itos(i);
      logBlob(this->blobs_[i].get(),blobStr);
    }
    //scaler 
    for (int i=0;i<this->numTransition;++i){
      string blobStr = localDir+"scaler_"+itos(i);
      logBlob(this->blobs_[this->numTransition+i].get(),blobStr);
    }
    //bias
    for (int i=0;i<this->numTransition;++i){
      string blobStr = localDir+"bias_"+itos(i);
      logBlob(this->blobs_[this->numTransition*2+i].get(),blobStr);
    }
}

template <typename Dtype>
Dtype getZeroPaddedValue(bool isDiff,Blob<Dtype>* inputData,int n,int c,int h,int w){
    int n_blob = inputData->shape(0);
    int c_blob = inputData->shape(1);
    int h_blob = inputData->shape(2);
    int w_blob = inputData->shape(3);
    if ((n<0) || (n>=n_blob)) return 0;
    if ((c<0) || (c>=c_blob)) return 0;
    if ((h<0) || (h>=h_blob)) return 0;
    if ((w<0) || (w>=w_blob)) return 0;
    if (isDiff) return inputData->diff_at(n,c,h,w);
    else return inputData->data_at(n,c,h,w);
}

//Assumption, h_filter and w_filter must be 3 for now
//naivest possible implementation of convolution, CPU forward and backward should not be used in production.
//CPU version of convolution assume img H,W does not change after convolution, which corresponds to denseBlock without BC
//input of size N*c_input*h_img*w_img
template <typename Dtype>
void convolution_Fwd(Blob<Dtype>* input, Blob<Dtype>* output, Blob<Dtype>* filter,int N,int c_output,int c_input,int h_img,int w_img,int h_filter,int w_filter){
    CHECK_EQ(h_filter,3);
    CHECK_EQ(w_filter,3);
    int outputShape[] = {N,c_output,h_img,w_img};
    vector<int> outputShapeVec (outputShape,outputShape + 4);
    output->Reshape(outputShapeVec);
    Dtype * outputPtr = output->mutable_cpu_data();
    for (int n=0;n<N;++n){
      for (int c_outIdx=0;c_outIdx<c_output;++c_outIdx){
        for (int hIdx=0;hIdx<h_img;++hIdx){
	  for (int wIdx=0;wIdx<w_img;++wIdx){
	    outputPtr[output->offset(n,c_outIdx,hIdx,wIdx)]=0;
	    for (int c_inIdx=0;c_inIdx<c_input;++c_inIdx){
	      for (int filter_x=0;filter_x<h_filter;++filter_x){
	        for (int filter_y=0;filter_y<w_filter;++filter_y){
		  int localX = hIdx + 1 - filter_x;
	          int localY = wIdx + 1 - filter_y;
	          outputPtr[output->offset(n,c_outIdx,hIdx,wIdx)] += (filter->data_at(c_outIdx,c_inIdx,filter_x,filter_y) * getZeroPaddedValue(false,input,n,c_inIdx,localX,localY));
		}
	      } 
	    }
	  }
	}
      }
    }
}

template <typename Dtype>
void convolution_Bwd(Blob<Dtype>* bottom,Blob<Dtype>* top,Blob<Dtype>* filter,int N,int c_output,int c_input,int h_img,int w_img,int h_filter,int w_filter){
    CHECK_EQ(h_filter,3);
    CHECK_EQ(w_filter,3);
    Dtype * filterDiffPtr = filter->mutable_cpu_diff();
    Dtype * bottomDiffPtr = bottom->mutable_cpu_diff();
    //compute FilterGrad
    for (int coutIdx=0;coutIdx<c_output;++coutIdx){
      for (int cinIdx=0;cinIdx<c_input;++cinIdx){
        for (int filter_x=0;filter_x<h_filter;++filter_x){
	  for (int filter_y=0;filter_y<w_filter;++filter_y){
	    Dtype localGradSum=0;
	    for (int n=0;n<N;++n){
	      for (int i_img=0;i_img<h_img;++i_img){
	        for (int j_img=0;j_img<w_img;++j_img){
		  int localX = i_img + 1 - filter_x;
		  int localY = j_img + 1 - filter_y;
		  localGradSum += top->diff_at(n,coutIdx,i_img,j_img) * getZeroPaddedValue(false,bottom,n,cinIdx,localX,localY);
		}
	      } 
	    }
	    filterDiffPtr[filter->offset(coutIdx,cinIdx,filter_x,filter_y)] = localGradSum;
	  }
	}
      }
    } 
    //compute BottomGrad
    for (int n=0;n<N;++n){
      for (int cinIdx=0;cinIdx<c_input;++cinIdx){
        for (int i_img=0;i_img<h_img;++i_img){
	  for (int j_img=0;j_img<w_img;++j_img){
	    Dtype localGradSum=0;
	    for (int coutIdx=0;coutIdx<c_output;++coutIdx){
	      for (int x_img=0;x_img<h_img;++x_img){
	        for (int y_img=0;y_img<w_img;++y_img){
		  int localX = x_img-i_img+1;
		  int localY = y_img-j_img+1;
		  localGradSum += top->diff_at(n,coutIdx,x_img,y_img) * getZeroPaddedValue(false,filter,coutIdx,cinIdx,localX,localY); 
		}
	      }
	    }
	    bottomDiffPtr[bottom->offset(n,cinIdx,i_img,j_img)] = localGradSum;
	  }
	}
      }
    } 
}

template <typename Dtype>
void ReLU_Fwd(Blob<Dtype>* bottom,Blob<Dtype>* top,int N,int C,int h_img,int w_img){
    //Reshape top
    int topShapeArr[] = {N,C,h_img,w_img};
    vector<int> topShapeVec(topShapeArr,topShapeArr+4);
    top->Reshape(topShapeVec);
    //ReLU Fwd
    Dtype* topPtr = top->mutable_cpu_data();
    for (int n=0;n<N;++n){
      for (int cIdx=0;cIdx<C;++cIdx){
        for (int hIdx=0;hIdx<h_img;++hIdx){
	  for (int wIdx=0;wIdx<w_img;++wIdx){
	    topPtr[top->offset(n,cIdx,hIdx,wIdx)] = std::max<Dtype>(0.0,bottom->data_at(n,cIdx,hIdx,wIdx));
	  }
	}
      } 
    }
}

template <typename Dtype>
void ReLU_Bwd(Blob<Dtype>* bottom,Blob<Dtype>* top,int N,int C,int h_img,int w_img){
    Dtype* bottomDiffPtr = bottom->mutable_cpu_diff();
    for (int n=0;n<N;++n){
      for (int cIdx=0;cIdx<C;++cIdx){
        for (int hIdx=0;hIdx<h_img;++hIdx){
	  for (int wIdx=0;wIdx<w_img;++wIdx){
	    bottomDiffPtr[bottom->offset(n,cIdx,hIdx,wIdx)] = bottom->data_at(n,cIdx,hIdx,wIdx)>0?top->diff_at(n,cIdx,hIdx,wIdx):0; 
	  }
	}
      }
    }
}

template <typename Dtype>
Dtype getMean(Blob<Dtype>* A,int channelIdx){
    int N = A->shape(0);
    int H = A->shape(2);
    int W = A->shape(3);
    int totalCount = N*H*W;

    Dtype sum = 0;
    for (int n=0;n<N;++n){
      for (int h=0;h<H;++h){
	for (int w=0;w<W;++w){
          sum += A->data_at(n,channelIdx,h,w);
	}	
      }
    }
    return sum/totalCount;
}

template <typename Dtype>
Dtype getVar(Blob<Dtype>* A,int channelIdx){
    int N = A->shape(0);
    int H = A->shape(2);
    int W = A->shape(3);
    int totalCount = N*H*W;
    Dtype mean = getMean(A,channelIdx);
    
    Dtype sum = 0;
    for (int n=0;n<N;++n){
      for (int h=0;h<H;++h){
        for (int w=0;w<W;++w){
	  sum += (A->data_at(n,channelIdx,h,w)-mean) * (A->data_at(n,channelIdx,h,w)-mean);
	}
      }
    }
    return sum / totalCount;
}

template <typename Dtype>
void BN_inf_Fwd(Blob<Dtype>* input,Blob<Dtype>* output,int N,int C,int h_img,int w_img,Blob<Dtype>* globalMean,Blob<Dtype>* globalVar,Blob<Dtype>* scaler,Blob<Dtype>* bias){
    //Reshape output
    int outputShape[] = {N,C,h_img,w_img};
    vector<int> outputShapeVec(outputShape,outputShape+4);
    output->Reshape(outputShapeVec);
    //BN Fwd inf
    double epsilon = 1e-5;
    Dtype* outputPtr = output->mutable_cpu_data();

    for (int n=0;n<N;++n){
      for (int cIdx=0;cIdx<C;++cIdx){
        Dtype denom = 1.0 / sqrt(globalVar->data_at(0,cIdx,0,0) + epsilon);
	for (int hIdx=0;hIdx<h_img;++hIdx){
	  for (int wIdx=0;wIdx<w_img;++wIdx){
	    outputPtr[output->offset(n,cIdx,hIdx,wIdx)] = scaler->data_at(0,cIdx,0,0) * (denom * (input->data_at(n,cIdx,hIdx,wIdx) - globalMean->data_at(0,cIdx,0,0))) + bias->data_at(0,cIdx,0,0);
	  }
	}
      }
    }
}

template <typename Dtype>
void BN_train_Fwd(Blob<Dtype>* bottom,Blob<Dtype>* top,Blob<Dtype>* output_xhat,Blob<Dtype>* globalMean,Blob<Dtype>* globalVar,Blob<Dtype>* batchMean,Blob<Dtype>* batchVar,Blob<Dtype>* scaler,Blob<Dtype>* bias,int trainCycleIdx,int N,int C,int h_img,int w_img){
    //reshape output
    int outputShape[] = {N,C,h_img,w_img};
    vector<int> outputShapeVec(outputShape,outputShape+4);
    top->Reshape(outputShapeVec);
    output_xhat->Reshape(outputShapeVec);
    //BN Fwd train
    double epsilon = 1e-5;
    double EMA_factor = 1.0 / (1+trainCycleIdx);
    //get batch/global Mean/Var
    for (int channelIdx=0;channelIdx<C;++channelIdx){
      int variance_adjust_m = N*h_img*w_img;
      //batch
      Dtype* batchMean_mutable = batchMean->mutable_cpu_data();
      Dtype* batchVar_mutable = batchVar->mutable_cpu_data();
      batchMean_mutable[channelIdx] = getMean(bottom,channelIdx);
      batchVar_mutable[channelIdx] = (variance_adjust_m / (variance_adjust_m - 1.0)) * getVar(bottom,channelIdx);
      //global
      Dtype* globalMean_mutable = globalMean->mutable_cpu_data();
      Dtype* globalVar_mutable = globalVar->mutable_cpu_data();
      globalMean_mutable[channelIdx] = (1-EMA_factor) * globalMean->data_at(0,channelIdx,0,0) + EMA_factor * batchMean->data_at(0,channelIdx,0,0);
      globalVar_mutable[channelIdx] = (1-EMA_factor) * globalVar->data_at(0,channelIdx,0,0) + EMA_factor * batchVar->data_at(0,channelIdx,0,0);
    }
    //process data
    for (int n=0;n<N;++n){
      for (int c=0;c<C;++c){
        for (int h=0;h<h_img;++h){
	  for (int w=0;w<w_img;++w){
	    Dtype* xhat_mutable = output_xhat->mutable_cpu_data();
	    xhat_mutable[output_xhat->offset(n,c,h,w)] = (bottom->data_at(n,c,h,w) - batchMean->data_at(0,c,0,0))/sqrt(batchVar->data_at(0,c,0,0) + epsilon);
	    Dtype* output_mutable = top->mutable_cpu_data();
	    output_mutable[top->offset(n,c,h,w)] = (scaler->data_at(0,c,0,0)) * (output_xhat->data_at(n,c,h,w)) + bias->data_at(0,c,0,0);
	  }
	}
      }
    }
}

template <typename Dtype>
void BN_train_Bwd(Blob<Dtype>* bottom,Blob<Dtype>* bottom_xhat,Blob<Dtype>* top,Blob<Dtype>* batchMean,Blob<Dtype>* batchVar,Blob<Dtype>* scaler,Blob<Dtype>* bias,int N,int C,int h_img,int w_img){
    double epsilon = 1e-5;
    //bias and scaler grad
    Dtype* biasGrad = bias->mutable_cpu_diff();
    Dtype* scalerGrad = scaler->mutable_cpu_diff();
    for (int channelIdx=0;channelIdx<C;++channelIdx){
      biasGrad[channelIdx] = 0;
      scalerGrad[channelIdx] = 0;
      for (int n=0;n<N;++n){
        for (int hIdx=0;hIdx<h_img;++hIdx){
	  for (int wIdx=0;wIdx<w_img;++wIdx){
	    biasGrad[channelIdx] += top->diff_at(n,channelIdx,hIdx,wIdx);
	    scalerGrad[channelIdx] += top->diff_at(n,channelIdx,hIdx,wIdx) * bottom_xhat->data_at(n,channelIdx,hIdx,wIdx);
	  }
	}
      }
    }
    //bottom data grad
    //helper 1:
    Dtype* XhatGrad = bottom_xhat->mutable_cpu_diff();
    for (int n=0;n<N;++n){
      for (int c=0;c<C;++c){
        for (int h=0;h<h_img;++h){
	  for (int w=0;w<w_img;++w){
	    XhatGrad[bottom_xhat->offset(n,c,h,w)] = top->diff_at(n,c,h,w) * scaler->data_at(0,c,0,0);
	  }
	}
      }
    } 
    //helper 2:
    Dtype* varGrad = batchVar->mutable_cpu_diff();
    for (int c=0;c<C;++c){
      for (int n=0;n<N;++n){
        for (int h=0;h<h_img;++h){
	  for (int w=0;w<w_img;++w){
	    varGrad[c] = bottom_xhat->diff_at(n,c,h,w) * (bottom->data_at(n,c,h,w)-batchMean->data_at(0,c,0,0)) * (-0.5) * pow(batchVar->data_at(0,c,0,0) + epsilon,-1.5); 
	  }
	}
      }
    }

    //helper 3:
    double m = N * h_img * w_img;
    Dtype* meanGrad = batchMean->mutable_cpu_diff();
    for (int c=0;c<C;++c){
      for (int n=0;n<N;++n){
        for (int h=0;h<h_img;++h){
	  for (int w=0;w<w_img;++w){
	    meanGrad[c] = bottom_xhat->diff_at(n,c,h,w) * (-1.0 / sqrt(batchVar->data_at(0,c,0,0) + epsilon)) + batchVar->diff_at(0,c,0,0) * (-2.0) * (bottom->data_at(n,c,h,w) - batchMean->data_at(0,c,0,0)) / m; 
	  }
	}
      }
    }

    //combine helpers
    Dtype* bottomDataGrad = bottom->mutable_cpu_data();
    for (int n=0;n<N;++n){
      for (int c=0;c<C;++c){
        for (int h=0;h<h_img;++h){
	  for (int w=0;w<w_img;++w){
	    Dtype term1=bottom_xhat->diff_at(n,c,h,w)*pow(batchVar->data_at(0,c,0,0)+epsilon,-0.5);
	    Dtype term2=batchVar->diff_at(0,c,0,0)*2.0*(bottom->data_at(n,c,h,w) - batchMean->data_at(0,c,0,0)) / m;
	    Dtype term3=batchMean->diff_at(0,c,0,0)/m;
	    bottomDataGrad[bottom->offset(n,c,h,w)] += term1 + term2 + term3;
	  }
	}
      }
    }
    
 }


template <typename Dtype>
void DenseBlockLayer<Dtype>::CPU_Initialization(){
    this->global_Mean.resize(this->numTransition);
    this->batch_Mean.resize(this->numTransition);
    this->global_Var.resize(this->numTransition);
    this->batch_Var.resize(this->numTransition);
    
    this->merged_conv.resize(this->numTransition);
    this->BN_XhatVec.resize(this->numTransition);
    this->postBN_blobVec.resize(this->numTransition);
    this->postReLU_blobVec.resize(this->numTransition);
    this->postConv_blobVec.resize(this->numTransition);

    for (int transitionIdx=0;transitionIdx<this->numTransition;++transitionIdx){
      int localChannels = transitionIdx==0?this->initChannel:this->growthRate;
      int mergeChannels = this->initChannel + this->growthRate * transitionIdx;
      int channelShapeArr[] = {1,localChannels,1,1};
      int tensorShapeArr[] = {this->N,localChannels,this->H,this->W};
      int mergeShapeArr[] = {this->N,mergeChannels,this->H,this->W};
      vector<int> channelShape(channelShapeArr,channelShapeArr+4);
      vector<int> tensorShape(tensorShapeArr,tensorShapeArr+4);
      vector<int> mergeShape(mergeShapeArr,mergeShapeArr+4);
      this->global_Mean[transitionIdx] = new Blob<Dtype>(channelShape);
      this->batch_Mean[transitionIdx] = new Blob<Dtype>(channelShape);
      this->global_Var[transitionIdx] = new Blob<Dtype>(channelShape);
      this->batch_Var[transitionIdx] = new Blob<Dtype>(channelShape);
      this->merged_conv[transitionIdx] = new Blob<Dtype>(mergeShape);
      this->BN_XhatVec[transitionIdx] = new Blob<Dtype>(tensorShape);
      this->postBN_blobVec[transitionIdx] = new Blob<Dtype>(tensorShape);
      this->postReLU_blobVec[transitionIdx] = new Blob<Dtype>(tensorShape);
      this->postConv_blobVec[transitionIdx] = new Blob<Dtype>(tensorShape);
    }
}

template <typename Dtype>
void mergeData(vector<Blob<Dtype>*>& blobVec,Blob<Dtype>* mergeBlob,int transitionIdx,int N,int initChannel,int growthRate,int H,int W){
    Dtype* mergePtr = mergeBlob->mutable_cpu_data();
    for (int n=0;n<N;++n){
      for (int localT=0;localT<=transitionIdx;++localT){
        int localChannel = localT==0?initChannel:growthRate;
	int localOffset = localT==0?0:initChannel+(localT-1)*growthRate;
	for (int c=0;c<localChannel;++c){
	  int cIdx = c + localOffset; 
	  for (int hIdx=0;hIdx<H;++hIdx){
	    for (int wIdx=0;wIdx<W;++wIdx){
	      mergePtr[mergeBlob->offset(n,cIdx,hIdx,wIdx)] = blobVec[localT]->data_at(n,c,hIdx,wIdx);
	    }
	  }
	}
      }
    }
}

template <typename Dtype>
void distributeGrad(vector<Blob<Dtype>*>& blobVec,Blob<Dtype>* mergeBlob,int transitionIdx,int N,int initChannel,int growthRate,int H,int W){
    for (int n=0;n<N;++n){
      for (int localT=0;localT<=transitionIdx;++localT){
	Dtype* localDiff = blobVec[localT]->mutable_cpu_diff();
	int localChannel = localT==0?initChannel:growthRate;
	int localOffset = localT==0?0:initChannel+(localT-1)*growthRate;
	for (int c=0;c<localChannel;++c){
	  int cIdx = c + localOffset;
	  for (int hIdx=0;hIdx<H;++hIdx){
	    for (int wIdx=0;wIdx<W;++wIdx){
	      localDiff[blobVec[localT]->offset(n,c,hIdx,wIdx)] = mergeBlob->diff_at(n,cIdx,hIdx,wIdx);
	    }
	  }
	}
        
      }
    }
}

template <typename Dtype>
void BlobSetZero(Blob<Dtype>* B,int count){
    Dtype* B_mutable_data = B->mutable_cpu_data();
    Dtype* B_mutable_diff = B->mutable_cpu_diff();
    for (int i=0;i<count;++i) {
      B_mutable_data[i] = 0;
      B_mutable_diff[i] = 0;
    }
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::LoopEndCleanup_cpu(){
    for (int transitionIdx=0;transitionIdx<this->numTransition;++transitionIdx){
      int localChannel = transitionIdx==0?this->initChannel:this->growthRate;
      int tensorCount = this->N * localChannel * this->H * this->W;
      int tensorMergeCount = this->N * (this->initChannel + this->growthRate * transitionIdx) * this->H * this->W;
      BlobSetZero<Dtype>(this->merged_conv[transitionIdx],tensorMergeCount);
      BlobSetZero<Dtype>(this->BN_XhatVec[transitionIdx],tensorCount);
      BlobSetZero<Dtype>(this->postBN_blobVec[transitionIdx],tensorCount);
      BlobSetZero<Dtype>(this->postReLU_blobVec[transitionIdx],tensorCount);
      BlobSetZero<Dtype>(this->postConv_blobVec[transitionIdx],tensorCount);
    }
}

  template <typename Dtype>
  void DenseBlockLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) 
  { 
    //init CPU
    if (!this->cpuInited){   
	this->CPU_Initialization();
        this->cpuInited = true;
    }
    //init CPU finish
    for (int transitionIdx=0;transitionIdx<this->numTransition;++transitionIdx){
      //BN
      Blob<Dtype>* BN_bottom = transitionIdx==0?bottom[0]:postConv_blobVec[transitionIdx-1];
      Blob<Dtype>* BN_top = postBN_blobVec[transitionIdx];
      Blob<Dtype>* Scaler = this->blobs_[numTransition + transitionIdx].get();
      Blob<Dtype>* Bias = this->blobs_[2*numTransition + transitionIdx].get();
      int localChannels = transitionIdx==0?this->initChannel:this->growthRate;
      if (this->phase_ == TEST){
        BN_inf_Fwd<Dtype>(BN_bottom,BN_top,this->N,localChannels,this->H,this->W,this->global_Mean[transitionIdx],this->global_Var[transitionIdx],Scaler,Bias);
      }
      else {
        BN_train_Fwd<Dtype>(BN_bottom,BN_top,this->BN_XhatVec[transitionIdx],this->global_Mean[transitionIdx],this->global_Var[transitionIdx],this->batch_Mean[transitionIdx],this->batch_Var[transitionIdx],Scaler,Bias,this->trainCycleIdx,this->N,localChannels,this->H,this->W);
	this->trainCycleIdx += 1;
      }
      //ReLU
      Blob<Dtype>* ReLU_top = this->postReLU_blobVec[transitionIdx];
      ReLU_Fwd<Dtype>(BN_top,ReLU_top,this->N,localChannels,this->H,this->W);
      //Pre-Conv
      mergeData<Dtype>(this->postReLU_blobVec,this->merged_conv[transitionIdx],transitionIdx,this->N,this->initChannel,this->growthRate,this->H,this->W);
      //Conv
      Blob<Dtype>* filterBlob = this->blobs_[transitionIdx].get();
      Blob<Dtype>* topConv = transitionIdx==this->numTransition-1?top[0]:postConv_blobVec[transitionIdx];
      int inConvChannel = this->initChannel + this->growthRate * transitionIdx;
      convolution_Fwd<Dtype>(this->merged_conv[transitionIdx],topConv,filterBlob,this->N,this->growthRate,inConvChannel,this->H,this->W,this->filter_H,this->filter_W); 
    }
  }

  template <typename Dtype>
  void DenseBlockLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<Dtype>*>& bottom) 
  {
    if (!this->cpuInited){
    	this->CPU_Initialization();
        this->cpuInited = true;
    }
    for (int transitionIdx=this->numTransition-1;transitionIdx>=0;--transitionIdx){
      //Conv Bwd
      Blob<Dtype>* conv_top=transitionIdx==this->numTransition-1?top[0]:this->postConv_blobVec[transitionIdx+1];
      Blob<Dtype>* conv_bottom=merged_conv[transitionIdx];
      Blob<Dtype>* filter = this->blobs_[transitionIdx].get();
      int c_input = this->initChannel + this->growthRate * transitionIdx;
      convolution_Bwd<Dtype>(conv_bottom,conv_top,filter,this->N,this->growthRate,c_input,this->H,this->W,this->conv_verticalStride,this->conv_horizentalStride);
      //Conv postprocessing
      distributeGrad<Dtype>(this->postReLU_blobVec,merged_conv[transitionIdx],transitionIdx,this->N,this->initChannel,this->growthRate,this->H,this->W); 
      //ReLU Bwd
      int localChannel = transitionIdx==0?this->initChannel:this->growthRate;
      ReLU_Bwd<Dtype>(postBN_blobVec[transitionIdx],postReLU_blobVec[transitionIdx],this->N,localChannel,this->H,this->W); 
      //BN Bwd
      Blob<Dtype>* BN_bottom = transitionIdx==0?bottom[0]:postConv_blobVec[transitionIdx];
      Blob<Dtype>* scaler = this->blobs_[this->numTransition+transitionIdx].get();
      Blob<Dtype>* bias = this->blobs_[2*this->numTransition+transitionIdx].get();
      BN_train_Bwd<Dtype>(BN_bottom,this->BN_XhatVec[transitionIdx],this->postBN_blobVec[transitionIdx],this->batch_Mean[transitionIdx],this->batch_Var[transitionIdx],scaler,bias,this->N,localChannel,this->H,this->W);
    }
    this->logInternal_cpu("TClog");
    this->LoopEndCleanup_cpu(); 
  }

template <typename Dtype>
void DenseBlockLayer<Dtype>::Forward_cpu_public(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
  this->Forward_cpu(bottom,top);
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Backward_cpu_public(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
  this->Backward_cpu(top,propagate_down,bottom);
}

#ifdef CPU_ONLY
STUB_GPU(DenseBlockLayer);
#endif

INSTANTIATE_CLASS(DenseBlockLayer);
REGISTER_LAYER_CLASS(DenseBlock);

}  // namespace caffe  
