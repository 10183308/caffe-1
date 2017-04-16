#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
#include <stdlib.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  random_order = this->layer_param_.data_param().random_order();
  init_seed = this->layer_param_.data_param().init_seed();
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
  //if in random order mode
  if (random_order){
    localKey = 0;
    int localI = 0;
    while (cursor_->valid()){
      Datum* localDatum = new Datum();
      localDatum->ParseFromString(cursor_->value());
      allDatum.push_back(localDatum); 
      indexVec.push_back(localI);
      localI++;
      cursor_->Next();
    } 
    //std::cout<<"DataRandomInitDone"<<std::endl;
  }
}

int localRandom(int rangeI){return std::rand()%rangeI;}

template <typename Dtype>
void DataLayer<Dtype>::localShuffle(){
  std::srand(this->init_seed);
  std::random_shuffle(indexVec.begin(),indexVec.end(),localRandom);
  this->init_seed++;
}

template <typename Dtype>
bool DataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep;
  if (random_order){
    keep = (localKey % size) == rank || 
	      this->layer_param_.phase() == TEST;
 
  }
  else {
    keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  }
  return !keep;
}

template<typename Dtype>
void DataLayer<Dtype>::Next() {
  if (random_order){
    localKey++;
    //reached the end, shuffle
    //std::cout<<"lK:"<<localKey<<",indexVecSize"<<indexVec.size()<<std::endl;
    if (localKey >= indexVec.size()){
      //LOG_IF(INFO, Caffe::root_solver())
      LOG_IF(INFO, Caffe::root_solver())
          << "Randomizing data prefetching from start.";
      localShuffle();
      localKey = 0;
    } 
  }
  else {
    cursor_->Next();
    if (!cursor_->valid()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
    offset_++;
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  //Datum datum;
  Datum* datumP = new Datum();
  //std::cout<< "Pre Batch of loadbatch"<<std::endl;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }
    //std::cout<< "After Skip and Next"<<std::endl;
    //std::cout<< "AllDatum Len:"<<allDatum.size()<<",indexVec Len:"<<indexVec.size()<<std::endl;
    if (random_order){datumP = (allDatum[indexVec[localKey]]);}
    else {datumP->ParseFromString(cursor_->value());}
    read_time += timer.MicroSeconds();
    //std::cout<< "Get Datum"<<std::endl;
    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(*datumP);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(*datumP, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      Dtype* top_label = batch->label_.mutable_cpu_data();
      top_label[item_id] = datumP->label();
    }
    trans_time += timer.MicroSeconds();
    Next();
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
