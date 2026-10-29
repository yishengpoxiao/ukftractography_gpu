// Copy recorded values without transferring unused fixed-stride capacity.
#ifndef UKF_GPU_RECORD_TRANSFER_CUH
#define UKF_GPU_RECORD_TRANSFER_CUH
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

__global__ void k_pack_records(const double* source, double* packed, const size_t* offsets,
                               int max_rec, int width, int first_fiber, size_t begin, size_t end)
{
  const int fiber = first_fiber+blockIdx.x;
  const size_t start = offsets[fiber]*width, finish = offsets[fiber+1]*width;
  const size_t lo = start>begin ? start : begin, hi = finish<end ? finish : end;
  for (size_t i = lo+threadIdx.x; i < hi; i += blockDim.x)
    { packed[i-begin] = source[(size_t)fiber*max_rec*width+i-start]; }
}

// Two bounded staging slots overlap the next GPU pack/copy with host scatter.
// The caller completes tracking before downloading and reuses the slots across batches.
class RecordTransfer
{
  struct Slot
    {
    double* device = nullptr;
    double* host = nullptr;
    cudaStream_t stream = nullptr;
    } slots[2];
  size_t capacity = 0;
  cudaEvent_t ready = nullptr;

public:
  RecordTransfer() = default;
  RecordTransfer(const RecordTransfer&) = delete;
  RecordTransfer& operator=(const RecordTransfer&) = delete;
  ~RecordTransfer()
    {
    for (auto& slot : slots)
      {
      if (slot.stream) { cudaStreamSynchronize(slot.stream); }
      if (slot.device) { cudaFree(slot.device); }
      if (slot.host) { cudaFreeHost(slot.host); }
      if (slot.stream) { cudaStreamDestroy(slot.stream); }
      }
    if (ready) { cudaEventDestroy(ready); }
    }

  bool initialize(size_t values, std::string& err)
    {
    capacity = values;
    CHK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    for (auto& slot : slots)
      {
      CHK(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking));
      CHK(cudaMalloc(reinterpret_cast<void**>(&slot.device), capacity*sizeof(double)));
      CHK(cudaMallocHost(reinterpret_cast<void**>(&slot.host), capacity*sizeof(double)));
      }
    return true;
    }

  bool download(double* destination, const void* source, int max_rec, int width,
                const std::vector<size_t>& offsets, const size_t* device_offsets,
                bool compact, std::string& err)
    {
    const size_t count = (compact ? offsets.back() : (offsets.size()-1)*max_rec)*width;
    const size_t pieces = (count+capacity-1)/capacity;
    // Include the offset upload: pageable H2D copies may return before DMA ends.
    CHK(cudaEventRecord(ready, nullptr));
    for (auto& slot : slots) { CHK(cudaStreamWaitEvent(slot.stream, ready, 0)); }
    auto enqueue = [&](size_t piece)
      {
      auto& slot = slots[piece%2];
      const size_t begin = piece*capacity, end = std::min(begin+capacity,count);
      const double* data = static_cast<const double*>(source)+begin;
      if (compact)
        {
        const int first = std::upper_bound(offsets.begin(),offsets.end(),begin/width)-offsets.begin()-1;
        const int last = std::upper_bound(offsets.begin(),offsets.end(),(end-1)/width)-offsets.begin()-1;
        k_pack_records<<<last-first+1,128,0,slot.stream>>>(static_cast<const double*>(source),
          slot.device,device_offsets,max_rec,width,first,begin,end);
        CHK(cudaGetLastError());
        data = slot.device;
        }
      CHK(cudaMemcpyAsync(slot.host,data,(end-begin)*sizeof(double),cudaMemcpyDeviceToHost,slot.stream));
      return true;
      };
    for (size_t piece = 0; piece < std::min(size_t(2),pieces); ++piece)
      { if (!enqueue(piece)) { return false; } }
    for (size_t piece = 0; piece < pieces; ++piece)
      {
      auto& slot = slots[piece%2];
      const size_t begin = piece*capacity, end = std::min(begin+capacity,count);
      CHK(cudaStreamSynchronize(slot.stream));
      if (compact)
        {
        const int first = std::upper_bound(offsets.begin(),offsets.end(),begin/width)-offsets.begin()-1;
        const int last = std::upper_bound(offsets.begin(),offsets.end(),(end-1)/width)-offsets.begin()-1;
        for (int fiber = first; fiber <= last; ++fiber)
          {
          const size_t start = offsets[fiber]*width;
          const size_t lo = std::max(start,begin), hi = std::min(offsets[fiber+1]*width,end);
          if (hi > lo)
            { std::memcpy(destination+(size_t)fiber*max_rec*width+lo-start,slot.host+lo-begin,(hi-lo)*sizeof(double)); }
          }
        }
      else { std::memcpy(destination+begin,slot.host,(end-begin)*sizeof(double)); }
      if (piece+2 < pieces && !enqueue(piece+2)) { return false; }
      }
    return true;
    }
};
#endif
