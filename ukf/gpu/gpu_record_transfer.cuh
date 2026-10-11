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

inline bool download_compact_records(double* destination, const void* source, int max_rec, int width,
                                     const std::vector<size_t>& offsets, const size_t* device_offsets,
                                     double* device_buffer, std::vector<double>& buffer, std::string& err)
{
  const size_t count = offsets.back()*width;
  for (size_t begin = 0; begin < count; begin += buffer.size())
    {
    const size_t end = std::min(begin+buffer.size(),count);
    const int first = std::upper_bound(offsets.begin(),offsets.end(),begin/width)-offsets.begin()-1;
    const int last = std::upper_bound(offsets.begin(),offsets.end(),(end-1)/width)-offsets.begin()-1;
    k_pack_records<<<last-first+1,128>>>(static_cast<const double*>(source),device_buffer,device_offsets,
                                        max_rec,width,first,begin,end);
    CHK(cudaGetLastError());
    CHK(cudaMemcpy(buffer.data(),device_buffer,(end-begin)*sizeof(double),cudaMemcpyDeviceToHost));
    for (int fiber = first; fiber <= last; ++fiber)
      {
      const size_t start = offsets[fiber]*width;
      const size_t lo = std::max(start,begin), hi = std::min(offsets[fiber+1]*width,end);
      if (hi > lo)
        { std::memcpy(destination+(size_t)fiber*max_rec*width+lo-start,buffer.data()+lo-begin,(hi-lo)*sizeof(double)); }
      }
    }
  return true;
}
#endif
