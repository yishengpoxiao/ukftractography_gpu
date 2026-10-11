#include <cuda_runtime.h>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>
#define CHK(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) { err = cudaGetErrorString(e_); return false; } } while (0)
#include "gpu_record_transfer.cuh"

bool check(int width,size_t capacity,std::string& err)
{
  const int max_rec=17, counts[]={0,1,7,0,17,3,0};
  const size_t fibers=sizeof(counts)/sizeof(counts[0]), values=fibers*max_rec*width, guard=16;
  std::vector<size_t> offsets(fibers+1,0);
  std::vector<double> source(values,std::numeric_limits<double>::quiet_NaN());
  std::vector<double> actual(values+2*guard,-777), expected=actual, buffer(capacity);
  for (size_t f=0;f<fibers;++f)
    {
    offsets[f+1]=offsets[f]+counts[f];
    for (int i=0;i<counts[f]*width;++i)
      { const size_t index=f*max_rec*width+i; source[index]=index+0.125; expected[guard+index]=source[index]; }
    }
  double *device_source=NULL,*device_buffer=NULL; size_t* device_offsets=NULL;
  CHK(cudaMalloc(&device_source,values*sizeof(double)));
  CHK(cudaMalloc(&device_buffer,capacity*sizeof(double)));
  CHK(cudaMalloc(&device_offsets,offsets.size()*sizeof(size_t)));
  CHK(cudaMemcpy(device_source,source.data(),values*sizeof(double),cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(device_offsets,offsets.data(),offsets.size()*sizeof(size_t),cudaMemcpyHostToDevice));
  if (!download_compact_records(actual.data()+guard,device_source,max_rec,width,offsets,
                                 device_offsets,device_buffer,buffer,err)) { return false; }
  CHK(cudaFree(device_source)); CHK(cudaFree(device_buffer)); CHK(cudaFree(device_offsets));
  if (actual!=expected) { err="record values, unused slots or output canaries changed"; return false; }
  return true;
}

int main()
{
  std::string err;
  for (int width : {1,3,5,10,18,324})
    { for (size_t capacity : {7,31,1024})
      { if (!check(width,capacity,err)) { fprintf(stderr,"FAIL width=%d capacity=%zu: %s\n",width,capacity,err.c_str()); return 1; } } }
  printf("PASS: 18 compact-transfer cases; zero/short/full rows, split records and output canaries\n");
}
