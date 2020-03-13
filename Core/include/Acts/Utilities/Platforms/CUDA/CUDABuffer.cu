#pragma once

#include <iostream>
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"

namespace Acts{

template<typename varT>
class CUDABuffer{

public:
  
  CUDABuffer(int size){ 
    fSize = size;
    cudaMalloc((void**)&fDevPtr, fSize*sizeof(varT));
  }

  SetData(varT* buffer, int len, int offset, cudaStream_t& stream){
    cudaMemcpyAsync(fDevPtr+offset, buffer, len*sizeof(varT), cudaMemcpyHostToDevice,stream);
  }

  SetData(const varT* buffer, int len, int offset, cudaStream_t& stream){
    cudaMemcpyAsync(fDevPtr+offset, buffer, len*sizeof(varT), cudaMemcpyHostToDevice,stream); 
  }

  SetData(varT* buffer, int len, int offset=0){
    cudaMemcpy(fDevPtr+offset, buffer, len*sizeof(varT), cudaMemcpyHostToDevice);
  }

  SetData(const varT* buffer, int len, int offset=0){
    cudaMemcpy(fDevPtr+offset, buffer, len*sizeof(varT), cudaMemcpyHostToDevice);
  }

  ~CUDABuffer(){ 
    cudaFree(fDevPtr); 
  }

  varT* dataHost(int len, int offset=0) const {
    varT* hostPtr = new varT[len];
    cudaMemcpy(hostPtr, fDevPtr+offset, len*sizeof(varT), cudaMemcpyDeviceToHost);   
    return hostPtr;
  }

  varT* data(int offset=0) const{ return fDevPtr+offset; }

private:

  varT* fDevPtr; 
  int   fSize;
};
}
