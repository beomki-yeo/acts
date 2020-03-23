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

  CUDABuffer(int size, varT* buffer, int offset=0){ 
    fSize = size;
    cudaMalloc((void**)&fDevPtr, fSize*sizeof(varT));
    cudaMemcpy(fDevPtr+offset, buffer, fSize*sizeof(varT), cudaMemcpyHostToDevice);
  }
  
  CUDABuffer(int size, const varT* buffer, int offset=0){ 
    fSize = size;
    cudaMalloc((void**)&fDevPtr, fSize*sizeof(varT));
    cudaMemcpy(fDevPtr+offset, buffer, fSize*sizeof(varT), cudaMemcpyHostToDevice);
  }
  
  void SetData(varT* buffer, int len, int offset, cudaStream_t& stream){
    cudaMemcpyAsync(fDevPtr+offset, buffer, len*sizeof(varT), cudaMemcpyHostToDevice,stream);
  }

  void SetData(const varT* buffer, int len, int offset, cudaStream_t& stream){
    cudaMemcpyAsync(fDevPtr+offset, buffer, len*sizeof(varT), cudaMemcpyHostToDevice,stream); 
  }

  void SetData(varT* buffer, int len, int offset=0){
    cudaMemcpy(fDevPtr+offset, buffer, len*sizeof(varT), cudaMemcpyHostToDevice);
  }

  void SetData(const varT* buffer, int len, int offset=0){
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
