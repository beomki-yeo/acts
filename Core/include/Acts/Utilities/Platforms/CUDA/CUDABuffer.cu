#pragma once

#include <iostream>
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"

namespace Acts{

template<typename Var_t>
class CUDABuffer{

public:
  
  CUDABuffer(int size){ 
    fSize = size;
    cudaMalloc((void**)&fDevPtr, fSize*sizeof(Var_t));
  }

  CUDABuffer(int size, Var_t* buffer, int offset=0){ 
    fSize = size;
    cudaMalloc((void**)&fDevPtr, fSize*sizeof(Var_t));
    cudaMemcpy(fDevPtr+offset, buffer, fSize*sizeof(Var_t), cudaMemcpyHostToDevice);
  }
  
  CUDABuffer(int size, const Var_t* buffer, int offset=0){ 
    fSize = size;
    cudaMalloc((void**)&fDevPtr, fSize*sizeof(Var_t));
    cudaMemcpy(fDevPtr+offset, buffer, fSize*sizeof(Var_t), cudaMemcpyHostToDevice);
  }

  void SetData(Var_t* buffer, int len, int offset=0){
    cudaMemcpy(fDevPtr+offset, buffer, len*sizeof(Var_t), cudaMemcpyHostToDevice);
  }

  void SetData(const Var_t* buffer, int len, int offset=0){
    cudaMemcpy(fDevPtr+offset, buffer, len*sizeof(Var_t), cudaMemcpyHostToDevice);
  }
  
  void SetData(Var_t* buffer, int len, int offset, cudaStream_t& stream){
    cudaMemcpyAsync(fDevPtr+offset, buffer, len*sizeof(Var_t), cudaMemcpyHostToDevice,stream);
  }

  void SetData(const Var_t* buffer, int len, int offset, cudaStream_t& stream){
    cudaMemcpyAsync(fDevPtr+offset, buffer, len*sizeof(Var_t), cudaMemcpyHostToDevice,stream); 
  }

  ~CUDABuffer(){ 
    cudaFree(fDevPtr); 
  }

  Var_t* dataHost(int len, int offset=0) const {
    Var_t* hostPtr = new Var_t[len];
    cudaMemcpy(hostPtr, fDevPtr+offset, len*sizeof(Var_t), cudaMemcpyDeviceToHost);   
    return hostPtr;
  }
  
  Var_t* data(int offset=0) const{ return fDevPtr+offset; }

private:

  Var_t* fDevPtr; 
  int   fSize;
};
}
