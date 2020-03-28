#pragma once

#include <iostream>
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"
#include "Acts/Utilities/Platforms/CPU/CPUBuffer.hxx"

namespace Acts{

template<typename Var_t>
class CUDABuffer{

public:
  
  CUDABuffer(int size){ 
    fSize = size;
    cudaMalloc((void**)&fDevPtr, fSize*sizeof(Var_t));
  }

  CUDABuffer(int size, Var_t* buffer, int len, int offset=0){ 
    fSize = size;
    cudaMalloc((void**)&fDevPtr, fSize*sizeof(Var_t));
    CopyH2D(buffer, len, offset);
  }

  CUDABuffer(int size, const Var_t* buffer, int len, int offset=0){ 
    fSize = size;
    cudaMalloc((void**)&fDevPtr, fSize*sizeof(Var_t));
    CopyH2D(buffer, len, offset);
  }
  
  ~CUDABuffer(){ 
    cudaFree(fDevPtr); 
  }

  Var_t* Get(int offset=0) const{ return fDevPtr+offset; }
  
  Var_t* GetHostBuffer(int len, int offset=0) const {
    Var_t* hostBuffer = new Var_t[len];
    cudaMemcpy(hostBuffer, fDevPtr+offset, len*sizeof(Var_t), cudaMemcpyDeviceToHost);   
    return hostBuffer;
  }

  CPUBuffer<Var_t>* GetCPUBuffer(int len, int offset=0) const {
    CPUBuffer<Var_t>* cpuBuffer = new CPUBuffer<Var_t>(len);
    cudaMemcpy(cpuBuffer->Get(), fDevPtr+offset, len*sizeof(Var_t), cudaMemcpyDeviceToHost);   
    return cpuBuffer;
  }
  
  // Need to test
   	Var_t& operator[](std::size_t idx)       { return fDevPtr[idx]; }
  const Var_t& operator[](std::size_t idx) const { return fDevPtr[idx]; }
  
  void CopyH2D(Var_t* buffer, int len, int offset=0){
    cudaMemcpy(fDevPtr+offset, buffer, len*sizeof(Var_t), cudaMemcpyHostToDevice);
  }

  void CopyH2D(const Var_t* buffer, int len, int offset=0){
    cudaMemcpy(fDevPtr+offset, buffer, len*sizeof(Var_t), cudaMemcpyHostToDevice);
  }
  
  /*
  void SetData(Var_t* buffer, int len, int offset, cudaStream_t& stream){
    cudaMemcpyAsync(fDevPtr+offset, buffer, len*sizeof(Var_t), cudaMemcpyHostToDevice,stream);
  }

  void SetData(const Var_t* buffer, int len, int offset, cudaStream_t& stream){
    cudaMemcpyAsync(fDevPtr+offset, buffer, len*sizeof(Var_t), cudaMemcpyHostToDevice,stream); 
  }
  */
  
private:
  Var_t* fDevPtr; 
  int    fSize;
};
}
