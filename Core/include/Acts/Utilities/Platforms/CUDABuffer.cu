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

  SetData(varT* buffer, int len, int offset=0, cudaStream_t stream=NULL){
    if (stream != NULL){
      cudaStreamCreate(&stream);
      cudaMemcpyAsync(fDevPtr+offset, buffer, len*sizeof(varT), cudaMemcpyHostToDevice,stream); 
    }
    else if (stream == NULL) {
	cudaMemcpy(fDevPtr+offset, buffer, len*sizeof(varT), cudaMemcpyHostToDevice);
    }
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

/*
template<typename varT, int row, int col>
class CUDAMatrix{

public:
  
  CUDAMatrix(){ 
    fNRows = row;
    fNCols = col;
    cudaMalloc((void**)&fDevPtr, row*col*sizeof(varT));
    cudaStreamCreate(&fStream);
  }

  CUDAMatrix(varT* buffer){   
    fNRows = row;
    fNCols = col;
    cudaMalloc((void**)&fDevPtr, row*col*sizeof(varT));
    cudaStreamCreate(&fStream);
    cudaMemcpyAsync(fDevPtr, buffer, row*col*sizeof(varT), cudaMemcpyHostToDevice,fStream); 
    
    std::cout << "Create Matrix" << std::endl;
  }

  ~CUDAMatrix(){ 
    cudaFree(fDevPtr); 
    cudaStreamDestroy(fStream);
    std::cout << "Destroy Matrix" << std::endl;
  }

  varT* dataHost() const {
    varT* hostPtr = new varT[row*col];
    cudaMemcpy(hostPtr, fDevPtr, row*col*sizeof(varT), cudaMemcpyDeviceToHost);   
    return hostPtr;
  }

  varT*  data() const{ return fDevPtr; }

private:

  varT* fDevPtr; 
  int fNRows;
  int fNCols;
  cudaStream_t fStream;
*/

