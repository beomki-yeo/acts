#include <iostream>
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"

namespace Acts{

template<typename varT, int row, int col>
class CUDAMatrix{

public:
  
  CUDAMatrix(){ 
    //fNRows = row;
    //fNCols = col;
    cudaMalloc((void**)&fDevPtr, row*col*sizeof(varT));
  }

  CUDAMatrix(varT* buffer){   
    //fNRows = row;
    //fNCols = col;
    cudaMalloc((void**)&fDevPtr, row*col*sizeof(varT));
    cudaMemcpy(fDevPtr, buffer, row*col*sizeof(varT), cudaMemcpyHostToDevice);     
  }

  ~CUDAMatrix(){ cudaFree(fDevPtr); }

  varT* dataHost() const {
    varT* hostPtr = new varT[row*col];
    cudaMemcpy(hostPtr, fDevPtr, row*col*sizeof(varT), cudaMemcpyDeviceToHost);   
    return hostPtr;
  }

  varT*  data() const{ return fDevPtr; }

private:

  varT* fDevPtr; 
  //int fNRows;
  //int fNCols;
  
};

}

