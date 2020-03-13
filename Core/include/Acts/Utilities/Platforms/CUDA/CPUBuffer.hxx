#ifndef CPUBUFFER
#define CPUBUFFER

#include <iostream>
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"

namespace Acts{

template<typename Var_t>
class CPUBuffer{

public:
  
  CPUBuffer(int size){ 
    fSize = size;
    // pinned memory definition
    cudaMallocHost((void**)&fHostPtr, fSize*sizeof(Var_t));
  }

  CPUMatrix(Var_t* buffer, int size){   
    fSize = size;
    cudaMallocHost((void**)&fHostPtr, fSize*sizeof(Var_t));
    fHostPtr = buffer;
  }

  ~CPUBuffer(){ cudaFreeHost(fHostPtr); }

  Var_t*  data() const{ return fHostPtr; }

private:
  Var_t* fHostPtr; 
  int    fSize;   
};

}

#endif
