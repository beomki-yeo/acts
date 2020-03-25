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

  CPUBuffer()=default;
  
  CPUBuffer(int size){ 
    fSize = size;
    // pinned memory definition
    cudaMallocHost(&fHostPtr, fSize*sizeof(Var_t));
  }

  CPUBuffer(int size, Var_t* buffer){   
    fSize = size;
    cudaMallocHost(&fHostPtr, fSize*sizeof(Var_t));
    fHostPtr = buffer;
  }

  ~CPUBuffer(){ cudaFreeHost(fHostPtr); }

  Var_t* Get(int offset=0){ return fHostPtr+offset; }

  // Need to test
   	Var_t& operator[](std::size_t idx)       { return fHostPtr[idx]; }
  const Var_t& operator[](std::size_t idx) const { return fHostPtr[idx]; }

  
private:
  Var_t* fHostPtr; 
  int    fSize;   
};

}

#endif
