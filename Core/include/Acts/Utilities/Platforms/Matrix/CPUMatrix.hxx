#ifndef MATRIX_CPUMATRIX
#define MATRIX_CPUMATRIX

#include <iostream>
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"

namespace Acts{

template<typename Var_t, int row, int col>
class CPUMatrix{

public:
  
  CPUMatrix(){ 
    //fNRows = row;
    //fNCols = col;
    // pinned memory definition
    cudaMallocHost((void**)&fHostPtr, row*col*sizeof(Var_t));
  }

  CPUMatrix(Var_t* buffer){   
    //fNRows = row;
    //fNCols = col;
    cudaMallocHost((void**)&fHostPtr, row*col*sizeof(Var_t));
    fHostPtr = buffer;
  }

  ~CPUMatrix(){ cudaFreeHost(fHostPtr); }

  Var_t*  data() const{ return fHostPtr; }

private:
  Var_t* fHostPtr; 
  //int fNRows;
  //int fNCols;   
};

}

#endif
