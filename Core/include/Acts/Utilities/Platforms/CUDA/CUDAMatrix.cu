#pragma once

#include "Acts/Utilities/Platforms/CUDA/CUDABuffer.cu"

namespace Acts{


template<typename Var_t>
class CUDAMatrix{

public:

  CUDAMatrix()=default;
  
  CUDAMatrix(size_t nRows, size_t nCols){
    fNCols = nCols;
    fNRows = nRows;
    fDevBuffer = new CUDABuffer<Var_t>(fNCols*fNRows);
  }

  Var_t* Get(size_t row, size_t col){
    fDevBuffer->data(row+col*fNRows);
  }
  
  void SetColumn(size_t col, Var_t* buffer){
    fDevBuffer->SetData(buffer, fNRows ,col*fNRows);
  }
  
private:
  CUDABuffer<Var_t>* fDevBuffer; 
  size_t fNCols;
  size_t fNRows;
};

}

