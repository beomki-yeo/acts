#pragma once

#include "Acts/Utilities/Platforms/CUDA/CUDABuffer.cu"
#include "Acts/Utilities/Platforms/CPU/CPUMatrix.hxx"

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

  CUDAMatrix(size_t nRows, size_t nCols, CPUMatrix<Var_t>* mat){
    fNCols = nCols;
    fNRows = nRows;
    fDevBuffer = new CUDABuffer<Var_t>(fNCols*fNRows);
    CopyH2D(mat->GetEl(0,0),fNRows*fNCols,0);
  }

  
  ~CUDAMatrix(){
    delete fDevBuffer;
  }

  size_t GetNCols(){ return fNCols; }
  size_t GetNRows(){ return fNRows; }

  Var_t* GetEl(size_t row, size_t col){
    fDevBuffer->Get(row+col*fNRows);
  }
    
  Var_t* GetHostBuffer(size_t len, size_t row, size_t col){
    fDevBuffer->GetHostBuffer(len, row+col*fNRows);
  }

  void CopyH2D(Var_t* buffer, int len, int offset=0){
    fDevBuffer->CopyH2D(buffer,len,offset);
  }

  void CopyH2D(const Var_t* buffer, int len, int offset=0){
    fDevBuffer->CopyH2D(buffer,len,offset);
  }
    
  void SetColumn(size_t col, Var_t* buffer){
    fDevBuffer->CopyH2D(buffer, fNRows ,col*fNRows);
  }
  
private:
  CUDABuffer<Var_t>* fDevBuffer; 
  size_t fNCols;
  size_t fNRows;
};

}

