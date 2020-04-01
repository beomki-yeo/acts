#pragma once

#include "Acts/Utilities/Platforms/CUDA/CUDAArray.cu"
#include "Acts/Utilities/Platforms/CUDA/CPUMatrix.hpp"

namespace Acts{

template<typename Var_t>
class CUDAMatrix{

public:

  CUDAMatrix()=default;
  CUDAMatrix(size_t nRows, size_t nCols){
    fNCols = nCols;
    fNRows = nRows;
    fDevArray = new CUDAArray<Var_t>(fNCols*fNRows);
  }

  CUDAMatrix(size_t nRows, size_t nCols, CPUMatrix<Var_t>* mat){
    fNCols = nCols;
    fNRows = nRows;
    fDevArray = new CUDAArray<Var_t>(fNCols*fNRows);
    CopyH2D(mat->GetEl(0,0),fNRows*fNCols,0);
  }
  
  ~CUDAMatrix(){
    delete fDevArray;
  }

  size_t GetNCols(){ return fNCols; }
  size_t GetNRows(){ return fNRows; }

  Var_t* GetEl(size_t row, size_t col){
    fDevArray->Get(row+col*fNRows);
  }
    
  Var_t* GetHostArray(size_t len, size_t row, size_t col){
    return fDevArray->GetHostArray(len, row+col*fNRows);
  }

  CPUArray<Var_t>* GetCPUArray(size_t len, size_t row, size_t col){
    return fDevArray->GetCPUArray(len, row+col*fNRows);
  }
  
  void CopyH2D(Var_t* array, size_t len, size_t offset=0){
    fDevArray->CopyH2D(array,len,offset);
  }

  void CopyH2D(const Var_t* array, size_t len, size_t offset=0){
    fDevArray->CopyH2D(array,len,offset);
  }
    
  void SetColumn(size_t col, Var_t* array){
    fDevArray->CopyH2D(array, fNRows ,col*fNRows);
  }
  
private:
  CUDAArray<Var_t>* fDevArray; 
  size_t fNCols;
  size_t fNRows;
};

}

