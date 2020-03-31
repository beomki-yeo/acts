#ifndef CPUMATRIX
#define CPUMATRIX

#include "Acts/Utilities/Platforms/CUDA/CPUArray.hxx"
#include "Acts/Utilities/Platforms/CUDA/CUDAMatrix.cu"

// column-major style Matrix Definition

namespace Acts{

template<typename Var_t>
class CUDAMatrix;
  
template<typename Var_t>
class CPUMatrix{
  
public:

  CPUMatrix() = default;
  CPUMatrix(size_t nRows, size_t nCols){ 
    fNCols = nCols;
    fNRows = nRows;
    fHostArray = new CPUArray<Var_t>(fNCols*fNRows);
  }

  CPUMatrix(size_t nRows, size_t nCols, CUDAMatrix<Var_t>* cuMat){ 
    fNCols = nCols;
    fNRows = nRows;
    fHostArray = cuMat->GetCPUArray(fNCols*fNRows,0,0);
  }
  
  ~CPUMatrix(){
    delete fHostArray;
  }

  size_t GetNCols(){ return fNCols; }
  size_t GetNRows(){ return fNRows; }

  Var_t* GetEl(size_t row=0, size_t col=0){
    return fHostArray->Get(row+col*fNRows);
  }
  
  void SetEl(size_t row, size_t col, Var_t val){
    (*fHostArray)[row+col*fNRows]=val;
  }
  
  Var_t* GetColumn(size_t col){
    // Need to retrive the pointer directly
    return fHostArray->Get()+col*fNRows;
  }
  Var_t* GetRow(size_t row){    
    Var_t* ret = new Var_t[fNCols];
    for(int i_c=0; i_c<fNCols; i_c++) ret[i_c] = (*fHostArray)[row+fNRows*i_c];
    return ret;    
  }

  void SetRow(size_t row, Var_t* input){
    for(size_t i_c=0; i_c<fNCols; i_c++){
      (*fHostArray)[row+fNRows*i_c]=input[i_c];
    }
  }
  
private:
  CPUArray<Var_t>* fHostArray; 
  size_t fNCols;
  size_t fNRows;

};
  
}

#endif
