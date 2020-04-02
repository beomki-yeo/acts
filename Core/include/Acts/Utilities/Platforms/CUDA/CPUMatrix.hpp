#ifndef CPUMATRIX
#define CPUMATRIX

#include "Acts/Utilities/Platforms/CUDA/CPUArray.hpp"
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
    fNRows = nRows;
    fNCols = nCols;
    //fHostArray = new CPUArray<Var_t>(fNCols*fNRows);
    cudaMallocHost(&fHostPtr, fNRows*fNCols*sizeof(Var_t));
  }

  CPUMatrix(size_t nRows, size_t nCols, CUDAMatrix<Var_t>* cuMat){
    fNRows = nRows;
    fNCols = nCols;
    //fHostArray = cuMat->GetCPUArray(fNCols*fNRows,0,0);
    fHostPtr = (cuMat->GetCPUArray(fNRows*fNCols,0,0))->Get();
  }
  
  ~CPUMatrix(){
    //delete fHostArray;
    cudaFreeHost(fHostPtr);
  }

  size_t GetNCols(){ return fNCols; }
  size_t GetNRows(){ return fNRows; }

  Var_t* GetEl(size_t row=0, size_t col=0){
    //return fHostArray->Get(row+col*fNRows);
    int offset=row+col*fNRows;
    return fHostPtr+offset;
  }
  
  void SetEl(size_t row, size_t col, Var_t val){
    //(*fHostArray)[row+col*fNRows]=val;
    int offset=row+col*fNRows;
    fHostPtr[offset] = val;
  }
  
  Var_t* GetColumn(size_t col){
    // Need to retrive the pointer directly
    //return fHostArray->Get()+col*fNRows;
    return fHostPtr+col*fNRows;
  }
  Var_t* GetRow(size_t row){    
    Var_t* ret = new Var_t[fNCols];
    //for(int i_c=0; i_c<fNCols; i_c++) ret[i_c] = (*fHostArray)[row+fNRows*i_c];
    for(int i_c=0; i_c<fNCols; i_c++) ret[i_c] = fHostPtr[row+fNRows*i_c];
    return ret;    
  }

  void SetRow(size_t row, Var_t* input){
    for(size_t i_c=0; i_c<fNCols; i_c++){
      //(*fHostArray)[row+fNRows*i_c]=input[i_c];
      fHostPtr[row+fNRows*i_c]=input[i_c];
    }
  }

  void SetColumn(size_t col, Var_t* input){
    /*
    for(size_t i_c=0; i_c<fNCols; i_c++){
      //(*fHostArray)[row+fNRows*i_c]=input[i_c];
      fHostPtr[row+fNRows*i_c]=input[i_c];
    }
    */
    //std::copy(fHostPtr+col*fNRows,fHostPtr+(col+1)*fNRows,input);
    fHostPtr[col*fNRows] = input[0];
  }

  
private:
  //CPUArray<Var_t>* fHostArray;
  Var_t* fHostPtr;
  size_t fNCols;
  size_t fNRows;

};
  
}

#endif
