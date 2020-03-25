#ifndef CPUMATRIX
#define CPUMATRIX

#include "Acts/Utilities/Platforms/CPU/CPUBuffer.hxx"

// column-major style Matrix Definition

namespace Acts{

template<typename Var_t>
class CPUMatrix{
  
public:

  CPUMatrix() = default;
  CPUMatrix(size_t nRows, size_t nCols){ 
    fNCols = nCols;
    fNRows = nRows;
    fHostBuffer = new CPUBuffer<Var_t>(fNCols*fNRows);
  }
  ~CPUMatrix(){
    delete fHostBuffer;
  }

  size_t GetNCols(){ return fNCols; }
  size_t GetNRows(){ return fNRows; }

  Var_t* GetEl(int row=0, int col=0){
    return fHostBuffer->Get(row+col*fNRows);
  }
  
  void SetEl(int row, int col, Var_t val){
    (*fHostBuffer)[row+col*fNRows]=val;
  }
  
  Var_t* GetColumn(int col){
    // Need to retrive the pointer directly
    return fHostBuffer->Get()+col*fNRows;

    //Var_t* ret = new Var_t[fNRows];
    //for (int i_r=0; i_r<fNRows; i_r++) ret[i_r] = (*fHostBuffer)[col*fNRows+i_r];
    //return ret;
  }
  Var_t* GetRow(int row){    
    Var_t* ret = new Var_t[fNCols];
    for(int i_c=0; i_c<fNCols; i_c++) ret[i_c] = (*fHostBuffer)[row+fNRows*i_c];
    return ret;    
  }

  void SetRow(int row, Var_t* input){
    for(int i_c=0; i_c<fNCols; i_c++){
      (*fHostBuffer)[row+fNRows*i_c]=input[i_c];
    }
  }
  
private:
  CPUBuffer<Var_t>* fHostBuffer; 
  size_t fNCols;
  size_t fNRows;

};
  
}

#endif
