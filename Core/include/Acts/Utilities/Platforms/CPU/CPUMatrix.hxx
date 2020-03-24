#ifndef CPUMATRIX
#define CPUMATRIX

#include "Acts/Utilities/Platforms/CPU/CPUBuffer.hxx"

namespace Acts{

template<typename Var_t>
class CPUMatrix{
  
public:

  CPUMatrix(size_t nRows, size_t nCols){ 
    fNCols = nCols;
    fNRows = nRows;
    fHostBuffer = new CPUBuffer<Var_t>(fNCols*fNRows);
  }
  ~CPUMatrix(){
    delete fHostBuffer;
  }
  
  Var_t* Get(int row=0, int col=0){
    return fHostBuffer->data(col+row*fNCols);
  }
  
  Var_t* GetColumn(int index){
    Var_t* ret = new int[fNRows];
    for (int i_r=0; i_r<fNRows; i_r++) ret[i_r] = (*fHostBuffer)[index+fNCols*i_r];
    return ret;
  }
  Var_t* GetRow(int index){
    Var_t* ret = new int[fNCols];
    for (int i_c=0; i_c<fNCols; i_c++) ret[i_c] = (*fHostBuffer)[index*fNCols+i_c];
    return ret;
  }
  
private:
  CPUBuffer<Var_t>* fHostBuffer; 
  size_t fNCols;
  size_t fNRows;

};
  
}

#endif
