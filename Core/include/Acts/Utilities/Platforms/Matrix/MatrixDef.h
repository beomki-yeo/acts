#ifndef MATRIX_MATRIXDEF
#define MATRIX_MATRIXDEF

#include "Acts/Utilities/Platforms/Matrix/CUDAMatrix.cu"
#include "Acts/Utilities/Platforms/Matrix/CPUMatrix.hxx"

// Type definition for CUDAMatrix and CPUMatrix

namespace Acts{

class CUDA{

public:

  template<typename Var_t, int row, int col>
  using MatrixX  = CUDAMatrix<Var_t, row, col>;

  template<typename Var_t, int row>
  using VectorX  = CUDAMatrix<Var_t, row, 1>;

  template<typename Var_t>
  using Vector2  = CUDAMatrix<Var_t, 2, 1>;  

  template<typename Var_t>
  using Vector3  = CUDAMatrix<Var_t, 3, 1>; 
};
  
class CPU{

public:

  template<typename Var_t, int row, int col>
  using MatrixX  = CPUMatrix<Var_t, row, col>;

  template<typename Var_t, int row>
  using VectorX  = CPUMatrix<Var_t, row, 1>;

  template<typename Var_t>
  using Vector2  = CPUMatrix<Var_t, 2, 1>;  

  template<typename Var_t>
  using Vector3  = CPUMatrix<Var_t, 3, 1>; 
};


}

#endif
