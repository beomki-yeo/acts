#pragma once
#ifndef PLATFORMDEF
#define PLATFORMDEF

#include "Acts/Utilities/Platforms/CUDA/CUDABuffer.cu"
#include "Acts/Utilities/Platforms/CUDA/CUDAMatrix.cu"
#include "Acts/Utilities/Platforms/CPU/CPUBuffer.hxx"
#include "Acts/Utilities/Platforms/CPU/CPUMatrix.hxx"

// Type definition for CUDAMatrix and CPUMatrix

namespace Acts{

class CUDA{

public:
  
  template<typename Var_t>
  using Buffer = CUDABuffer<Var_t>;

  template<typename Var_t>
  using Matrix = CUDAMatrix<Var_t>;
  
};
  
class CPU{

public:

  template<typename Var_t>
  using Buffer = CPUBuffer<Var_t>;

  template<typename Var_t>
  using Matrix = CPUMatrix<Var_t>;

};
}

#endif
