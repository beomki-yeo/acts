#pragma once
#ifndef PLATFORMDEF
#define PLATFORMDEF

#include "Acts/Utilities/Platforms/CUDA/CUDABuffer.cu"
#include "Acts/Utilities/Platforms/CUDA/CPUBuffer.hxx"

// Type definition for CUDAMatrix and CPUMatrix

namespace Acts{

class CUDA{

public:
  
  template<typename Var_t>
  using Buffer = CUDABuffer<Var_t>;

};
  
class CPU{

public:

  template<typename Var_t>
  using Buffer = CPUBuffer<Var_t>;

};
}

#endif
