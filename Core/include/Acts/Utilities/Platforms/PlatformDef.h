#pragma once
#ifndef PLATFORMDEF
#define PLATFORMDEF

#include "Acts/Utilities/Platforms/CUDA/CUDAArray.cu"
#include "Acts/Utilities/Platforms/CUDA/CPUArray.hxx"
#include "Acts/Utilities/Platforms/CUDA/CUDAMatrix.cu"
#include "Acts/Utilities/Platforms/CUDA/CPUMatrix.hxx"

// Type definition for each platform

namespace Acts{

class CPU;
class CUDA;

}

#endif
