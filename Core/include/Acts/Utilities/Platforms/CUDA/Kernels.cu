#include <cuda.h>
#include <cuda_runtime.h>

template<typename Var_t>
__global__ void ArraySum(Var_t* bufferA, Var_t* bufferB, Var_t* bufferC){
  int id = threadIdx.x;
  bufferC[id] = bufferA[id] + bufferB[id];
}

