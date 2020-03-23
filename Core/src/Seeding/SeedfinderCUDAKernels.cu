#include "Acts/Seeding/SeedfinderCUDAKernels.cuh"
#include "Acts/Utilities/Platforms/CUDA/CuUtils.cu"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void cuSearchDoublet(const int* isBottom,  
				const float* rBvec, const float* zBvec, 
				const float* rMvec, const float* zMvec,
				const float* deltaRMin,const float* deltaRMax,const float* cotThetaMax, 
				const float* collisionRegionMin, const float* collisionRegionMax, 
				int* isCompatible);


void SeedfinderCUDAKernels::SearchDoublet(
			        dim3 grid, dim3 block,
				const int* isBottom,
				const float* rBvec, const float* zBvec, 
				const float* rMvec, const float* zMvec,
				const float* deltaRMin,const float* deltaRMax,const float* cotThetaMax, 
				const float* collisionRegionMin, const float* collisionRegionMax,  
				int* isCompatible  ){
  
  cuSearchDoublet<<< grid, block >>>( isBottom,
				      rBvec, zBvec, rMvec, zMvec,
				      deltaRMin, deltaRMax, cotThetaMax, 
				      collisionRegionMin, collisionRegionMax, 
				      isCompatible );
  gpuErrChk( cudaGetLastError() );
}


__global__ void cuSearchDoublet(const int* isBottom,
				const float* rBvec, const float* zBvec, 
				const float* rMvec, const float* zMvec,   
				const float* deltaRMin,const float* deltaRMax,const float* cotThetaMax, 
				const float* collisionRegionMin, const float* collisionRegionMax,
				int* isCompatible ){

  int globalId = threadIdx.x+blockDim.x*blockIdx.x;
  
  float rB = rBvec[threadIdx.x];
  float zB = zBvec[threadIdx.x];
  float rM = rMvec[blockIdx.x];
  float zM = zMvec[blockIdx.x];  

  isCompatible[globalId] = true;
  
  // Doublet search for bottom hits
  if (*isBottom == true){

    float deltaR = rM - rB;

    if (deltaR > *deltaRMax){
      isCompatible[globalId] = false;
    }

    if (deltaR < *deltaRMin){
      isCompatible[globalId] = false;
    }

    float cotTheta = (zM - zB)/deltaR;
    if (fabs(cotTheta) > *cotThetaMax){
      isCompatible[globalId] = false;
    }

    float zOrigin = zM - rM*cotTheta;
    if (zOrigin < *collisionRegionMin || zOrigin > *collisionRegionMax){
      isCompatible[globalId] = false;
    }
  }

  // Doublet search for top hits
  else if (*isBottom == false){

    float deltaR = rB - rM;

    if (deltaR < *deltaRMin){
      isCompatible[globalId] = false;
    }

    if (deltaR > *deltaRMax){
      isCompatible[globalId] = false;
    }

    if (isCompatible[globalId] == true){
      float cotTheta = (zB - zM)/deltaR;
      if (fabs(cotTheta) > *cotThetaMax){
	isCompatible[globalId] = false;
      }
      
      float zOrigin = zM - rM*cotTheta;
      if (zOrigin < *collisionRegionMin || zOrigin > *collisionRegionMax){
	isCompatible[globalId] = false;
      }
    }
  }
}
