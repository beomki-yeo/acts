#include "Acts/Seeding/SeedfinderKernels.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void cuSearchDoublet(const float* rBvec, const float* zBvec, const float* rM, const float* zM, const bool* isBottom,  const float* deltaRMin,  const float* deltaRMax, const float* cotThetaMax, const float* collisionRegionMin, const float* collisionRegionMax, int* isCompatible);

void SeedfinderKernels::SearchDoublet( dim3 grid, dim3 block, const float* rBvec, const float* zBvec, const float* rM, const float* zM, const bool* isBottom,  const float* deltaRMin,   const float* deltaRMax, const float* cotThetaMax, const float* collisionRegionMin, const float* collisionRegionMax,  int* isCompatible  ){
  cuSearchDoublet<<<grid, block>>>( rBvec, zBvec, rM, zM, isBottom, deltaRMin, deltaRMax, cotThetaMax, collisionRegionMin, collisionRegionMax, isCompatible );
}

__global__ void cuSearchDoublet(const float* rBvec, const float* zBvec, 
				const float* rM, const float* zM, const bool* isBottom,  
				const float* deltaRMin,  const float* deltaRMax, 
				const float* cotThetaMax, const float* collisionRegionMin,  const float* collisionRegionMax,
				 int* isCompatible ){
  int globalId = blockIdx.x+blockDim.x * blockIdx.x;
  float rB = rBvec[globalId];
  float zB = zBvec[globalId];
  float deltaR = *rM - rB;

  // Doublet search for bottom hits
  if (*isBottom == true){
    if (deltaR > *deltaRMax){
      isCompatible[globalId] = false;
    }

    if (deltaR < *deltaRMin){
      isCompatible[globalId] = false;
    }

    float cotTheta = (*zM - zB)/deltaR;
    if (fabs(cotTheta) > *cotThetaMax){
      isCompatible[globalId] = false;
    }

    float zOrigin = *zM - (*rM) * cotTheta;
    if (zOrigin < *collisionRegionMin || zOrigin > *collisionRegionMax){
      isCompatible[globalId] = false;
    }
  }

  // Doublet search for top hits
  else if (*isBottom == false){

    if (deltaR < *deltaRMin){
      isCompatible[globalId] = false;
    }

    if (deltaR > *deltaRMax){
      isCompatible[globalId] = false;
    }

    if (isCompatible[globalId] == true){
      float cotTheta = (zB -*zM)/deltaR;
      if (fabs(cotTheta) > *cotThetaMax){
	isCompatible[globalId] = false;
      }
      
      float zOrigin = *zM - (*rM) * cotTheta;
      if (zOrigin < *collisionRegionMin || zOrigin > *collisionRegionMax){
	isCompatible[globalId] = false;
      }
    }
  }
}

