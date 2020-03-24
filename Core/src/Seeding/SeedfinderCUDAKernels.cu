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

__global__ void cuTransformCoordinates(const int* isBottom,
					const float* spM,
					const int* nSpB,
					const float* spBmat,
					float* circBmat);

__global__ void cuSearchTriplet(const float* spM,
				const int* nSpB, const float* circBmat,
				const int* nSpT, const float* circTmat,
				const float* maxScatteringAngle2, const float* sigmaScattering,
				const float* minHelixDiameter2,    const float* pT2perRadius,
				const float* impactMax );

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

void SeedfinderCUDAKernels::TransformCoordinates( dim3 grid, dim3 block,
						  const int* isBottom, 
						  const float* spM,
						  const int*   nSpB,
						  const float* spBmat,
						  float* circBmat){
  
  cuTransformCoordinates<<< grid, block >>>(isBottom, spM, nSpB, spBmat, circBmat);
  gpuErrChk( cudaGetLastError() );  
}

void SeedfinderCUDAKernels::SearchTriplet(
                                dim3 grid, dim3 block,
				const float* spM,
				const int* nSpB, const float* circBmat,
				const int* nSpT, const float* circTmat,
				const float* maxScatteringAngle2, const float* sigmaScattering,
				const float* minHelixDiameter2,    const float* pT2perRadius,
				const float* impactMax ){
  
  cuSearchTriplet<<< grid, block >>>(spM,
				     nSpB, circBmat,
				     nSpT, circTmat,
				     maxScatteringAngle2, sigmaScattering,
				     minHelixDiameter2, pT2perRadius,
				     impactMax);
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


__global__ void cuTransformCoordinates(const int* isBottom,
				       const float* spM,
				       const int* nSpB,
				       const float* spBmat,
				       float* circBmat){
  float xB = spBmat[threadIdx.x+(*nSpB)*0];
  float yB = spBmat[threadIdx.x+(*nSpB)*1];
  float zB = spBmat[threadIdx.x+(*nSpB)*2];
  float rB = spBmat[threadIdx.x+(*nSpB)*3];
  float varianceR = spBmat[threadIdx.x+(*nSpB)*4];
  float varianceZ = spBmat[threadIdx.x+(*nSpB)*5];
  //float deltaX = spM[0];
  
  
}

__global__ void cuSearchTriplet(const float* spM,
				const int* nSpB, const float* circBmat,
				const int* nSpT, const float* circTmat,
				const float* maxScatteringAngle2, const float* sigmaScattering,
				const float* minHelixDiameter2,    const float* pT2perRadius,
				const float* impactMax ){
  
}

