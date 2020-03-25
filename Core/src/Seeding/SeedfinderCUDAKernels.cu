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
				const float* circBmat,
				const float* circTmat,
				const float* maxScatteringAngle2, const float* sigmaScattering,
				const float* minHelixDiameter2, const float* pT2perRadius,
				const float* impactMax );

void SeedfinderCUDAKernels::searchDoublet(
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

void SeedfinderCUDAKernels::transformCoordinates( dim3 grid, dim3 block,
						  const int* isBottom, 
						  const float* spM,
						  const int*   nSpB,
						  const float* spBmat,
						  float* circBmat){
  
  cuTransformCoordinates<<< grid, block >>>(isBottom, spM, nSpB, spBmat, circBmat);
  gpuErrChk( cudaGetLastError() );  
}

void SeedfinderCUDAKernels::searchTriplet(
                                dim3 grid, dim3 block,
				const float* spM,
				const float* circBmat,
				const float* circTmat,
				const float* maxScatteringAngle2, const float* sigmaScattering,
				const float* minHelixDiameter2, const float* pT2perRadius,
				const float* impactMax ){
  
  cuSearchTriplet<<< grid, block >>>(spM,
				     circBmat,circTmat,
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

  int globalId = threadIdx.x+blockDim.x*blockIdx.x;
  
  float xM = spM[0];
  float yM = spM[1];
  float zM = spM[2];
  float rM = spM[3];
  float varianceRM = spM[4];
  float varianceZM = spM[5];
  float cosPhiM = xM / rM;
  float sinPhiM = yM / rM;
    
  float xB = spBmat[globalId+(*nSpB)*0];
  float yB = spBmat[globalId+(*nSpB)*1];
  float zB = spBmat[globalId+(*nSpB)*2];
  float rB = spBmat[globalId+(*nSpB)*3];
  float varianceRB = spBmat[globalId+(*nSpB)*4];
  float varianceZB = spBmat[globalId+(*nSpB)*5];
  
  float deltaX = xB - xM;
  float deltaY = yB - yM;
  float deltaZ = zB - zM;
  
  // calculate projection fraction of spM->sp vector pointing in same
  // direction as
  // vector origin->spM (x) and projection fraction of spM->sp vector pointing
  // orthogonal to origin->spM (y)
  float x = deltaX * cosPhiM + deltaY * sinPhiM;
  float y = deltaY * cosPhiM - deltaX * sinPhiM;
  // 1/(length of M -> SP)
  float iDeltaR2 = 1. / (deltaX * deltaX + deltaY * deltaY);
  float iDeltaR = std::sqrt(iDeltaR2);

  int bottomFactor = 1 * (int(!(*isBottom))) - 1 * (int(*isBottom));
  // cot_theta = (deltaZ/deltaR)
  float cot_theta = deltaZ * iDeltaR * bottomFactor;
  // VERY frequent (SP^3) access

  // location on z-axis of this SP-duplet
  float Zo = zM - rM * cot_theta;
  
  // transformation of circle equation (x,y) into linear equation (u,v)
  // x^2 + y^2 - 2x_0*x - 2y_0*y = 0
  // is transformed into
  // 1 - 2x_0*u - 2y_0*v = 0
  // using the following m_U and m_V
  // (u = A + B*v); A and B are created later on  
  float U  = x*iDeltaR2;
  float V  = y*iDeltaR2;
  // error term for sp-pair without correlation of middle space point  
  float Er = ((varianceZM + varianceZB) +
	      (cot_theta * cot_theta) * (varianceRM + varianceRB)) * iDeltaR2;  
  
  circBmat[globalId+(*nSpB)*0] = Zo;
  circBmat[globalId+(*nSpB)*1] = cot_theta;
  circBmat[globalId+(*nSpB)*2] = iDeltaR;
  circBmat[globalId+(*nSpB)*3] = Er;
  circBmat[globalId+(*nSpB)*4] = U;
  circBmat[globalId+(*nSpB)*5] = V; 
  
}

__global__ void cuSearchTriplet(const float* spM,
				const float* circBmat,
				const float* circTmat,
				const float* maxScatteringAngle2, const float* sigmaScattering,
				const float* minHelixDiameter2,    const float* pT2perRadius,
				const float* impactMax ){
  
}
