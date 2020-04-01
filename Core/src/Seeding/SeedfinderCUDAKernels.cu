#include "Acts/Seeding/SeedfinderCUDAKernels.cuh"
#include "Acts/Utilities/Platforms/CUDA/CuUtils.cu"
#include "Acts/Seeding/IExperimentCuts.hpp"
#include "Acts/Seeding/SeedFilter.hpp"
#include "Acts/Seeding/SeedfinderConfig.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void cuSearchDoublet(const unsigned char* isBottom,
				const float* rMvec, const float* zMvec,
				const int* nSpB, const float* rBvec, const float* zBvec, 
				//const Acts::CuSeedfinderConfig* config,
				const float* deltaRMin,const float*deltaRMax,const float*cotThetaMax, 
				const float* collisionRegionMin, const float* collisionRegionMax,
				unsigned char* isCompatible				
				);

__global__ void cuTransformCoordinates(const unsigned char* isBottom,
				       const float* spM,
				       const int* nSpB,
				       const float* spBmat,
				       float* circBmat);

__global__ void cuSearchTriplet(const int*   offset,
				const float* spM,
				const int*   nSpB, const float* spBmat,
				const int*   nSpT, const float* spTmat,
				const float* circBmat,
				const float* circTmat,
				//const Acts::CuSeedfinderConfig* config
				const float* maxScatteringAngle2, const float* sigmaScattering,
				const float* minHelixDiameter2, const float* pT2perRadius,
				const float* impactMax,
				const int*   nTopPassLimit,
				int* nTopPass,
				int* tIndex,
				float* curvatures,
				float* impactparameters				
				);

namespace Acts{

  
  void SeedfinderCUDAKernels::searchDoublet(
			        dim3 grid, dim3 block,
				const unsigned char* isBottom,
				const float* rMvec, const float* zMvec,
				const int* nSpB, const float* rBvec, const float* zBvec, 
				const float* deltaRMin,const float*deltaRMax,const float*cotThetaMax, 
				const float* collisionRegionMin, const float* collisionRegionMax,
				//const Acts::CuSeedfinderConfig* config,
				unsigned char* isCompatible  ){
    
  cuSearchDoublet<<< grid, block >>>(//offset,
				     isBottom,
				     rMvec, zMvec,
				     nSpB, rBvec, zBvec, 
				     deltaRMin, deltaRMax, cotThetaMax, 
				     collisionRegionMin, collisionRegionMax,
				     //config,
				     isCompatible );
  gpuErrChk( cudaGetLastError() );
  }

  void SeedfinderCUDAKernels::transformCoordinates(
				   dim3 grid, dim3 block,
				   const unsigned char* isBottom, 
				   const float* spM,
				   const int*   nSpB,
				   const float* spBmat,
				   float* circBmat){
    
    cuTransformCoordinates<<< grid, block >>>(isBottom, spM, nSpB, spBmat, circBmat);
    gpuErrChk( cudaGetLastError() );  
  }

  void SeedfinderCUDAKernels::searchTriplet(
				dim3 grid, dim3 block,
				const int*   offset,
				const float* spM,
				const int*   nSpB, const float* spBmat,
				const int*   nSpT, const float* spTmat,
				const float* circBmat,
				const float* circTmat,
				//const Acts::CuSeedfinderConfig* config
				// finder config
				const float* maxScatteringAngle2, const float* sigmaScattering,
				const float* minHelixDiameter2,   const float* pT2perRadius,
				const float* impactMax,           const int*   nTopPassLimit,	  
				int*   nTopPass,   int*   tIndex,
				float* curvatures, float* impactparameters
				// filter config
				//const float* deltaInvHelixDiameter,
				//const float* impactWeightFactor,
				//const float* deltaRMin,
				//const float* compatSeedWeight,
				//const size_t* compatSeedLimit,
				){
    
  cuSearchTriplet<<< grid, block, sizeof(unsigned char)*block.x >>>(
			       offset,
			       spM,
			       nSpB, spBmat,
			       nSpT, spTmat,				     
			       circBmat,circTmat,
			       //config				     
			       maxScatteringAngle2, sigmaScattering,
			       minHelixDiameter2, pT2perRadius,
			       impactMax, nTopPassLimit,
			       //output
			       nTopPass, tIndex,
			       curvatures, impactparameters
			       );
  gpuErrChk( cudaGetLastError() );
  }
  
}

__global__ void cuSearchDoublet(const unsigned char* isBottom,
				const float* rMvec, const float* zMvec,
				const int* nSpB, const float* rBvec, const float* zBvec, 	   
				//const Acts::CuSeedfinderConfig* config,
				const float* deltaRMin,const float*deltaRMax,const float*cotThetaMax, 
				const float* collisionRegionMin, const float* collisionRegionMax,
				unsigned char* isCompatible 				
				){
  
  int globalId = threadIdx.x+(*nSpB)*blockIdx.x;
  
  float rB = rBvec[threadIdx.x];
  float zB = zBvec[threadIdx.x];
  float rM = rMvec[blockIdx.x];
  float zM = zMvec[blockIdx.x];  
  
  // Doublet search for bottom hits
  isCompatible[globalId] = true;
  
  if (*isBottom == true){    
    float deltaR = rM - rB;
    //printf("%d %d \n", globalId, *nSpB);
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
    //printf("%d %d \n", globalId, *nSpB);
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


__global__ void cuTransformCoordinates(const unsigned char* isBottom,
				       const float* spM,
				       const int* nSpB,
				       const float* spBmat,
				       float* circBmat){

  int globalId = threadIdx.x+blockDim.x*blockIdx.x;
  if (globalId>=*nSpB) return;
  
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
  //float rB = spBmat[globalId+(*nSpB)*3];
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

__global__ void cuSearchTriplet(const int*   offset,
				const float* spM,
				const int*   nSpB, const float* spBmat,
				const int*   nSpT, const float* spTmat,
				const float* circBmat,
				const float* circTmat,
				//const Acts::CuSeedfinderConfig* config
				const float* maxScatteringAngle2, const float* sigmaScattering,
				const float* minHelixDiameter2,    const float* pT2perRadius,
				const float* impactMax,
				const int*   nTopPassLimit,
				int* nTopPass,
				int* tIndex,
				float* curvatures,
				float* impactparameters
				){
  __shared__ extern unsigned char isPassed[];
  
  int threadId = threadIdx.x;
  int blockId  = blockIdx.x;

  //rT[threadIdx.x] = spTmat[threadId+(*nSpT)*3];
  
  float rM = spM[3];
  //float zM = spM[2];
  float varianceRM = spM[4];
  float varianceZM = spM[5];

  //float spB[6];
  //spB[0] = spBmat[blockId+(*nSpB)*0];
  //spB[1] = spBmat[blockId+(*nSpB)*1];
  //spB[2] = spBmat[blockId+(*nSpB)*2];
  //spB[3] = spBmat[blockId+(*nSpB)*3];
  //spB[4] = spBmat[blockId+(*nSpB)*4];
  //spB[5] = spBmat[blockId+(*nSpB)*5];

  // Zob values from CPU and CUDA are slightly different
  //float Zob        = circBmat[blockId+(*nSpB)*0];
  float cotThetaB  = circBmat[blockId+(*nSpB)*1];
  float iDeltaRB   = circBmat[blockId+(*nSpB)*2];
  float ErB        = circBmat[blockId+(*nSpB)*3];
  float Ub         = circBmat[blockId+(*nSpB)*4];
  float Vb         = circBmat[blockId+(*nSpB)*5];
  float iSinTheta2 = (1. + cotThetaB * cotThetaB);
  float scatteringInRegion2 = (*maxScatteringAngle2) * iSinTheta2;
  scatteringInRegion2 *= (*sigmaScattering) * (*sigmaScattering);

  //float Zot        = circTmat[threadId+(*nSpT)*0];
  float cotThetaT  = circTmat[threadId+(*nSpT)*1];
  float iDeltaRT   = circTmat[threadId+(*nSpT)*2];
  float ErT        = circTmat[threadId+(*nSpT)*3];
  float Ut         = circTmat[threadId+(*nSpT)*4];
  float Vt         = circTmat[threadId+(*nSpT)*5];

  // add errors of spB-spM and spM-spT pairs and add the correlation term
  // for errors on spM
  float error2 = ErT + ErB +
    2 * (cotThetaB * cotThetaT * varianceRM + varianceZM) * iDeltaRB * iDeltaRT;
  
  float deltaCotTheta = cotThetaB - cotThetaT;
  float deltaCotTheta2 = deltaCotTheta * deltaCotTheta;
  float error;
  float dCotThetaMinusError2;
  
  isPassed[threadId] = true;
  
  // if the error is larger than the difference in theta, no need to
  // compare with scattering
  if (deltaCotTheta2 - error2 > 0) {
    deltaCotTheta = fabs(deltaCotTheta);
    // if deltaTheta larger than the scattering for the lower pT cut, skip
    error = sqrt(error2);
    dCotThetaMinusError2 =
      deltaCotTheta2 + error2 - 2 * deltaCotTheta * error;
    // avoid taking root of scatteringInRegion
    // if left side of ">" is positive, both sides of unequality can be
    // squared
    // (scattering is always positive)
    
    if (dCotThetaMinusError2 > scatteringInRegion2) {
      isPassed[threadId] = false;
    }
  }

  // protects against division by 0
  float dU = Ut - Ub;
  if (dU == 0.) {
    isPassed[threadId] = false;
  }

  // A and B are evaluated as a function of the circumference parameters
  // x_0 and y_0
  float A = (Vt - Vb) / dU;
  float S2 = 1. + A * A;
  float B = Vb - A * Ub;
  float B2 = B * B;
  // sqrt(S2)/B = 2 * helixradius
  // calculated radius must not be smaller than minimum radius
  if (S2 < B2 * (*minHelixDiameter2)) {
    isPassed[threadId] = false;
  }
  
  // 1/helixradius: (B/sqrt(S2))/2 (we leave everything squared)
  float iHelixDiameter2 = B2 / S2;
  // calculate scattering for p(T) calculated from seed curvature
  float pT2scatter = 4 * iHelixDiameter2 * (*pT2perRadius);
  // TODO: include upper pT limit for scatter calc
  // convert p(T) to p scaling by sin^2(theta) AND scale by 1/sin^4(theta)
  // from rad to deltaCotTheta
  float p2scatter = pT2scatter * iSinTheta2;
  // if deltaTheta larger than allowed scattering for calculated pT, skip
  if ((deltaCotTheta2 - error2 > 0) &&
      (dCotThetaMinusError2 >
       p2scatter * (*sigmaScattering) * (*sigmaScattering))) {
    isPassed[threadId] = false;
  }
  // A and B allow calculation of impact params in U/V plane with linear
  // function
  // (in contrast to having to solve a quadratic function in x/y plane)

  float impact   = fabs((A - B * rM) * rM);
  float invHelix = B / sqrt(S2);
  
  if (impact > (*impactMax)){
    isPassed[threadId] = false;
  }  

  // Consider a full reduction to count nTopPass
  // Now just use atomicAdd
  if (isPassed[threadId] == true){
    int pos = atomicAdd(&nTopPass[blockId],1);
    if (pos<*nTopPassLimit){
      //printf("%d %d\n", blockId, nTopPass[blockId]);
      tIndex          [pos+(*nTopPassLimit)*blockIdx.x] = threadIdx.x + (*offset);
      impactparameters[pos+(*nTopPassLimit)*blockIdx.x] = impact;
      curvatures      [pos+(*nTopPassLimit)*blockIdx.x] = invHelix;
      
    }
  }
  
  //__syncthreads();

  /*
  if (threadId == 0 && blockId==0 ){
    printf("%f %f %f %f %f %f  \n", Zob, cotThetaB, iDeltaRB, ErB, Ub, Vb);
    printf("%f %f %f %f %f %f  \n", Zot, cotThetaT, iDeltaRT, ErT, Ut, Vt);
  }
  */

  /*
  if (threadId == 0 && blockId==0 ){
    printf("%f %f \n", iSinTheta2, scatteringInRegion2);
  }
  */
  
  /*
  if (threadId == 0 ){
    int passCount =0;
    for (int i=0; i<blockDim.x; i++){
      if (isPassed[i] == true) passCount++;
    }
    if (passCount >0){
      printf("Pass top seeds: %d \n", passCount);
    }
  }
  */
  /*
  config->seedFilter.filterSeeds_2SpFixed(&threadId, spM, spB, nSpT, spTmat,
  					  isPassed, curvatures, impactParameters, &Zob,
  					  weight, isTriplet);
  */
}
