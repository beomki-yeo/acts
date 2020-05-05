// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cuda_runtime.h>
#include <iostream>

typedef struct{
  int bIndex;
  int tIndex;
  float topRadius;  
  float impactParameter;  
  float invHelixDiameter;
  float weight;
} Triplet;

namespace Acts{
  
void searchDoublet( const dim3 grid, const dim3 block,
		    const int* nSpM, const float* spMmat,
		    const int* nSpB, const float* spBmat,
		    const int* nSpT, const float* spTmat,
		    const float* deltaRMin,
		    const float* deltaRMax,
		    const float* cotThetaMax, 
		    const float* collisionRegionMin, 
		    const float* collisionRegionMax,
		    int*  nSpMcomp,
		    int*  nSpBcompPerSpM_Max,
		    int*  nSpTcompPerSpM_Max,			     
		    int*  nSpBcompPerSpM,
		    int*  nSpTcompPerSpM,
		    int*  McompIndex,			     
		    int*  BcompIndex,
		    int*  tmpBcompIndex,
		    int*  TcompIndex,
		    int*  tmpTcompIndex
		    );
  
void transformCoordinate( const dim3 grid, const dim3 block,
			  const int*   nSpM,
			  const float* spMmat,
			  const int*   McompIndex,
			  const int*   nSpB,
			  const float* spBmat,
			  const int*   nSpBcompPerSpM_Max,
			  const int*   BcompIndex,
			  const int*   nSpT,
			  const float* spTmat,
			  const int*   nSpTcompPerSpM_Max,
			  const int*   TcompIndex,
			  float* spMcompMat,			    
			  float* spBcompMatPerSpM,
			  float* circBcompMatPerSpM,
			  float* spTcompMatPerSpM,
			  float* circTcompMatPerSpM);
  
void searchTriplet( const dim3 grid, const dim3 block,
		    //const int*   offset,
		    const int*   nSpTcompPerSpM_cpu,
		    const int*   nSpTcompPerSpM_cuda,
		    const int*   nSpMcomp,
		    const float* spMcompMat,
		    const int*   nSpBcompPerSpM_Max,
		    const int*   BcompIndex,
		    const float* circBcompMatPerSpM,
		    const int*   nSpTcompPerSpM_Max,
		    const int*   TcompIndex,
		    const float* spTcompMatPerSpM,
		    const float* circTcompMatPerSpM,
		    const float* maxScatteringAngle2,
		    const float* sigmaScattering,
		    const float* minHelixDiameter2,
		    const float* pT2perRadius,
		    const float* impactMax,
		    const int*   nTrplPerSpMLimit,
		    const float* deltaInvHelixDiameter,
		    const float* impactWeightFactor,
		    const float* deltaRMin,
		    const float* compatSeedWeight,
		    const size_t* compatSeedLimit_cpu,
		    const size_t* compatSeedLimit_cuda,
		    int* nTrplPerSpM,
		    int* tIndex,
		    int* bIndex,
		    float* curvatures,
		    float* impactparameters,
		    Triplet* TripletsPerSpM,
		    cudaStream_t* stream
		    );
}
