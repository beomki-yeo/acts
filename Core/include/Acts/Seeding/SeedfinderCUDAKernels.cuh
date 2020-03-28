// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>


namespace Acts{
class CuSeedfinderConfig;
class CuIExperimentCuts;
  
class SeedfinderCUDAKernels {

public: 
  
  static void searchDoublet( dim3 grid, dim3 block,
			     const int* isBottom, 
			     const float* rMvec, const float* zMvec,
			     const int* nSpB, const float* rBvec, const float* zBvec, 
			     //const Acts::CuSeedfinderConfig* config,
			     const float* deltaRMin, const float* deltaRMax,
			     const float* cotThetaMax, 
			     const float* collisionRegionMin, 
			     const float* collisionRegionMax,
			     int* isCompatible 			     
			     );

  static void transformCoordinates( dim3 grid, dim3 block,
				    const int*   isBottom, 
				    const float* spM,
				    const int*   nSpB,
				    const float* spBmat,
				    float* circBmat);
  
  static void searchTriplet( dim3 grid, dim3 block,
			     const int*   offset,
			     const float* spM,
			     const int*   nSpB, const float* rBmat,
			     const int*   nSpT, const float* rTmat,			     
			     const float* circBmat,
			     const float* circTmat,
			     const float* maxScatteringAngle2, const float* sigmaScattering,
			     const float* minHelixDiameter2,    const float* pT2perRadius,
			     const float* impactMax,
			     int* nTopPass,
			     int* tIndex	     
			     //const Acts::CuSeedfinderConfig* config
			     // finder config			     
			     // filter config
			     //const float* deltaInvHelixDiameter,
			     //const float* impactWeightFactor,
			     //const float* deltaRMin,
			     //const float* compatSeedWeight,
			     //const size_t* compatSeedLimit,
			     );
  
private:
  
};
}
