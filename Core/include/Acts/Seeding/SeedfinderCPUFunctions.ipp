// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

namespace Acts{

  template< typename external_spacepoint_t, typename sp_range_t >
  std::vector<const InternalSpacePoint<external_spacepoint_t>*>
  SeedfinderCPUFunctions<external_spacepoint_t, sp_range_t>::SearchDoublet(
    bool isBottom, sp_range_t& SPs,
    const float& rM, const float& zM, const float& varianceRM, const float& varianceZM,
    const float& deltaRMin, const float& deltaRMax,
    const float& cotThetaMax,
    const float& collisionRegionMin,
    const float& collisionRegionMax){

    std::vector<const InternalSpacePoint<external_spacepoint_t>*>
      compatSPs;
  
    // For bottom space points
    if (isBottom){
                  
      for (auto sp : SPs) {
	float rB = sp->radius();
	float deltaR = rM - rB;
	// if r-distance is too big, try next SP in bin
	if (deltaR > deltaRMax) {
	  continue;
	}
	// if r-distance is too small, break because bins are NOT r-sorted
	if (deltaR < deltaRMin) {
	  continue;
	}
	// ratio Z/R (forward angle) of space point duplet
	float cotTheta = (zM - sp->z()) / deltaR;
	if (std::fabs(cotTheta) > cotThetaMax) {
	  continue;
	}
	// check if duplet origin on z axis within collision region
	float zOrigin = zM - rM * cotTheta;
	if (zOrigin < collisionRegionMin ||
	    zOrigin > collisionRegionMax) {
	  continue;
	}
	compatSPs.push_back(sp);
      }
    }

    // For top space points
    else if (!isBottom){
      for (auto sp : SPs) {
	float rT = sp->radius();
	float deltaR = rT - rM;
	// this condition is the opposite of the condition for bottom SP
	if (deltaR < deltaRMin) {
	  continue;
	}
	if (deltaR > deltaRMax) {
	  break;
	}
	
	float cotTheta = (sp->z() - zM) / deltaR;
	if (std::fabs(cotTheta) > cotThetaMax) {
	  continue;
	}
	float zOrigin = zM - rM * cotTheta;
	if (zOrigin < collisionRegionMin ||
	    zOrigin > collisionRegionMax) {
	  continue;
	}
	compatSPs.push_back(sp);
      }            
    }

    return compatSPs;
  }  
}
