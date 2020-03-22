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
    const InternalSpacePoint<external_spacepoint_t>& spM,
    //const float& rM, const float& zM, const float& varianceRM, const float& varianceZM,
    const float& deltaRMin, const float& deltaRMax,
    const float& cotThetaMax,
    const float& collisionRegionMin,
    const float& collisionRegionMax){

    float rM = spM.radius();
    float zM = spM.z();
    float varianceRM = spM.varianceR();
    float varianceZM = spM.varianceZ();
    
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

  template< typename external_spacepoint_t, typename sp_range_t > 
  void SeedfinderCPUFunctions<external_spacepoint_t, sp_range_t>::transformCoordinates(
       std::vector<const InternalSpacePoint<external_spacepoint_t>*>& vec,
       const InternalSpacePoint<external_spacepoint_t>& spM, bool bottom,
       std::vector<LinCircle>& linCircleVec) {
    float xM = spM.x();
    float yM = spM.y();
    float zM = spM.z();
    float rM = spM.radius();
    float varianceZM = spM.varianceZ();
    float varianceRM = spM.varianceR();
    float cosPhiM = xM / rM;
    float sinPhiM = yM / rM;
    for (auto sp : vec) {
      float deltaX = sp->x() - xM;
      float deltaY = sp->y() - yM;
      float deltaZ = sp->z() - zM;
      // calculate projection fraction of spM->sp vector pointing in same
      // direction as
      // vector origin->spM (x) and projection fraction of spM->sp vector pointing
      // orthogonal to origin->spM (y)
      float x = deltaX * cosPhiM + deltaY * sinPhiM;
      float y = deltaY * cosPhiM - deltaX * sinPhiM;
      // 1/(length of M -> SP)
      float iDeltaR2 = 1. / (deltaX * deltaX + deltaY * deltaY);
      float iDeltaR = std::sqrt(iDeltaR2);
      //
      int bottomFactor = 1 * (int(!bottom)) - 1 * (int(bottom));
      // cot_theta = (deltaZ/deltaR)
      float cot_theta = deltaZ * iDeltaR * bottomFactor;
      // VERY frequent (SP^3) access
      LinCircle l;
      l.cotTheta = cot_theta;
      // location on z-axis of this SP-duplet
      l.Zo = zM - rM * cot_theta;
      l.iDeltaR = iDeltaR;
      // transformation of circle equation (x,y) into linear equation (u,v)
      // x^2 + y^2 - 2x_0*x - 2y_0*y = 0
      // is transformed into
      // 1 - 2x_0*u - 2y_0*v = 0
      // using the following m_U and m_V
      // (u = A + B*v); A and B are created later on
      l.U = x * iDeltaR2;
      l.V = y * iDeltaR2;
      // error term for sp-pair without correlation of middle space point
      l.Er = ((varianceZM + sp->varianceZ()) +
	      (cot_theta * cot_theta) * (varianceRM + sp->varianceR())) * iDeltaR2;
      linCircleVec.push_back(l);
    }              
  }
}
