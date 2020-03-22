// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Seeding/InternalSeed.hpp"
#include "Acts/Seeding/InternalSpacePoint.hpp"

namespace Acts{

  template< typename external_spacepoint_t, typename sp_range_t >
  class SeedfinderCPUFunctions {

  public: 
    
    static std::vector<const InternalSpacePoint<external_spacepoint_t>*>
    SearchDoublet(bool isBottom, sp_range_t& SPs,
		  const InternalSpacePoint<external_spacepoint_t>& spM,
		  //const float& rM, const float& zM, const float& varianceRM, const float& varianceZM,
		  const float& deltaRMin, const float& deltaRMax,
		  const float& cotThetaMax, 
		  const float& collisionRegionMin,
		  const float& collisionRegionMax);

    static void transformCoordinates(std::vector<const InternalSpacePoint<external_spacepoint_t>*>& vec,
				     const InternalSpacePoint<external_spacepoint_t>& spM, bool bottom,
				     std::vector<LinCircle>& linCircleVec);
    
  private:
    
  };
  
} // namespace Acts

#include "Acts/Seeding/SeedfinderCPUFunctions.ipp"
