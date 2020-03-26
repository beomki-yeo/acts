// This file is part of the Acts project.
//
// Copyright (C) 2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <memory>
#include "Acts/Seeding/InternalSeed.hpp"

namespace Acts {
/// @class IExperimentCuts can be used to increase or decrease seed weights
///  based on the space points used in a seed. Seed weights are also
/// influenced by the SeedFilter default implementation. This tool is also used
/// to decide if a seed passes a seed weight cut. As the weight is stored in
/// seeds, there are two distinct methods.
template <typename SpacePoint>
class IExperimentCuts {
 public:
  virtual ~IExperimentCuts() = default;
  /// Returns seed weight bonus/malus depending on detector considerations.
  /// @param bottom bottom space point of the current seed
  /// @param middle middle space point of the current seed
  /// @param top top space point of the current seed
  /// @return seed weight to be added to the seed's weight
  virtual float seedWeight(const InternalSpacePoint<SpacePoint>& bottom,
                           const InternalSpacePoint<SpacePoint>& middle,
                           const InternalSpacePoint<SpacePoint>& top) const = 0;
  /// @param weight the current seed weight
  /// @param bottom bottom space point of the current seed
  /// @param middle middle space point of the current seed
  /// @param top top space point of the current seed
  /// @return true if the seed should be kept, false if the seed should be
  /// discarded
  virtual bool singleSeedCut(
      float weight, const InternalSpacePoint<SpacePoint>& bottom,
      const InternalSpacePoint<SpacePoint>& middle,
      const InternalSpacePoint<SpacePoint>& top) const = 0;
  /// @param seeds contains pairs of weight and seed created for one middle
  /// space
  /// point
  /// @return vector of seeds that pass the cut
  virtual std::vector<
      std::pair<float, std::unique_ptr<const InternalSeed<SpacePoint>>>>
  cutPerMiddleSP(
      std::vector<
          std::pair<float, std::unique_ptr<const InternalSeed<SpacePoint>>>>
          seeds) const = 0;

};

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif
  
class CuIExperimentCuts{
public:
  //virtual ~CuIExperimentCuts() = default;  
CUDA_HOSTDEV virtual float seedWeight(const float* bottom,
				      const float* middle,
				      const float* top) const = 0;
CUDA_HOSTDEV virtual bool singleSeedCut(float weight,
					const float* bottom,
					const float* middle,
					const float* top) const = 0;
};
  
}  // namespace Acts

/*
#ifdef __CUDACC__
class IcuExperimentCuts{
__device__ virtual float seedWeight(const float* bottom,
				   const float* middle,
				   const float* top);
__device__ virtual bool singleSeedCut(float weight,
			      const float* bottom,
			      const float* middle,
			      const float* top);
};
#endif
*/
