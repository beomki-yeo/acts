// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class SeedfinderKernels {

public: 
  //template < typename Var_t >
  static void SearchDoublet( dim3 grid, dim3 block, const float* rBvec, const float* zBvec, const float* rM, const float* zM, const bool* isBottom, const float* deltaRMin, const float* deltaRMax, const float* cotThetaMax, const float* collisionRegionMin, const float* collisionRegionMax, int* isCompatible );

  static void test();

private:
  
};
