// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

namespace Acts{
  class CPU;
}

#ifdef ACTS_HAS_CUDA

#include "Acts/Utilities/Platforms/CUDA/CudaScalar.cu"
#include "Acts/Utilities/Platforms/CUDA/CudaVector.cu"
#include "Acts/Utilities/Platforms/CUDA/CudaMatrix.cu"
#include "Acts/Utilities/Platforms/CUDA/CpuVector.hpp"
#include "Acts/Utilities/Platforms/CUDA/CpuMatrix.hpp"

namespace Acts{
  class CUDA;
}

#endif



