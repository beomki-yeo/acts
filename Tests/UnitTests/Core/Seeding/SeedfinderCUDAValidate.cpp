// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

#include <boost/type_erasure/any_cast.hpp>

#include "Acts/Seeding/BinFinder.hpp"
#include "Acts/Seeding/BinnedSPGroup.hpp"
#include "Acts/Seeding/InternalSeed.hpp"
#include "Acts/Seeding/InternalSpacePoint.hpp"
#include "Acts/Seeding/Seed.hpp"
#include "Acts/Seeding/SeedFilter.hpp"
#include "Acts/Seeding/Seedfinder.hpp"
#include "Acts/Seeding/SpacePointGrid.hpp"

#include "ATLASCuts.hpp"
#include "SpacePoint.hpp"

#include "Acts/Utilities/Platforms/PlatformDef.h"
#include <cuda_profiler_api.h>

std::vector<const SpacePoint*> readFile(std::string filename) {
  std::string line;
  int layer;
  std::vector<const SpacePoint*> readSP;

  std::ifstream spFile(filename);
  if (spFile.is_open()) {
    while (!spFile.eof()) {
      std::getline(spFile, line);
      std::stringstream ss(line);
      std::string linetype;
      ss >> linetype;
      float x, y, z, r, varianceR, varianceZ;
      if (linetype == "lxyz") {
        ss >> layer >> x >> y >> z >> varianceR >> varianceZ;
        r = std::sqrt(x * x + y * y);
        float f22 = varianceR;
        float wid = varianceZ;
        float cov = wid * wid * .08333;
        if (cov < f22)
          cov = f22;
        if (std::abs(z) > 450.) {
          varianceZ = 9. * cov;
          varianceR = .06;
        } else {
          varianceR = 9. * cov;
          varianceZ = .06;
        }
        SpacePoint* sp =
            new SpacePoint{x, y, z, r, layer, varianceR, varianceZ};
        //     if(r < 200.){
        //       sp->setClusterList(1,0);
        //     }
        readSP.push_back(sp);
      }
    }
  }
  return readSP;
}

int main(int argc, char** argv) {
  std::string file{"sp.txt"};
  bool help(false);
  bool quiet(false);

  int opt;
  while ((opt = getopt(argc, argv, "hf:q")) != -1) {
    switch (opt) {
      case 'f':
        file = optarg;
        break;
      case 'q':
        quiet = true;
        break;
      case 'h':
        help = true;
        [[fallthrough]];
      default: /* '?' */
        std::cerr << "Usage: " << argv[0] << " [-hq] [-f FILENAME]\n";
        if (help) {
          std::cout << "      -h : this help" << std::endl;
          std::cout
              << "      -f FILE : read spacepoints from FILE. Default is \""
              << file << "\"" << std::endl;
          std::cout << "      -q : don't print out all found seeds"
                    << std::endl;
        }

        exit(EXIT_FAILURE);
    }
  }
  
  std::ifstream f(file);
  if (!f.good()) {
    std::cerr << "input file \"" << file << "\" does not exist\n";
    exit(EXIT_FAILURE);
  }

  auto start_read = std::chrono::system_clock::now();
  std::vector<const SpacePoint*> spVec = readFile(file);
  auto end_read = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_read = end_read - start_read;

  std::cout << "read " << spVec.size() << " SP from file " << file << " in "
            << elapsed_read.count() << "s" << std::endl;

  /// For CPU seed finder
  Acts::SeedfinderConfig<SpacePoint> config;
  // silicon detector max
  config.rMax = 160.;
  config.deltaRMin = 5.;
  config.deltaRMax = 160.;
  config.collisionRegionMin = -250.;
  config.collisionRegionMax = 250.;
  config.zMin = -2800.;
  config.zMax = 2800.;
  config.maxSeedsPerSpM = 5;
  // 2.7 eta
  config.cotThetaMax = 7.40627;
  config.sigmaScattering = 1.00000;

  config.minPt = 500.;
  config.bFieldInZ = 0.00199724;

  config.beamPos = {-.5, -.5};
  config.impactMax = 10.;

  auto bottomBinFinder = std::make_shared<Acts::BinFinder<SpacePoint>>(
      Acts::BinFinder<SpacePoint>());
  auto topBinFinder = std::make_shared<Acts::BinFinder<SpacePoint>>(
      Acts::BinFinder<SpacePoint>());
  Acts::SeedFilterConfig sfconf;
  Acts::ATLASCuts<SpacePoint> atlasCuts = Acts::ATLASCuts<SpacePoint>();
  config.seedFilter = std::make_unique<Acts::SeedFilter<SpacePoint>>(
      Acts::SeedFilter<SpacePoint>(sfconf, &atlasCuts));  
  Acts::Seedfinder<SpacePoint> seedfinder_cpu(config);

  /// For CUDA seed finder
  Acts::CuSeedfinderConfig cuConfig;
  // silicon detector max
  cuConfig.rMax = 160.;
  cuConfig.deltaRMin = 5.;
  cuConfig.deltaRMax = 160.;
  cuConfig.collisionRegionMin = -250.;
  cuConfig.collisionRegionMax = 250.;
  cuConfig.zMin = -2800.;
  cuConfig.zMax = 2800.;
  cuConfig.maxSeedsPerSpM = 5;
  // 2.7 eta
  cuConfig.cotThetaMax = 7.40627;
  cuConfig.sigmaScattering = 1.00000;

  cuConfig.minPt = 500.;
  cuConfig.bFieldInZ = 0.00199724;

  //cuConfig.beamPos = {-.5, -.5};
  cuConfig.impactMax = 10.;
  //Acts::CuATLASCuts cuAtlasCuts = Acts::CuATLASCuts();
  Acts::CuIExperimentCuts cuExpCuts = Acts::CuIExperimentCuts();
  //cuConfig.seedFilter = Acts::CuSeedFilter(sfconf, cuExpCuts);
  
  Acts::CuSeedfinder<SpacePoint> seedfinder_cuda(cuConfig);

  std::cout << "size of CuSeedfinderConfig: " << sizeof(Acts::CuSeedfinderConfig) << std::endl;
  std::cout << "size of CuSeedFilter: "       << sizeof(Acts::CuSeedFilter) << std::endl;
  std::cout << "size of CuIExperimentCuts : " << sizeof(Acts::CuIExperimentCuts) << std::endl;
  
  // covariance tool, sets covariances per spacepoint as required
  auto ct = [=](const SpacePoint& sp, float, float, float) -> Acts::Vector2D {
    return {sp.varianceR, sp.varianceZ};
  };
  
  // setup spacepoint grid config
  Acts::SpacePointGridConfig gridConf;
  gridConf.bFieldInZ = config.bFieldInZ;
  gridConf.minPt = config.minPt;
  gridConf.rMax = config.rMax;
  gridConf.zMax = config.zMax;
  gridConf.zMin = config.zMin;
  gridConf.deltaRMax = config.deltaRMax;
  gridConf.cotThetaMax = config.cotThetaMax;
  // create grid with bin sizes according to the configured geometry
  std::unique_ptr<Acts::SpacePointGrid<SpacePoint>> grid =
      Acts::SpacePointGridCreator::createGrid<SpacePoint>(gridConf);
  auto spGroup = Acts::BinnedSPGroup<SpacePoint>(spVec.begin(), spVec.end(), ct,
                                                 bottomBinFinder, topBinFinder,
                                                 std::move(grid), config);

  int nGroupToIterate = 2;
  int group_count;
  ///////// CPU
  group_count=0;
  std::vector<std::vector<Acts::Seed<SpacePoint>>> seedVector_cpu;
  auto start_cpu = std::chrono::system_clock::now();
  auto groupIt = spGroup.begin();
  auto endOfGroups = spGroup.end();
  for (; !(groupIt == endOfGroups); ++groupIt) {
    seedVector_cpu.push_back(seedfinder_cpu.createSeedsForGroup(
        groupIt.bottom(), groupIt.middle(), groupIt.top()));
    group_count++;
    if (group_count >= nGroupToIterate) break;
  }
  auto end_cpu = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsec_cpu = end_cpu - start_cpu;
  std::cout << "CPU Time: " << elapsec_cpu.count() << std::endl;
  std::cout << "Number of regions: " << seedVector_cpu.size() << std::endl;
  

  ///////// CUDA
  cudaProfilerStart();
  
  group_count=0;
  std::vector<std::vector<Acts::Seed<SpacePoint>>> seedVector_cuda;
  auto start_cuda = std::chrono::system_clock::now();
  groupIt = spGroup.begin();
  
  for (; !(groupIt == endOfGroups); ++groupIt) {
    seedVector_cuda.push_back(seedfinder_cuda.createSeedsForGroup(
        groupIt.bottom(), groupIt.middle(), groupIt.top()));
    group_count++;
    if (group_count >= nGroupToIterate) break;
  }
  
  auto end_cuda = std::chrono::system_clock::now();  
  std::chrono::duration<double> elapsec_cuda = end_cuda - start_cuda;
  std::cout << "CUDA Time: " << elapsec_cuda.count() << std::endl;
  std::cout << "Number of regions: " << seedVector_cpu.size() << std::endl;

  cudaProfilerStop();
  /*
  int numSeeds = 0;
  for (auto& outVec : seedVector_cpu) {
    numSeeds += outVec.size();
  }
  std::cout << "Number of seeds generated: " << numSeeds << std::endl;
  if (!quiet) {
    for (auto& regionVec : seedVector_cpu) {
      for (size_t i = 0; i < regionVec.size(); i++) {
        const Acts::Seed<SpacePoint>* seed = &regionVec[i];
        const SpacePoint* sp = seed->sp()[0];
        std::cout << " (" << sp->x() << ", " << sp->y() << ", " << sp->z()
                  << ") ";
        sp = seed->sp()[1];
        std::cout << sp->surface << " (" << sp->x() << ", " << sp->y() << ", "
                  << sp->z() << ") ";
        sp = seed->sp()[2];
        std::cout << sp->surface << " (" << sp->x() << ", " << sp->y() << ", "
                  << sp->z() << ") ";
        std::cout << std::endl;
      }
    }
  }
  */
  return 0;
}


