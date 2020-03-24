// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <numeric>
#include <type_traits>
#include <algorithm>
#include <chrono>
#include <Acts/Seeding/SeedfinderCPUFunctions.hpp>
#include <Acts/Seeding/SeedfinderCUDAKernels.cuh>

#define WARP_SIZE 64
#define MAX_BLOCK_SIZE 1024

namespace Acts {

  template <typename external_spacepoint_t, typename architecture_t>
  Seedfinder<external_spacepoint_t, architecture_t>::Seedfinder(
    Acts::SeedfinderConfig<external_spacepoint_t> config)
    : m_config(std::move(config)) {
  // calculation of scattering using the highland formula
  // convert pT to p once theta angle is known
  m_config.highland = 13.6 * std::sqrt(m_config.radLengthPerSeed) *
                      (1 + 0.038 * std::log(m_config.radLengthPerSeed));
  float maxScatteringAngle = m_config.highland / m_config.minPt;
  m_config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;
  // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV and
  // millimeter
  // TODO: change using ACTS units
  m_config.pTPerHelixRadius = 300. * m_config.bFieldInZ;
  m_config.minHelixDiameter2 =
      std::pow(m_config.minPt * 2 / m_config.pTPerHelixRadius, 2);
  m_config.pT2perRadius =
      std::pow(m_config.highland / m_config.pTPerHelixRadius, 2);
  }

  // CPU seed finding
  template< typename external_spacepoint_t, typename architecture_t>
  template< typename T, typename sp_range_t>
  typename std::enable_if< std::is_same<T, Acts::CPU>::value, std::vector<Seed<external_spacepoint_t> > >::type
  Seedfinder<external_spacepoint_t, architecture_t>::createSeedsForGroup(
    sp_range_t bottomSPs, sp_range_t middleSPs, sp_range_t topSPs) const {
  std::vector<Seed<external_spacepoint_t>> outputVec;

  int i_m=0;
  for (auto spM : middleSPs) {    
    i_m++;
    
    float rM = spM->radius();
    float zM = spM->z();
    float varianceRM = spM->varianceR();
    float varianceZM = spM->varianceZ();

    // Doublet search    
    auto compatBottomSP =
      SeedfinderCPUFunctions<external_spacepoint_t,
			     sp_range_t>::SearchDoublet(true, bottomSPs, *spM, m_config);
    
    // no bottom SP found -> try next spM
    if (compatBottomSP.empty()) {
      continue;
    }

    auto compatTopSP =
      SeedfinderCPUFunctions<external_spacepoint_t,
			     sp_range_t>::SearchDoublet(false, topSPs, *spM, m_config);

    // no top SP found -> try next spM
    if (compatTopSP.empty()) {
      continue;
    }

    std::cout << i_m << "   CPU compatible Hits: " << compatBottomSP.size() << "  " << compatTopSP.size() << std::endl;

    
    // contains parameters required to calculate circle with linear equation
    // ...for bottom-middle
    std::vector<LinCircle> linCircleBottom;
    // ...for middle-top
    std::vector<LinCircle> linCircleTop;
    
    SeedfinderCPUFunctions<external_spacepoint_t,sp_range_t>::transformCoordinates(compatBottomSP, *spM, true, linCircleBottom);
    SeedfinderCPUFunctions<external_spacepoint_t,sp_range_t>::transformCoordinates(compatTopSP, *spM, false, linCircleTop);
    /*
    auto seedsPerSpM = SeedfinderCPUFunctions<external_spacepoint_t,sp_range_t>::SearchTriplet(*spM, compatBottomSP, compatTopSP, linCircleBottom, linCircleTop, m_config);
    
    m_config.seedFilter->filterSeeds_1SpFixed(seedsPerSpM, outputVec);
    */
  }
  
  return outputVec;
  }

  
  // CUDA seed finding
  template< typename external_spacepoint_t, typename architecture_t>
  template< typename T, typename sp_range_t>
  typename std::enable_if< std::is_same<T, Acts::CUDA>::value, std::vector<Seed<external_spacepoint_t> > >::type
  Seedfinder<external_spacepoint_t, architecture_t>::createSeedsForGroup(
    sp_range_t bottomSPs, sp_range_t middleSPs, sp_range_t topSPs) const {
  std::vector<Seed<external_spacepoint_t>> outputVec;

  int isBottom_cpu;
  CUDA::Buffer<int> isBottom_cuda(1);
  
  /* ---------------------------------
     Algorithm 0. Matrix Flattening 
  ------------------------------------*/
  std::vector<float> rM_cpu;
  std::vector<float> zM_cpu;

  for (auto sp: middleSPs){
    rM_cpu.push_back(sp->radius());
    zM_cpu.push_back(sp->z());
  }
  
  std::vector<float> rB_cpu;
  std::vector<float> zB_cpu;

  for (auto sp: bottomSPs){
    rB_cpu.push_back(sp->radius());
    zB_cpu.push_back(sp->z());
  }

  std::vector<float> rT_cpu;
  std::vector<float> zT_cpu;

  for (auto sp: topSPs){
    rT_cpu.push_back(sp->radius());
    zT_cpu.push_back(sp->z());
  }

  if (rB_cpu.size() == 0 || rM_cpu.size() == 0 || rT_cpu.size() == 0) return outputVec;
  
  /* ------------------------------------
     Algorithm 1. Doublet Search (DS)
  ---------------------------------------*/
  
  int  offset;
  int  BlockSize;
  dim3 DS_BlockSize;
  dim3 DS_GridSize(rM_cpu.size(),1,1);
  
  CUDA::Buffer<float> deltaRMin_cuda(1,          &m_config.deltaRMin);
  CUDA::Buffer<float> deltaRMax_cuda(1,          &m_config.deltaRMax);
  CUDA::Buffer<float> cotThetaMax_cuda(1,        &m_config.cotThetaMax);
  CUDA::Buffer<float> collisionRegionMin_cuda(1, &m_config.collisionRegionMin);
  CUDA::Buffer<float> collisionRegionMax_cuda(1, &m_config.collisionRegionMax);  
  CUDA::Buffer<float> rM_cuda(rM_cpu.size(), &rM_cpu[0]);
  CUDA::Buffer<float> zM_cuda(zM_cpu.size(), &zM_cpu[0]);

  ///// For bottom space points
  isBottom_cpu = true;
  isBottom_cuda.SetData(&isBottom_cpu,1);	

  CUDA::Buffer<int>   isCompatBottomSP_cuda(rB_cpu.size()*rM_cpu.size());
  auto isCompatBottomMat_cpu  = CPU::Matrix<int>(rM_cpu.size(), rB_cpu.size());
  
  offset=0;
  while(offset<rB_cpu.size()){
    BlockSize = fmin(MAX_BLOCK_SIZE,rB_cpu.size());
    BlockSize = fmin(BlockSize,rB_cpu.size()-offset);
    DS_BlockSize = dim3(BlockSize,1,1);    
    CUDA::Buffer<float> rB_cuda(BlockSize, &rB_cpu[offset]);    
    CUDA::Buffer<float> zB_cuda(BlockSize, &zB_cpu[offset]);  
    SeedfinderCUDAKernels::SearchDoublet( DS_GridSize, DS_BlockSize, 
					  isBottom_cuda.data(),
					  rB_cuda.data(), zB_cuda.data(), 
					  rM_cuda.data(), zM_cuda.data(), 
					  deltaRMin_cuda.data(), deltaRMax_cuda.data(), 
					  cotThetaMax_cuda.data(),
					  collisionRegionMin_cuda.data(),collisionRegionMax_cuda.data(),
					  isCompatBottomSP_cuda.data(offset*rM_cpu.size()) );
    offset+=BlockSize;
  }
  // Rearrange the doublet (Note: It spents time more than CUDA kernel...)
  // Prev: [mid1: bot_1, ..., bot_N]    [mid2: bot_1, ..., bot_N] ...    [midN: bot_1, ..., bot_N]
  //       [mid1: bot_N+1, ..., bot_2N] [mid2: bot_N+1, ..., bot_2N] ... [midN: bot_N+1, ..., bot_2N]
  //       ...
  //
  // New : [mid1: bot_1, ..., bot_TN] [mid2: bot_1, ..., bot_TN] ... [midN: bot_1, ..., bot_TN]
  auto bottomBuffer = CPU::Buffer<int>(rB_cpu.size()*rM_cpu.size(),
				       isCompatBottomSP_cuda.dataHost(rB_cpu.size()*rM_cpu.size()));
  offset=0;
  while(offset<rB_cpu.size()){
    BlockSize = fmin(MAX_BLOCK_SIZE,rB_cpu.size());
    BlockSize = fmin(BlockSize,rB_cpu.size()-offset);
    for (int i_m=0; i_m<rM_cpu.size(); i_m++){     
      std::copy(bottomBuffer.data()+offset*rM_cpu.size()+i_m*BlockSize,
		bottomBuffer.data()+offset*rM_cpu.size()+(i_m+1)*BlockSize,
		isCompatBottomMat_cpu.Get(i_m,offset));
		//isCompatBottomSP_cpu.get()+i_m*rB_cpu.size()+offset);
    }
    offset+= BlockSize;
  }
  
  ///// For top space points
  isBottom_cpu = false;
  isBottom_cuda.SetData(&isBottom_cpu,1);	
  CUDA::Buffer<int>   isCompatTopSP_cuda(rT_cpu.size()*rM_cpu.size());
  auto isCompatTopMat_cpu = CPU::Matrix<int>(rM_cpu.size(), rT_cpu.size());
  
  offset=0;
  while(offset<rT_cpu.size()){
    BlockSize = fmin(MAX_BLOCK_SIZE,rT_cpu.size());
    BlockSize = fmin(BlockSize,rT_cpu.size()-offset);
    DS_BlockSize = dim3(BlockSize,1,1);    
    CUDA::Buffer<float> rT_cuda(BlockSize, &rT_cpu[offset]);    
    CUDA::Buffer<float> zT_cuda(BlockSize, &zT_cpu[offset]);  
    SeedfinderCUDAKernels::SearchDoublet( DS_GridSize, DS_BlockSize, 
					  isBottom_cuda.data(),
					  rT_cuda.data(), zT_cuda.data(), 
					  rM_cuda.data(), zM_cuda.data(), 
					  deltaRMin_cuda.data(), deltaRMax_cuda.data(), 
					  cotThetaMax_cuda.data(),
					  collisionRegionMin_cuda.data(),collisionRegionMax_cuda.data(),
					  isCompatTopSP_cuda.data(offset*rM_cpu.size()) );
    offset+= BlockSize;
  }
  // Rearrange the doublet (Note: It spents time more than CUDA kernel...)
  // Prev: [mid1: top_1, ..., top_N]    [mid2: top_1, ..., top_N] ...    [midN: top_1, ..., top_N]
  //       [mid1: top_N+1, ..., top_2N] [mid2: top_N+1, ..., top_2N] ... [midN: top_N+1, ..., top_2N]
  //       ...
  //
  // New : [mid1: top_1, ..., top_TN] [mid2: top_1, ..., top_TN] ... [midN: top_1, ..., top_TN]
  auto topBuffer = CPU::Buffer<int>(rT_cpu.size()*rM_cpu.size(),
				    isCompatTopSP_cuda.dataHost(rT_cpu.size()*rM_cpu.size()));
  offset=0;
  while(offset<rT_cpu.size()){
    BlockSize = fmin(MAX_BLOCK_SIZE,rT_cpu.size());
    BlockSize = fmin(BlockSize,rT_cpu.size()-offset);
    for (int i_m=0; i_m<rM_cpu.size(); i_m++){
      std::copy(topBuffer.data()+offset*rM_cpu.size()+i_m*BlockSize,
		topBuffer.data()+offset*rM_cpu.size()+(i_m+1)*BlockSize,
		isCompatTopMat_cpu.Get(i_m,offset));
    }
    offset+= BlockSize;
  }
  
  /* ----------------------------------------
     Algorithm 2. Transform coordinate (TC)
  -------------------------------------------*/

  // Make Matrix for compatible SPs
  /*
  for (int i_m=0; i_m<rM_cpu.size(); i_m++){
    
  }
  */
  /*
  CPU::Matrix<float> xyzrMcomp_cpu;
  std::vector< CPU::Matrix<float> > xyzrBcomp_cpu;
  std::vector< CPU::Matrix<float> > xyzrTcomp_cpu;
  */
  /*
  std::vector<float> xMcomp_cpu;
  std::vector<float> yMcomp_cpu;
  std::vector<float> zMcomp_cpu;
  std::vector<float> rMcomp_cpu;

  std::vector< std::vector<float> > xBcomp_cpu;
  std::vector< std::vector<float> > yBcomp_cpu;
  std::vector< std::vector<float> > zBcomp_cpu;
  std::vector< std::vector<float> > rBcomp_cpu;

  std::vector< std::vector<float> > xTcomp_cpu;
  std::vector< std::vector<float> > yTcomp_cpu;
  std::vector< std::vector<float> > zTcomp_cpu;
  std::vector< std::vector<float> > rTcomp_cpu;
  */
  
  int nMiddle = rM_cpu.size();
  int nBottom = rB_cpu.size();
  int nTop    = rT_cpu.size();
  
  for (int i_m=0; i_m<nMiddle; i_m++){
    //if(i_m>20) break;
    //std::cout << "Middle Index: " << i_m << "  " << nBottom << "  " << nTop  << std::endl;
    
    int nBottomCompat=0;    
    int nTopCompat=0;
    auto botRow = isCompatBottomMat_cpu.GetRow(i_m);
    auto topRow = isCompatTopMat_cpu.GetRow(i_m);
    
    for (int i_b=0; i_b<nBottom; i_b++){
      //std::cout << botRow[i_b] << "  ";
      if (botRow[i_b]) nBottomCompat++;
    }
    //std::cout << std::endl;
    
    for (int i_t=0; i_t<nTop; i_t++){
      //std::cout << topRow[i_t] << "  ";
      if (topRow[i_t]) nTopCompat++;
    }
    //std::cout << std::endl;
    
    if (nBottomCompat && nTopCompat){
      std::cout << " CUDA compatible hits: " << nBottomCompat << "  " << nTopCompat << std::endl;
    }

    delete botRow;
    delete topRow;
  }
  
  return outputVec;
  
  }  // namespace Acts
}
