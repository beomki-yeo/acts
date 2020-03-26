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

  template <typename external_spacepoint_t>
  Seedfinder<external_spacepoint_t>::Seedfinder(
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
  
  template< typename external_spacepoint_t>
  template <typename sp_range_t>
  std::vector<Seed<external_spacepoint_t>> Seedfinder<external_spacepoint_t>::createSeedsForGroup(
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
			     sp_range_t>::searchDoublet(true, bottomSPs, *spM, m_config);
    
    // no bottom SP found -> try next spM
    if (compatBottomSP.empty()) {
      continue;
    }

    auto compatTopSP =
      SeedfinderCPUFunctions<external_spacepoint_t,
			     sp_range_t>::searchDoublet(false, topSPs, *spM, m_config);

    // no top SP found -> try next spM
    if (compatTopSP.empty()) {
      continue;
    }    
    // contains parameters required to calculate circle with linear equation
    
    // ...for bottom-middle
    std::vector<LinCircle> linCircleBottom;
    // ...for middle-top
    std::vector<LinCircle> linCircleTop;
    
    SeedfinderCPUFunctions<external_spacepoint_t,sp_range_t>::transformCoordinates(compatBottomSP, *spM, true, linCircleBottom);
    SeedfinderCPUFunctions<external_spacepoint_t,sp_range_t>::transformCoordinates(compatTopSP, *spM, false, linCircleTop);

    std::cout << i_m << "   CPU Compatible Hits: " << compatBottomSP.size() << "  " << compatTopSP.size() << std::endl;
    
    //int i_b = 0;    
    //for (auto circ: linCircleBottom){      
    // std::cout << i_b << "  " << circ.Zo << "  " << circ.cotTheta << "  " << circ.iDeltaR << "  " << circ.Er << "  " << circ.U << "  " << circ.V << std::endl;
    //  i_b++;
    //}
       
    //auto seedsPerSpM = SeedfinderCPUFunctions<external_spacepoint_t,sp_range_t>::searchTriplet(*spM, compatBottomSP, compatTopSP, linCircleBottom, linCircleTop, m_config);
    //m_config.seedFilter->filterSeeds_1SpFixed(seedsPerSpM, outputVec);
   
  }
  
  return outputVec;
  }

  template <typename external_spacepoint_t>
  CuSeedfinder<external_spacepoint_t>::CuSeedfinder(
       Acts::CuSeedfinderConfig config)  
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
    
  // CUDA seed finding
  template< typename external_spacepoint_t>
  template <typename sp_range_t>
  std::vector<Seed<external_spacepoint_t>>
  CuSeedfinder<external_spacepoint_t>::createSeedsForGroup(
    sp_range_t bottomSPs, sp_range_t middleSPs, sp_range_t topSPs) const {
  std::vector<Seed<external_spacepoint_t>> outputVec;

  int isBottom_cpu;
  CUDA::Buffer<int> isBottom_cuda(1);
  
  /*----------------------------------
     Algorithm 0. Matrix Flattening 
  ----------------------------------*/

  // Get Size of spacepoints
  int nMiddle = 0;
  int nBottom = 0;
  int nTop    = 0;

  for (auto sp: middleSPs) nMiddle++;
  for (auto sp: bottomSPs) nBottom++;
  for (auto sp: topSPs)    nTop++;

  if (nMiddle == 0 || nBottom == 0 || nTop == 0) return outputVec;
  
  // Define Matrix and Do flattening
  CPU::Matrix<float> spMmat_cpu(nMiddle, 6); // x y z r varR varZ
  CPU::Matrix<float> spBmat_cpu(nBottom, 6);
  CPU::Matrix<float> spTmat_cpu(nTop   , 6);
    
  int i_m=0;
  for (auto sp: middleSPs){
    spMmat_cpu.SetEl(i_m,0,sp->x());
    spMmat_cpu.SetEl(i_m,1,sp->y());
    spMmat_cpu.SetEl(i_m,2,sp->z());
    spMmat_cpu.SetEl(i_m,3,sp->radius());
    spMmat_cpu.SetEl(i_m,4,sp->varianceR());
    spMmat_cpu.SetEl(i_m,5,sp->varianceZ());
    i_m++;
  }

  int i_b=0;
  for (auto sp: bottomSPs){
    spBmat_cpu.SetEl(i_b,0,sp->x());
    spBmat_cpu.SetEl(i_b,1,sp->y());
    spBmat_cpu.SetEl(i_b,2,sp->z());
    spBmat_cpu.SetEl(i_b,3,sp->radius());
    spBmat_cpu.SetEl(i_b,4,sp->varianceR());
    spBmat_cpu.SetEl(i_b,5,sp->varianceZ());
    i_b++;
  }

  int i_t=0;
  for (auto sp: topSPs){
    spTmat_cpu.SetEl(i_t,0,sp->x());
    spTmat_cpu.SetEl(i_t,1,sp->y());
    spTmat_cpu.SetEl(i_t,2,sp->z());
    spTmat_cpu.SetEl(i_t,3,sp->radius());
    spTmat_cpu.SetEl(i_t,4,sp->varianceR());
    spTmat_cpu.SetEl(i_t,5,sp->varianceZ());
    i_t++;    
  }

  /* ------------------------------------
     Algorithm 1. Doublet Search (DS)
  ---------------------------------------*/
  
  int  offset;
  int  BlockSize;
  dim3 DS_BlockSize;
  dim3 DS_GridSize(nMiddle,1,1);
  
  //CUDA::Buffer<float> deltaRMin_cuda(1, &m_config.deltaRMin, 1);
  //CUDA::Buffer<float> deltaRMax_cuda(1, &m_config.deltaRMax, 1);
  //CUDA::Buffer<float> cotThetaMax_cuda(1, &m_config.cotThetaMax, 1);
  //CUDA::Buffer<float> collisionRegionMin_cuda(1, &m_config.collisionRegionMin, 1);
  //CUDA::Buffer<float> collisionRegionMax_cuda(1, &m_config.collisionRegionMax, 1);  
  CUDA::Buffer<float> rM_cuda(nMiddle, spMmat_cpu.GetEl(0,3), nMiddle);
  CUDA::Buffer<float> zM_cuda(nMiddle, spMmat_cpu.GetEl(0,2), nMiddle);
  CUDA::Buffer<Acts::CuSeedfinderConfig> config_cuda(1, &m_config, 1);
  
  ///// For bottom space points
  isBottom_cpu = true;
  isBottom_cuda.CopyH2D(&isBottom_cpu,1);	

  CUDA::Buffer<int>   isCompatBottomSP_cuda(nBottom*nMiddle);
  auto isCompatBottomMat_cpu  = CPU::Matrix<int>(nBottom, nMiddle);
  
  offset=0;
  while(offset<nBottom){
    BlockSize = fmin(MAX_BLOCK_SIZE,nBottom);
    BlockSize = fmin(BlockSize,nBottom-offset);
    DS_BlockSize = dim3(BlockSize,1,1);    
    CUDA::Buffer<float> rB_cuda(BlockSize, spBmat_cpu.GetEl(offset,3), BlockSize);    
    CUDA::Buffer<float> zB_cuda(BlockSize, spBmat_cpu.GetEl(offset,2), BlockSize);  

    SeedfinderCUDAKernels::searchDoublet( DS_GridSize, DS_BlockSize, 
					  isBottom_cuda.Get(),
					  rB_cuda.Get(), zB_cuda.Get(), 
					  rM_cuda.Get(), zM_cuda.Get(), 
					  //deltaRMin_cuda.Get(), deltaRMax_cuda.Get(), 
					  //cotThetaMax_cuda.Get(),
					  //collisionRegionMin_cuda.Get(),collisionRegionMax_cuda.Get(),
					  config_cuda.Get(),
					  isCompatBottomSP_cuda.Get(offset*nMiddle));
    offset+=BlockSize;
  }
  // Rearrange the doublet 
  // Prev: [mid1: bot_1, ..., bot_N]    [mid2: bot_1, ..., bot_N] ...    [midN: bot_1, ..., bot_N]
  //       [mid1: bot_N+1, ..., bot_2N] [mid2: bot_N+1, ..., bot_2N] ... [midN: bot_N+1, ..., bot_2N]
  //       ...
  //
  // New : [mid1: bot_1, ..., bot_TN] [mid2: bot_1, ..., bot_TN] ... [midN: bot_1, ..., bot_TN]
  auto bottomBuffer = std::shared_ptr<int>(isCompatBottomSP_cuda.GetHostBuffer(nBottom*nMiddle));
  offset=0;
  while(offset<nBottom){
    BlockSize = fmin(MAX_BLOCK_SIZE,nBottom);
    BlockSize = fmin(BlockSize,nBottom-offset);
    for (int i_m=0; i_m<nMiddle; i_m++){     
      std::copy(bottomBuffer.get()+offset*nMiddle+i_m*BlockSize,
		bottomBuffer.get()+offset*nMiddle+(i_m+1)*BlockSize,
		isCompatBottomMat_cpu.GetEl(offset,i_m));
    }
    offset+= BlockSize;
  }
  
  ///// For top space points
  isBottom_cpu = false;
  isBottom_cuda.CopyH2D(&isBottom_cpu,1);	
  CUDA::Buffer<int>   isCompatTopSP_cuda(nTop*nMiddle);
  auto isCompatTopMat_cpu = CPU::Matrix<int>(nTop, nMiddle);
  
  offset=0;
  while(offset<nTop){
    BlockSize = fmin(MAX_BLOCK_SIZE,nTop);
    BlockSize = fmin(BlockSize,nTop-offset);
    DS_BlockSize = dim3(BlockSize,1,1);    
    CUDA::Buffer<float> rT_cuda(BlockSize, spTmat_cpu.GetEl(offset,3), BlockSize);    
    CUDA::Buffer<float> zT_cuda(BlockSize, spTmat_cpu.GetEl(offset,2), BlockSize);  

    SeedfinderCUDAKernels::searchDoublet( DS_GridSize, DS_BlockSize, 
					  isBottom_cuda.Get(),
					  rT_cuda.Get(), zT_cuda.Get(), 
					  rM_cuda.Get(), zM_cuda.Get(), 
					  //deltaRMin_cuda.Get(), deltaRMax_cuda.Get(), 
					  //cotThetaMax_cuda.Get(),
					  //collisionRegionMin_cuda.Get(),collisionRegionMax_cuda.Get(),
					  config_cuda.Get(),
					  isCompatTopSP_cuda.Get(offset*nMiddle) );
    offset+= BlockSize;
  }
  // Rearrange the doublet 
  // Prev: [mid1: top_1, ..., top_N]    [mid2: top_1, ..., top_N] ...    [midN: top_1, ..., top_N]
  //       [mid1: top_N+1, ..., top_2N] [mid2: top_N+1, ..., top_2N] ... [midN: top_N+1, ..., top_2N]
  //       ...
  //
  // New : [mid1: top_1, ..., top_TN] [mid2: top_1, ..., top_TN] ... [midN: top_1, ..., top_TN]
  auto topBuffer = std::shared_ptr<int>(isCompatTopSP_cuda.GetHostBuffer(nTop*nMiddle));
  offset=0;
  while(offset<nTop){
    BlockSize = fmin(MAX_BLOCK_SIZE,nTop);
    BlockSize = fmin(BlockSize,nTop-offset);
    for (int i_m=0; i_m<nMiddle; i_m++){
      std::copy(topBuffer.get()+offset*nMiddle+i_m*BlockSize,
		topBuffer.get()+offset*nMiddle+(i_m+1)*BlockSize,
		isCompatTopMat_cpu.GetEl(offset,i_m));
    }
    offset+= BlockSize;
  }
  
  for (int i_m=0; i_m<nMiddle; i_m++){
    
    std::vector< int > bIndex;
    for (int i=0; i<nBottom; i++){
      if (*isCompatBottomMat_cpu.GetEl(i,i_m)) bIndex.push_back(i);
    }
    if (bIndex.empty()) continue;
    
    std::vector< int > tIndex;
    for (int i=0; i<nTop; i++){
      if (*isCompatTopMat_cpu.GetEl(i,i_m)) tIndex.push_back(i);
    }
    if (tIndex.empty()) continue;

    std::cout<< "CUDA Compatible Hits: " << bIndex.size() << "  " << tIndex.size() << std::endl;

    /* -----------------------------------------
       Algorithm 2. Transform Coordinates (TC)
     -------------------------------------------*/
    
    int nSpB = bIndex.size();
    int nSpT = tIndex.size();
    dim3 TC_GridSize;
    dim3 TC_BlockSize(WARP_SIZE);
    CUDA::Buffer<float> spM_cuda(6,spMmat_cpu.GetRow(i_m), 6); // input
    
    // bottom
    TC_GridSize = dim3(int(nSpB/TC_BlockSize.x)+1,1,1);
    isBottom_cpu = true;
    isBottom_cuda.CopyH2D(&isBottom_cpu,1);	
    
    CUDA::Buffer<int>   nSpB_cuda(1, &nSpB, 1);    // input
    CPU::Matrix<float> spBcompMat_cpu(nSpB,6);
    for (int i=0; i<bIndex.size(); i++){
      int i_b = bIndex[i];
      spBcompMat_cpu.SetRow(i,spBmat_cpu.GetRow(i_b));
    }
    CUDA::Matrix<float> spBcompMat_cuda(nSpB,6, &spBcompMat_cpu); // input    
    CUDA::Matrix<float> circBcompMat_cuda(nSpB,6);                // output

    SeedfinderCUDAKernels::transformCoordinates(TC_GridSize, TC_BlockSize,
						isBottom_cuda.Get(),
						spM_cuda.Get(),
						nSpB_cuda.Get(),
						spBcompMat_cuda.GetEl(0,0),
						circBcompMat_cuda.GetEl(0,0));

    // top
    TC_GridSize = dim3(int(nSpT/TC_BlockSize.x)+1,1,1);
    isBottom_cpu = false;
    isBottom_cuda.CopyH2D(&isBottom_cpu,1);	
    
    CUDA::Buffer<int>   nSpT_cuda(1, &nSpT, 1);    // input
    CPU::Matrix<float> spTcompMat_cpu(nSpT,6);
    for (int i=0; i<tIndex.size(); i++){
      int i_t = tIndex[i];
      spTcompMat_cpu.SetRow(i,spTmat_cpu.GetRow(i_t));
    }    
    CUDA::Matrix<float> spTcompMat_cuda(nSpT,6, &spTcompMat_cpu); // input    
    CUDA::Matrix<float> circTcompMat_cuda(nSpT,6);                // output

    SeedfinderCUDAKernels::transformCoordinates(TC_GridSize, TC_BlockSize,
						isBottom_cuda.Get(),
						spM_cuda.Get(),
						nSpT_cuda.Get(),
						spTcompMat_cuda.GetEl(0,0),
						circTcompMat_cuda.GetEl(0,0));

    /* -----------------------------------
       Algorithm 3. Triplet Search (TS)
     -------------------------------------*/
    
    dim3 TS_GridSize(nSpB,1,1);
    dim3 TS_BlockSize(nSpT,1,1);
    //CUDA::Buffer<float> maxScatteringAngle2_cuda(1, &m_config.maxScatteringAngle2,1);
    //CUDA::Buffer<float> sigmaScattering_cuda(1, &m_config.sigmaScattering,1);
    //CUDA::Buffer<float> minHelixDiameter2_cuda(1, &m_config.minHelixDiameter2,1);
    //CUDA::Buffer<float> pT2perRadius_cuda(1, &m_config.pT2perRadius,1);
    //CUDA::Buffer<float> impactMax_cuda(1, &m_config.impactMax,1);
    //auto sf_config = (m_config.seedFilter).m_cfg;    
    //CUDA::Buffer<float> deltaInvHelixDiameter_cuda(1, &sf_config.deltaInvHelixDiameter,1);
    //CUDA::Buffer<float> impactWeightFactor_cuda(1, &sf_config.impactWeightFactor,1);
    //CUDA::Buffer<float> sf_deltaRMin_cuda(1, &sf_config.deltaRMin,1);
    //CUDA::Buffer<float> compatSeedWeight_cuda(1, &sf_config.compatSeedWeight,1);
    //CUDA::Buffer<size_t> compatSeedLimit_cuda(1, &sf_config.compatSeedLimit,1);    
    
    SeedfinderCUDAKernels::searchTriplet(TS_GridSize, TS_BlockSize,
					 spM_cuda.Get(),
					 nSpB_cuda.Get(), spBcompMat_cuda.GetEl(0,0),
					 nSpT_cuda.Get(), spTcompMat_cuda.GetEl(0,0),
					 circBcompMat_cuda.GetEl(0,0),
					 circTcompMat_cuda.GetEl(0,0),
					 config_cuda.Get()
					 // seed finder config
					 //maxScatteringAngle2_cuda.Get(),
					 //sigmaScattering_cuda.Get(),
					 //minHelixDiameter2_cuda.Get(),
					 //pT2perRadius_cuda.Get(),
					 //impactMax_cuda.Get(),
					 // seed filter config
					 //deltaInvHelixDiameter_cuda.Get(),
					 //impactWeightFactor_cuda.Get(),
					 //sf_deltaRMin_cuda.Get(),
					 //compatSeedWeight_cuda.Get(),
					 //compatSeedLimit_cuda.Get(),
					 );
    

    /*
    float* Zo        = circBcompMat_cuda.GetHostBuffer(nSpB,0,0);
    float* cot_theta = circBcompMat_cuda.GetHostBuffer(nSpB,0,1);
    float* iDeltaR   = circBcompMat_cuda.GetHostBuffer(nSpB,0,2);
    float* Er        = circBcompMat_cuda.GetHostBuffer(nSpB,0,3);    
    float* U         = circBcompMat_cuda.GetHostBuffer(nSpB,0,4);
    float* V         = circBcompMat_cuda.GetHostBuffer(nSpB,0,5);
    
    for (int i_b=0; i_b<nSpB; i_b++){
      std::cout << i_b << "  " << Zo[i_b] << "  " << cot_theta[i_b] << "  " << iDeltaR[i_b] << "  " << Er[i_b] << "  "  << U[i_b] << "  " << V[i_b] << std::endl;
    }
    */
    
  } 
  
  return outputVec;
  
  }

}// namespace Acts

