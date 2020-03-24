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

  // Get Size of spacepoints
  int nMiddle = 0;
  int nBottom = 0;
  int nTop    = 0;

  for (auto sp: middleSPs) nMiddle++;
  for (auto sp: bottomSPs) nBottom++;
  for (auto sp: topSPs)    nTop++;

  if (nMiddle == 0 || nBottom == 0 || nTop == 0) return outputVec;
  
  // Define Matrix and Do flattening
  CPU::Matrix<float> spMmat_cpu(6,nMiddle); // x y z r varR varZ
  CPU::Matrix<float> spBmat_cpu(6,nBottom);
  CPU::Matrix<float> spTmat_cpu(6,nTop);
  //CPU::Matrix<float> spMmat_cpu(nMiddle, 6); // x y z r varR varZ
  //CPU::Matrix<float> spBmat_cpu(nBottom, 6);
  //CPU::Matrix<float> spTmat_cpu(nTop   , 6);
  
  
  int i_m=0;
  for (auto sp: middleSPs){
    //spMmat_cpu.Set(0,i_m,sp->x());
    //spMmat_cpu.Set(1,i_m,sp->y());
    //spMmat_cpu.Set(2,i_m,sp->z());
    //spMmat_cpu.Set(3,i_m,sp->radius());
    //spMmat_cpu.Set(4,i_m,sp->varianceR());
    //spMmat_cpu.Set(5,i_m,sp->varianceZ());

   
    spMmat_cpu.Set(0,i_m,sp->x());
    spMmat_cpu.Set(1,i_m,sp->y());
    spMmat_cpu.Set(2,i_m,sp->z());
    spMmat_cpu.Set(3,i_m,sp->radius());
    spMmat_cpu.Set(4,i_m,sp->varianceR());
    spMmat_cpu.Set(5,i_m,sp->varianceZ());
    i_m++;
  }

  int i_b=0;
  for (auto sp: bottomSPs){
    spBmat_cpu.Set(0,i_b,sp->x());
    spBmat_cpu.Set(1,i_b,sp->y());
    spBmat_cpu.Set(2,i_b,sp->z());
    spBmat_cpu.Set(3,i_b,sp->radius());
    spBmat_cpu.Set(4,i_b,sp->varianceR());
    spBmat_cpu.Set(5,i_b,sp->varianceZ());
    i_b++;
  }

  int i_t=0;
  for (auto sp: topSPs){
    spTmat_cpu.Set(0,i_t,sp->x());
    spTmat_cpu.Set(1,i_t,sp->y());
    spTmat_cpu.Set(2,i_t,sp->z());
    spTmat_cpu.Set(3,i_t,sp->radius());
    spTmat_cpu.Set(4,i_t,sp->varianceR());
    spTmat_cpu.Set(5,i_t,sp->varianceZ());
    i_t++;    
  }
  /*
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
  */
  /* ------------------------------------
     Algorithm 1. Doublet Search (DS)
  ---------------------------------------*/
  
  int  offset;
  int  BlockSize;
  dim3 DS_BlockSize;
  dim3 DS_GridSize(nMiddle,1,1);
  
  CUDA::Buffer<float> deltaRMin_cuda(1,          &m_config.deltaRMin);
  CUDA::Buffer<float> deltaRMax_cuda(1,          &m_config.deltaRMax);
  CUDA::Buffer<float> cotThetaMax_cuda(1,        &m_config.cotThetaMax);
  CUDA::Buffer<float> collisionRegionMin_cuda(1, &m_config.collisionRegionMin);
  CUDA::Buffer<float> collisionRegionMax_cuda(1, &m_config.collisionRegionMax);  
  CUDA::Buffer<float> rM_cuda(nMiddle, spMmat_cpu.Get(3,0));
  CUDA::Buffer<float> zM_cuda(nMiddle, spMmat_cpu.Get(2,0));

  
  ///// For bottom space points
  isBottom_cpu = true;
  isBottom_cuda.SetData(&isBottom_cpu,1);	

  CUDA::Buffer<int>   isCompatBottomSP_cuda(nBottom*nMiddle);
  auto isCompatBottomMat_cpu  = CPU::Matrix<int>(nMiddle, nBottom);
  
  offset=0;
  while(offset<nBottom){
    BlockSize = fmin(MAX_BLOCK_SIZE,nBottom);
    BlockSize = fmin(BlockSize,nBottom-offset);
    DS_BlockSize = dim3(BlockSize,1,1);    
    CUDA::Buffer<float> rB_cuda(BlockSize, spBmat_cpu.Get(3,offset));    
    CUDA::Buffer<float> zB_cuda(BlockSize, spBmat_cpu.Get(2,offset));  
    SeedfinderCUDAKernels::SearchDoublet( DS_GridSize, DS_BlockSize, 
					  isBottom_cuda.data(),
					  rB_cuda.data(), zB_cuda.data(), 
					  rM_cuda.data(), zM_cuda.data(), 
					  deltaRMin_cuda.data(), deltaRMax_cuda.data(), 
					  cotThetaMax_cuda.data(),
					  collisionRegionMin_cuda.data(),collisionRegionMax_cuda.data(),
					  isCompatBottomSP_cuda.data(offset*nMiddle) );
    offset+=BlockSize;
  }
  // Rearrange the doublet (Note: It spents time more than CUDA kernel...)
  // Prev: [mid1: bot_1, ..., bot_N]    [mid2: bot_1, ..., bot_N] ...    [midN: bot_1, ..., bot_N]
  //       [mid1: bot_N+1, ..., bot_2N] [mid2: bot_N+1, ..., bot_2N] ... [midN: bot_N+1, ..., bot_2N]
  //       ...
  //
  // New : [mid1: bot_1, ..., bot_TN] [mid2: bot_1, ..., bot_TN] ... [midN: bot_1, ..., bot_TN]
  auto bottomBuffer = CPU::Buffer<int>(nBottom*nMiddle,
				       isCompatBottomSP_cuda.dataHost(nBottom*nMiddle));
  offset=0;
  while(offset<nBottom){
    BlockSize = fmin(MAX_BLOCK_SIZE,nBottom);
    BlockSize = fmin(BlockSize,nBottom-offset);
    for (int i_m=0; i_m<nMiddle; i_m++){     
      std::copy(bottomBuffer.data()+offset*nMiddle+i_m*BlockSize,
		bottomBuffer.data()+offset*nMiddle+(i_m+1)*BlockSize,
		isCompatBottomMat_cpu.Get(i_m,offset));
    }
    offset+= BlockSize;
  }
  
  ///// For top space points
  isBottom_cpu = false;
  isBottom_cuda.SetData(&isBottom_cpu,1);	
  CUDA::Buffer<int>   isCompatTopSP_cuda(nTop*nMiddle);
  auto isCompatTopMat_cpu = CPU::Matrix<int>(nMiddle, nTop);
  
  offset=0;
  while(offset<nTop){
    BlockSize = fmin(MAX_BLOCK_SIZE,nTop);
    BlockSize = fmin(BlockSize,nTop-offset);
    DS_BlockSize = dim3(BlockSize,1,1);    
    CUDA::Buffer<float> rT_cuda(BlockSize, spTmat_cpu.Get(3,offset));    
    CUDA::Buffer<float> zT_cuda(BlockSize, spTmat_cpu.Get(2,offset));  
    SeedfinderCUDAKernels::SearchDoublet( DS_GridSize, DS_BlockSize, 
					  isBottom_cuda.data(),
					  rT_cuda.data(), zT_cuda.data(), 
					  rM_cuda.data(), zM_cuda.data(), 
					  deltaRMin_cuda.data(), deltaRMax_cuda.data(), 
					  cotThetaMax_cuda.data(),
					  collisionRegionMin_cuda.data(),collisionRegionMax_cuda.data(),
					  isCompatTopSP_cuda.data(offset*nMiddle) );
    offset+= BlockSize;
  }
  // Rearrange the doublet (Note: It spents time more than CUDA kernel...)
  // Prev: [mid1: top_1, ..., top_N]    [mid2: top_1, ..., top_N] ...    [midN: top_1, ..., top_N]
  //       [mid1: top_N+1, ..., top_2N] [mid2: top_N+1, ..., top_2N] ... [midN: top_N+1, ..., top_2N]
  //       ...
  //
  // New : [mid1: top_1, ..., top_TN] [mid2: top_1, ..., top_TN] ... [midN: top_1, ..., top_TN]
  auto topBuffer = CPU::Buffer<int>(nTop*nMiddle,
				    isCompatTopSP_cuda.dataHost(nTop*nMiddle));
  offset=0;
  while(offset<nTop){
    BlockSize = fmin(MAX_BLOCK_SIZE,nTop);
    BlockSize = fmin(BlockSize,nTop-offset);
    for (int i_m=0; i_m<nMiddle; i_m++){
      std::copy(topBuffer.data()+offset*nMiddle+i_m*BlockSize,
		topBuffer.data()+offset*nMiddle+(i_m+1)*BlockSize,
		isCompatTopMat_cpu.Get(i_m,offset));
    }
    offset+= BlockSize;
  }
  
  /* ----------------------------------------
     Algorithm 2. Transform coordinate (TC)
  -------------------------------------------*/
  
  std::vector< int > middleIndex;
  std::vector< std::vector< int > > compatBottomIndex;
  std::vector< std::vector< int > > compatTopIndex;
    
  for (int i_m=0; i_m<nMiddle; i_m++){
    // Bottom
    auto isCompatBottom = CPU::Buffer<int>(isCompatBottomMat_cpu.GetNCols(),
					   isCompatBottomMat_cpu.GetRow(i_m));
    std::vector< int > bIndex;
    for (int i=0; i<isCompatBottomMat_cpu.GetNCols(); i++){
      if (isCompatBottom[i]) bIndex.push_back(i);
    }
    if (bIndex.empty()) continue;

    // Top
    auto isCompatTop = CPU::Buffer<int>(isCompatTopMat_cpu.GetNCols(),
					isCompatTopMat_cpu.GetRow(i_m));
    std::vector< int > tIndex;
    for (int i=0; i<isCompatTopMat_cpu.GetNCols(); i++){
      if (isCompatTop[i]) tIndex.push_back(i);
    }
    if (tIndex.empty()) continue;

    middleIndex.push_back(i_m);
    compatBottomIndex.push_back(bIndex);
    compatTopIndex.push_back(tIndex);

    std::cout<< "CUDA Compatible Hits: " << bIndex.size() << "  " << tIndex.size() << std::endl;
  }
  
  /*
  CPU::Matrix<float> ixyzrM_cpu(middleIndex.size(),5);
  std::vector< CPU::Matrix<float> > ixyzrB_cpu;
  std::vector< CPU::Matrix<float> > ixyzrT_cpu;

  for (int i_m=0; i_m<middleIndex.size(); i_m++){
    auto spM middleIndex[i_m];
    
    ixyzrM_cpu.SetColumn(i_m, )
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
  /*
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
  */
  return outputVec;
  
  }  // namespace Acts
}
