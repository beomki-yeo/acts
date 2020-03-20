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
#include <Acts/Seeding/SeedfinderCUDAKernels.cuh>

#define WARP_SIZE 64

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

  template< typename external_spacepoint_t, typename architecture_t>
  template< typename T, typename sp_range_t>
  typename std::enable_if< std::is_same<T, Acts::CPU>::value, std::vector<Seed<external_spacepoint_t> > >::type
  Seedfinder<external_spacepoint_t, architecture_t>::createSeedsForGroup(
    sp_range_t bottomSPs, sp_range_t middleSPs, sp_range_t topSPs) const {
  std::vector<Seed<external_spacepoint_t>> outputVec;

  double doublet_time = 0;
  double triplet_time = 0;

  int i_middleSP = 0;

  for (auto spM : middleSPs) {
   
    float rM = spM->radius();
    float zM = spM->z();
    float varianceRM = spM->varianceR();
    float varianceZM = spM->varianceZ();

    auto start_doublet = std::chrono::system_clock::now();

    // bottom space point
    std::vector<const InternalSpacePoint<external_spacepoint_t>*>
        compatBottomSP;

    for (auto bottomSP : bottomSPs) {
      float rB = bottomSP->radius();
      float deltaR = rM - rB;
      // if r-distance is too big, try next SP in bin
      if (deltaR > m_config.deltaRMax) {
        continue;
      }
      // if r-distance is too small, break because bins are NOT r-sorted
      if (deltaR < m_config.deltaRMin) {
        continue;
      }
      // ratio Z/R (forward angle) of space point duplet
      float cotTheta = (zM - bottomSP->z()) / deltaR;
      if (std::fabs(cotTheta) > m_config.cotThetaMax) {
        continue;
      }
      // check if duplet origin on z axis within collision region
      float zOrigin = zM - rM * cotTheta;
      if (zOrigin < m_config.collisionRegionMin ||
          zOrigin > m_config.collisionRegionMax) {
        continue;
      }
      compatBottomSP.push_back(bottomSP);
    }
    // no bottom SP found -> try next spM
    if (compatBottomSP.empty()) {
      continue;
    }

    std::vector<const InternalSpacePoint<external_spacepoint_t>*> compatTopSP;

    for (auto topSP : topSPs) {
      float rT = topSP->radius();
      float deltaR = rT - rM;
      // this condition is the opposite of the condition for bottom SP
      if (deltaR < m_config.deltaRMin) {
        continue;
      }
      if (deltaR > m_config.deltaRMax) {
        break;
      }

      float cotTheta = (topSP->z() - zM) / deltaR;
      if (std::fabs(cotTheta) > m_config.cotThetaMax) {
        continue;
      }
      float zOrigin = zM - rM * cotTheta;
      if (zOrigin < m_config.collisionRegionMin ||
          zOrigin > m_config.collisionRegionMax) {
        continue;
      }
      compatTopSP.push_back(topSP);
    }
    if (compatTopSP.empty()) {
      continue;
    }
  
    std::cout << i_middleSP << "  CPU  Compatible Bot: " << compatBottomSP.size() << "  Top: " << compatTopSP.size() << std::endl;

    ////////////////////////////////////////////////////////////
    // Disable other parts temporariliy
    i_middleSP++;
    if (i_middleSP == m_config.nMiddleSPsToIterate) break;

    continue;
    ////////////////////////////////////////////////////////////
    auto end_doublet = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds_doublet = end_doublet - start_doublet;
    doublet_time+=elapsed_seconds_doublet.count();

    auto start_triplet = std::chrono::system_clock::now();

    // contains parameters required to calculate circle with linear equation
    // ...for bottom-middle
    std::vector<LinCircle> linCircleBottom;
    // ...for middle-top
    std::vector<LinCircle> linCircleTop;
    transformCoordinates(compatBottomSP, *spM, true, linCircleBottom);
    transformCoordinates(compatTopSP, *spM, false, linCircleTop);

    // create vectors here to avoid reallocation in each loop
    std::vector<const InternalSpacePoint<external_spacepoint_t>*> topSpVec;
    std::vector<float> curvatures;
    std::vector<float> impactParameters;

    std::vector<std::pair<
        float, std::unique_ptr<const InternalSeed<external_spacepoint_t>>>>
        seedsPerSpM;
    size_t numBotSP = compatBottomSP.size();
    size_t numTopSP = compatTopSP.size();

    for (size_t b = 0; b < numBotSP; b++) {

      auto lb = linCircleBottom[b];
      float Zob = lb.Zo;
      float cotThetaB = lb.cotTheta;
      float Vb = lb.V;
      float Ub = lb.U;
      float ErB = lb.Er;
      float iDeltaRB = lb.iDeltaR;

      // 1+(cot^2(theta)) = 1/sin^2(theta)
      float iSinTheta2 = (1. + cotThetaB * cotThetaB);
      // calculate max scattering for min momentum at the seed's theta angle
      // scaling scatteringAngle^2 by sin^2(theta) to convert pT^2 to p^2
      // accurate would be taking 1/atan(thetaBottom)-1/atan(thetaTop) <
      // scattering
      // but to avoid trig functions we approximate cot by scaling by
      // 1/sin^4(theta)
      // resolving with pT to p scaling --> only divide by sin^2(theta)
      // max approximation error for allowed scattering angles of 0.04 rad at
      // eta=infinity: ~8.5%
      float scatteringInRegion2 = m_config.maxScatteringAngle2 * iSinTheta2;
      // multiply the squared sigma onto the squared scattering
      scatteringInRegion2 *=
          m_config.sigmaScattering * m_config.sigmaScattering;

      // clear all vectors used in each inner for loop
      topSpVec.clear();
      curvatures.clear();
      impactParameters.clear();
      for (size_t t = 0; t < numTopSP; t++) {
        auto lt = linCircleTop[t];

        // add errors of spB-spM and spM-spT pairs and add the correlation term
        // for errors on spM
        float error2 = lt.Er + ErB +
                       2 * (cotThetaB * lt.cotTheta * varianceRM + varianceZM) *
                           iDeltaRB * lt.iDeltaR;

        float deltaCotTheta = cotThetaB - lt.cotTheta;
        float deltaCotTheta2 = deltaCotTheta * deltaCotTheta;
        float error;
        float dCotThetaMinusError2;
        // if the error is larger than the difference in theta, no need to
        // compare with scattering
        if (deltaCotTheta2 - error2 > 0) {
          deltaCotTheta = std::abs(deltaCotTheta);
          // if deltaTheta larger than the scattering for the lower pT cut, skip
          error = std::sqrt(error2);
          dCotThetaMinusError2 =
              deltaCotTheta2 + error2 - 2 * deltaCotTheta * error;
          // avoid taking root of scatteringInRegion
          // if left side of ">" is positive, both sides of unequality can be
          // squared
          // (scattering is always positive)

          if (dCotThetaMinusError2 > scatteringInRegion2) {
            continue;
          }
        }

        // protects against division by 0
        float dU = lt.U - Ub;
        if (dU == 0.) {
          continue;
        }
        // A and B are evaluated as a function of the circumference parameters
        // x_0 and y_0
        float A = (lt.V - Vb) / dU;
        float S2 = 1. + A * A;
        float B = Vb - A * Ub;
        float B2 = B * B;
        // sqrt(S2)/B = 2 * helixradius
        // calculated radius must not be smaller than minimum radius
        if (S2 < B2 * m_config.minHelixDiameter2) {
          continue;
        }
        // 1/helixradius: (B/sqrt(S2))/2 (we leave everything squared)
        float iHelixDiameter2 = B2 / S2;
        // calculate scattering for p(T) calculated from seed curvature
        float pT2scatter = 4 * iHelixDiameter2 * m_config.pT2perRadius;
        // TODO: include upper pT limit for scatter calc
        // convert p(T) to p scaling by sin^2(theta) AND scale by 1/sin^4(theta)
        // from rad to deltaCotTheta
        float p2scatter = pT2scatter * iSinTheta2;
        // if deltaTheta larger than allowed scattering for calculated pT, skip
        if ((deltaCotTheta2 - error2 > 0) &&
            (dCotThetaMinusError2 >
             p2scatter * m_config.sigmaScattering * m_config.sigmaScattering)) {
          continue;
        }
        // A and B allow calculation of impact params in U/V plane with linear
        // function
        // (in contrast to having to solve a quadratic function in x/y plane)
        float Im = std::abs((A - B * rM) * rM);

        if (Im <= m_config.impactMax) {
          topSpVec.push_back(compatTopSP[t]);
          // inverse diameter is signed depending if the curvature is
          // positive/negative in phi
          curvatures.push_back(B / std::sqrt(S2));
          impactParameters.push_back(Im);
        }
      }

      if (!topSpVec.empty()) {
        std::vector<std::pair<
            float, std::unique_ptr<const InternalSeed<external_spacepoint_t>>>>
            sameTrackSeeds;
        sameTrackSeeds = std::move(m_config.seedFilter->filterSeeds_2SpFixed(
            *compatBottomSP[b], *spM, topSpVec, curvatures, impactParameters,
            Zob));
        seedsPerSpM.insert(seedsPerSpM.end(),
                           std::make_move_iterator(sameTrackSeeds.begin()),
                           std::make_move_iterator(sameTrackSeeds.end()));
      }
    }

    auto end_triplet = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds_triplet = end_triplet - start_triplet;
    triplet_time+=elapsed_seconds_triplet.count();

    //std::cout << elapsed_seconds_doublet.count() << "  " << elapsed_seconds_triplet.count() << "  " << std::endl;

    m_config.seedFilter->filterSeeds_1SpFixed(seedsPerSpM, outputVec);

  }

  //std::cout << "Doublet Time: " << doublet_time << "  Triplet Time: " << triplet_time << std::endl;

  return outputVec;
  }

  template< typename external_spacepoint_t, typename architecture_t>
  template< typename T, typename sp_range_t>
  typename std::enable_if< std::is_same<T, Acts::CUDA>::value, std::vector<Seed<external_spacepoint_t> > >::type
  Seedfinder<external_spacepoint_t, architecture_t>::createSeedsForGroup(
    sp_range_t bottomSPs, sp_range_t middleSPs, sp_range_t topSPs) const {
  std::vector<Seed<external_spacepoint_t>> outputVec;

  // ----- Algorithm 0. Matrix Flattening ----- //
  std::vector<float> rBvec;
  std::vector<float> zBvec;

  for (auto sp: bottomSPs){
    rBvec.push_back(sp->radius());
    zBvec.push_back(sp->z());
  }

  std::vector<float> rTvec;
  std::vector<float> zTvec;

  for (auto sp: topSPs){
    rTvec.push_back(sp->radius());
    zTvec.push_back(sp->z());
  }

  // ----- Algorithm 1. Doublet Search (DS) ----- //

  // ----- GPU configuration setup ----- //
  dim3 DSBlockSize(WARP_SIZE*10,1,1); // 640
  dim3 DSGridSize(128,1,1);  
  //const int nHitsPerStream=1;
  //dim3 DSBlockSize(1,1,1); // 128
  //dim3 DSGridSize(1,1,1);  
  const int nHitsPerStream = DSBlockSize.x*DSGridSize.x; // 8192
  std::vector<cudaStream_t*> streams;

  int i_middleSP = 0;

  for (auto sp: middleSPs){

    float rM = sp->radius();
    float zM = sp->z();

    int offset;
    
    // ----- BOTTOM DOUBLET SEARCH ----- //    

    CUDA::Buffer<float> rB_cuda(rBvec.size());
    CUDA::Buffer<float> zB_cuda(zBvec.size());
    std::vector<int>    isBotCompat(rBvec.size(), true);
    CUDA::Buffer<int>   isBotCompat_cuda(rBvec.size());

    offset =0;
    while (true){
      int isBottom = 1;

      //cudaStream_t aStream;
      //cudaStreamCreate(&aStream);
      //streams.push_back(&aStream);

      int len = fmin(nHitsPerStream, rBvec.size()-offset);

      rB_cuda.SetData(&rBvec[offset],len,offset);
      zB_cuda.SetData(&zBvec[offset],len,offset);
      isBotCompat_cuda.SetData(&isBotCompat[offset],len,offset);
      
      CUDA::Buffer<float> rM_cuda(1);
      CUDA::Buffer<float> zM_cuda(1);
      CUDA::Buffer<int>   isBottom_cuda(1);
      CUDA::Buffer<float> deltaRMin_cuda(1);
      CUDA::Buffer<float> deltaRMax_cuda(1);
      CUDA::Buffer<float> cotThetaMax_cuda(1);
      CUDA::Buffer<float> collisionRegionMin_cuda(1);
      CUDA::Buffer<float> collisionRegionMax_cuda(1);

      rM_cuda.SetData(&rM,1,0);
      zM_cuda.SetData(&zM,1,0);
      isBottom_cuda.SetData(&isBottom,1,0);
      deltaRMin_cuda.SetData(&m_config.deltaRMin,1,0);
      deltaRMax_cuda.SetData(&m_config.deltaRMax,1,0);
      cotThetaMax_cuda.SetData(&m_config.cotThetaMax,1,0);
      collisionRegionMin_cuda.SetData(&m_config.collisionRegionMin,1,0);
      collisionRegionMax_cuda.SetData(&m_config.collisionRegionMax,1,0);
      
      SeedfinderCUDAKernels::SearchDoublet( DSBlockSize, DSGridSize, //NULL,//&streams.back(),
					    rB_cuda.data(offset), zB_cuda.data(offset), 
					    rM_cuda.data(), zM_cuda.data(), 
					    isBottom_cuda.data(), 
					    deltaRMin_cuda.data(), deltaRMax_cuda.data(), 
					    cotThetaMax_cuda.data(),
					    collisionRegionMin_cuda.data(), collisionRegionMax_cuda.data(),
					    isBotCompat_cuda.data(offset) );     
      
      auto output = isBotCompat_cuda.dataHost(len,offset);
      
      std::copy(output,output+len,isBotCompat.begin()+offset);
      delete output;
      
      //if (offset>=rBvec.size()) break;
      //std::cout << offset << "  " << len << "  " << nHitsPerStream << std::endl;
      offset += len;
      if (len < nHitsPerStream) break;
    }

    int nBotCompat=0;
    for (int i=0; i< rBvec.size(); i++) {
      if (isBotCompat[i] == true) nBotCompat++;
    }
    if (nBotCompat == 0) continue;

    // ----- TOP DOUBLET SEARCH ----- //    

    CUDA::Buffer<float> rT_cuda(rTvec.size());
    CUDA::Buffer<float> zT_cuda(zTvec.size());
    std::vector<int>    isTopCompat(rTvec.size(), true);
    CUDA::Buffer<int>   isTopCompat_cuda(rTvec.size());

    offset =0;
    while (true){
      int isBottom = 0;

      //cudaStream_t aStream;
      //cudaStreamCreate(&aStream);
      //streams.push_back(&aStream);
      
      int len = fmin(nHitsPerStream, rTvec.size()-offset);

      rT_cuda.SetData(&rTvec[offset],len,offset);
      zT_cuda.SetData(&zTvec[offset],len,offset);
      isTopCompat_cuda.SetData(&isTopCompat[offset],len,offset);
      
      CUDA::Buffer<float> rM_cuda(1);
      CUDA::Buffer<float> zM_cuda(1);
      CUDA::Buffer<int>   isBottom_cuda(1);
      CUDA::Buffer<float> deltaRMin_cuda(1);
      CUDA::Buffer<float> deltaRMax_cuda(1);
      CUDA::Buffer<float> cotThetaMax_cuda(1);
      CUDA::Buffer<float> collisionRegionMin_cuda(1);
      CUDA::Buffer<float> collisionRegionMax_cuda(1);

      rM_cuda.SetData(&rM,1,0);
      zM_cuda.SetData(&zM,1,0);
      isBottom_cuda.SetData(&isBottom,1,0);
      deltaRMin_cuda.SetData(&m_config.deltaRMin,1,0);
      deltaRMax_cuda.SetData(&m_config.deltaRMax,1,0);
      cotThetaMax_cuda.SetData(&m_config.cotThetaMax,1,0);
      collisionRegionMin_cuda.SetData(&m_config.collisionRegionMin,1,0);
      collisionRegionMax_cuda.SetData(&m_config.collisionRegionMax,1,0);

      SeedfinderCUDAKernels::SearchDoublet( DSBlockSize, DSGridSize, //NULL,//&streams.back(),
					    rT_cuda.data(offset), zT_cuda.data(offset), 
					    rM_cuda.data(), zM_cuda.data(), 
					    isBottom_cuda.data(), 
					    deltaRMin_cuda.data(), deltaRMax_cuda.data(), 
					    cotThetaMax_cuda.data(),
					    collisionRegionMin_cuda.data(), collisionRegionMax_cuda.data(),
					    isTopCompat_cuda.data(offset) );     
      
      auto output = isTopCompat_cuda.dataHost(len,offset);      
      std::copy(output,output+len,isTopCompat.begin()+offset);
      delete output;

      //if (offset>=rTvec.size()) break;
      offset += len;
      if (len < nHitsPerStream) break;
    }

    int nTopCompat=0;    
    for (int i=0; i< rTvec.size(); i++) {
      if (isTopCompat[i] == true) nTopCompat++;
    }
    if (nTopCompat == 0) continue;

    if (nBotCompat && nTopCompat){   

      std::cout << i_middleSP << "  Cuda Compatible Bot: " << nBotCompat << "  Top: " << nTopCompat << std::endl;
    }

    /*
    std::vector<const InternalSpacePoint<external_spacepoint_t>*>
    compatBottomSP;

    for(int i=0; i< rBvec.size(); i++){
      if (isBotCompat[i] == true) compatBottomSP.push_back(bottomSPs.begin()+i);
    }

    std::vector<const InternalSpacePoint<external_spacepoint_t>*>
        compatTopSP;
    */
    i_middleSP++;
    if (i_middleSP == m_config.nMiddleSPsToIterate) break;

  }  
 
  return outputVec;
  }


 
template <typename external_spacepoint_t, typename architecture_t>
void Seedfinder<external_spacepoint_t, architecture_t>::transformCoordinates(
    std::vector<const InternalSpacePoint<external_spacepoint_t>*>& vec,
    const InternalSpacePoint<external_spacepoint_t>& spM, bool bottom,
    std::vector<LinCircle>& linCircleVec) const {
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
            (cot_theta * cot_theta) * (varianceRM + sp->varianceR())) *
           iDeltaR2;
    linCircleVec.push_back(l);
  }
}
}  // namespace Acts
