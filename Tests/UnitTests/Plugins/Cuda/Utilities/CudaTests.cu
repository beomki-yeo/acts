// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/Cuda/Cuda.hpp"
#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <cuda_profiler_api.h>

namespace Acts{
namespace Test{

template<typename AFloat, int row, int col>
__global__ void MatrixLoadStore1(const Eigen::Matrix<AFloat,row,col>* input,
				Eigen::Matrix<AFloat,row,col>* output){
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  output[globalId] = input[globalId];   
}

template<typename AFloat, int row, int col>
__global__ void MatrixLoadStore2(const Eigen::Matrix<AFloat,row,col>* input,
				Eigen::Matrix<AFloat,row,col>* output){
  
  for (int i=0; i<col; i++){
    output[blockIdx.x](threadIdx.x,i) = input[blockIdx.x](threadIdx.x,i);
  }
}

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_CASE( CUDAOBJ_TEST ){

  // Profiler
  cudaProfilerStart();
  
  using AFloat = float;
  
  const int vecDim = 16;  // Vector Dimension
  const int nVec   = 128; // Number of vectors
    
  dim3 gridSize;
  dim3 blockSize;
  int  bufSize;

  // Case 1) For misaligned memory access
  // global memory load/store efficiency is 4Bytes(float size)/32Bytes(cache line) ~ 12.5%
  gridSize  = dim3(1,1,1);
  blockSize = dim3(nVec,1,1);
  bufSize   = gridSize.x * blockSize.x;
  Eigen::Matrix<AFloat, vecDim, 1>  inMat1_cpu[bufSize];
  for (int i=0; i< bufSize; i++){
    inMat1_cpu[i] = Eigen::Matrix<AFloat,vecDim,1>::Random();
  }

  CudaVector<Eigen::Matrix<AFloat, vecDim, 1>> inMat1_cuda(bufSize, inMat1_cpu, bufSize, 0);
  CudaVector<Eigen::Matrix<AFloat, vecDim, 1>> outMat1_cuda(bufSize);
 
  MatrixLoadStore1<AFloat, vecDim, 1><<< gridSize, blockSize >>>(inMat1_cuda.Get(),outMat1_cuda.Get());
  cudaErrChk ( cudaGetLastError() );
    
  Eigen::Matrix<AFloat, vecDim, 1>* outMat1_cpu = outMat1_cuda.GetHost();

  // Case 2) For aligned memory access
  // global memory load/store efficiency ~ 100%
  gridSize  = dim3(1,1,1);
  blockSize = dim3(vecDim,1,1);
  bufSize   = gridSize.x * blockSize.x;

  Eigen::Matrix<AFloat, vecDim, nVec>  inMat2_cpu[bufSize];
  for (int i=0; i< bufSize; i++){
    inMat2_cpu[i] = Eigen::Matrix<AFloat,vecDim,nVec>::Random();
  }

  CudaVector<Eigen::Matrix<AFloat,vecDim,nVec>> inMat2_cuda(bufSize, inMat2_cpu, bufSize, 0);
  CudaVector<Eigen::Matrix<AFloat,vecDim,nVec>> outMat2_cuda(bufSize);

  MatrixLoadStore2<AFloat, vecDim, nVec><<< gridSize, blockSize >>>(inMat2_cuda.Get(),outMat2_cuda.Get());

  cudaErrChk ( cudaGetLastError() );
    
  Eigen::Matrix<AFloat,vecDim,nVec>* outMat2_cpu = outMat2_cuda.GetHost();
  
  cudaProfilerStop();

  for (int i=0; i<nVec; i++){
    BOOST_REQUIRE( inMat1_cpu[i] == outMat1_cpu[i] );
  }
  BOOST_REQUIRE( inMat2_cpu[0] == outMat2_cpu[0] ); 
}
BOOST_AUTO_TEST_SUITE_END()

}
}
