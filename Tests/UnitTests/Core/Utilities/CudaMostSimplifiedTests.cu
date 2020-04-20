// This file is part of the Acts project.
//
// Copyright (C) 2018-2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Utilities/Platforms/CUDA/CudaVector.cu"
#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <cuda_profiler_api.h>

template<typename AFloat, int row, int col>
__global__ void MatrixLoadStore(const Eigen::Matrix<AFloat,row,col>* input,
				Eigen::Matrix<AFloat,row,col>* output){
  
  for (int i=0; i<col; i++){
    output[blockIdx.x](threadIdx.x,i) = input[blockIdx.x](threadIdx.x,i);
  }
}

namespace Acts{
namespace Test{

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_CASE( CUDAOBJ_TEST ){

  const int vecDim = 100;  // Vector Dimension
  const int nVec   = 128; // Number of vectors
    
  dim3 gridSize(1,1,1);
  dim3 blockSize(vecDim,1,1);
  int  bufSize;

  bufSize   = gridSize.x * blockSize.x;
  Eigen::Matrix<float, vecDim, nVec>  iMat_cpu[bufSize];
  for (int i=0; i< bufSize; i++){
    iMat_cpu[i] = Eigen::Matrix<float,vecDim,nVec>::Random();
  }

  cudaProfilerStart();
  
  CudaVector<Eigen::Matrix<float,vecDim,nVec>> iMat_cuda(bufSize, iMat_cpu, bufSize, 0);
  CudaVector<Eigen::Matrix<float,vecDim,nVec>> oMat_cuda(bufSize);
  MatrixLoadStore<float, vecDim, nVec><<< gridSize, blockSize >>>(iMat_cuda.Get(),oMat_cuda.Get());
  Eigen::Matrix<float,vecDim,nVec>* oMat_cpu = oMat_cuda.GetHost();

  cudaProfilerStop();
  
  BOOST_REQUIRE( iMat_cpu[0] == oMat_cpu[0] ); 
}
BOOST_AUTO_TEST_SUITE_END()

}
}