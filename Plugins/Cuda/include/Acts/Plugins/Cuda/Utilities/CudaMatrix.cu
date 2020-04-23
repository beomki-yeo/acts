// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"
#include "Acts/Plugins/Cuda/Utilities/CpuMatrix.hpp"
#include "CudaUtils.cu"

namespace Acts{

template<typename Var_t>
class CudaMatrix{

public:

  CudaMatrix()=default;
  CudaMatrix(size_t nRows, size_t nCols){
    SetSize(nRows,nCols);
    ACTS_CUDA_ERROR_CHECK( cudaMalloc((Var_t**)&m_devPtr, m_nRows*m_nCols*sizeof(Var_t)) );
  }

  CudaMatrix(size_t nRows, size_t nCols, Var_t* mat){
    SetSize(nRows,nCols);
    ACTS_CUDA_ERROR_CHECK( cudaMalloc((Var_t**)&m_devPtr, m_nRows*m_nCols*sizeof(Var_t)) );
    CopyH2D(mat, m_size, 0);
  }
  
  CudaMatrix(size_t nRows, size_t nCols, CpuMatrix<Var_t>* mat){
    SetSize(nRows,nCols);
    ACTS_CUDA_ERROR_CHECK( cudaMalloc((Var_t**)&m_devPtr, m_nRows*m_nCols*sizeof(Var_t)) );
    CopyH2D(mat->Get(0,0), m_size, 0);
  }

  CudaMatrix(size_t nRows, size_t nCols, Var_t* mat, size_t len, size_t offset){
    SetSize(nRows,nCols);
    ACTS_CUDA_ERROR_CHECK( cudaMalloc((Var_t**)&m_devPtr, m_nRows*m_nCols*sizeof(Var_t)) );
    CopyH2D(mat, len, offset);
  }
  
  CudaMatrix(size_t nRows, size_t nCols, CpuMatrix<Var_t>* mat, size_t len, size_t offset){
    SetSize(nRows,nCols);
    ACTS_CUDA_ERROR_CHECK( cudaMalloc((Var_t**)&m_devPtr, m_nRows*m_nCols*sizeof(Var_t)) );
    CopyH2D(mat->Get(0,0),len,offset);
  }
  
  ~CudaMatrix(){
    cudaFree(m_devPtr);
  }

  void SetSize(size_t row, size_t col){
    m_nRows = row;
    m_nCols = col;
    m_size  = m_nRows*m_nCols; 
  }
  
  size_t GetNCols(){ return m_nCols; }
  size_t GetNRows(){ return m_nRows; }
  size_t GetSize() { return m_size; }
  
  Var_t* Get(size_t row=0, size_t col=0){
    int offset = row+col*m_nRows;
    return m_devPtr+offset;
  }

  void CopyH2D(Var_t* matrix, size_t len, size_t offset=0){
    ACTS_CUDA_ERROR_CHECK( cudaMemcpy(m_devPtr+offset, matrix, len*sizeof(Var_t), cudaMemcpyHostToDevice) );
  }

  void CopyH2D(const Var_t* matrix, size_t len, size_t offset=0){
    ACTS_CUDA_ERROR_CHECK( cudaMemcpy(m_devPtr+offset, matrix, len*sizeof(Var_t), cudaMemcpyHostToDevice) );
  }
  
  void Zeros(){
    ACTS_CUDA_ERROR_CHECK( cudaMemset(m_devPtr, 0, m_size*sizeof(Var_t)) );
  }
  
private:
  Var_t* m_devPtr; 
  size_t m_nCols;
  size_t m_nRows;
  size_t m_size;
};

}

