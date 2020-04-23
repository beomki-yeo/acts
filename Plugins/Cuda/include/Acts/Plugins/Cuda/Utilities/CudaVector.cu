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
#include "Acts/Plugins/Cuda/Utilities/CpuVector.hpp"
#include "CudaUtils.cu"

namespace Acts{

template<typename Var_t>
class CudaVector{

public:
  
  CudaVector(size_t size){ 
    m_size = size;
    cudaErrChk( cudaMalloc((Var_t**)&m_devPtr, m_size*sizeof(Var_t)) );
  }

  CudaVector(size_t size, Var_t* vector){
    m_size = size;
    cudaErrChk( cudaMalloc((Var_t**)&m_devPtr, m_size*sizeof(Var_t)) );
    CopyH2D(vector, m_size, 0);
  }
     
  CudaVector(size_t size, Var_t* vector, size_t len, size_t offset){ 
    m_size = size;
    cudaErrChk( cudaMalloc((Var_t**)&m_devPtr, m_size*sizeof(Var_t)) );
    CopyH2D(vector, len, offset);
  }
  
  ~CudaVector(){ 
    cudaFree(m_devPtr); 
  }

  size_t GetSize(){return m_size;}
  
  Var_t* Get(size_t offset=0) { return m_devPtr+offset; }

  Var_t* GetHost() {
    Var_t* fHostPtr = new Var_t[m_size];
    cudaErrChk( cudaMemcpy(fHostPtr, m_devPtr, m_size*sizeof(Var_t), cudaMemcpyDeviceToHost) );
    return fHostPtr;
  }

  void CopyH2D(Var_t* vector, size_t len, size_t offset){
    cudaErrChk( cudaMemcpy(m_devPtr+offset, vector, len*sizeof(Var_t), cudaMemcpyHostToDevice) );
  }
  void CopyH2D(Var_t* vector, size_t len, size_t offset, cudaStream_t* stream){
    cudaErrChk( cudaMemcpyAsync(m_devPtr+offset, vector, len*sizeof(Var_t), cudaMemcpyHostToDevice, *stream) );
  }

  void Zeros(){
    cudaErrChk( cudaMemset(m_devPtr, 0, m_size*sizeof(Var_t)) );
  }
  
private:
  Var_t* m_devPtr; 
  size_t m_size;
};
}
