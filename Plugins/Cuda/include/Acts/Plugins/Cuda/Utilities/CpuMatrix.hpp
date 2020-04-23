// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/Cuda/Utilities/CudaMatrix.cu"

// column-major style Matrix Definition

namespace Acts {

template <typename Var_t>
class CudaMatrix;

template <typename Var_t>
class CpuMatrix {
 public:
  CpuMatrix() = default;
  CpuMatrix(size_t nRows, size_t nCols, bool pinned = 0) {
    SetSize(nRows, nCols);
    m_pinned = pinned;
    if (pinned == 0) {
      m_hostPtr = new Var_t[m_size];
    } else if (pinned == 1) {
      cudaMallocHost(&m_hostPtr, m_size * sizeof(Var_t));
    }
  }

  CpuMatrix(size_t nRows, size_t nCols, CudaMatrix<Var_t>* cuMat,
            bool pinned = 0) {
    SetSize(nRows, nCols);
    m_pinned = pinned;
    if (pinned == 0) {
      m_hostPtr = new Var_t[m_size];
    } else if (pinned == 1) {
      cudaMallocHost(&m_hostPtr, m_nRows * m_nCols * sizeof(Var_t));
    }
    cudaMemcpy(m_hostPtr, cuMat->Get(0, 0), m_size * sizeof(Var_t),
               cudaMemcpyDeviceToHost);
  }

  ~CpuMatrix() {
    if (!m_pinned) {
      delete m_hostPtr;
    } else if (m_pinned) {
      cudaFreeHost(m_hostPtr);
    }
  }

  void SetSize(size_t row, size_t col) {
    m_nRows = row;
    m_nCols = col;
    m_size = m_nRows * m_nCols;
  }

  size_t GetNCols() { return m_nCols; }
  size_t GetNRows() { return m_nRows; }
  size_t GetSize() { return m_size; }

  Var_t* Get(size_t row = 0, size_t col = 0) {
    size_t offset = row + col * m_nRows;
    return m_hostPtr + offset;
  }

  void Set(size_t row, size_t col, Var_t val) {
    size_t offset = row + col * m_nRows;
    m_hostPtr[offset] = val;
  }

  Var_t* GetColumn(size_t col) { return m_hostPtr + col * m_nRows; }
  Var_t* GetRow(size_t row) {
    Var_t* ret = new Var_t[m_nCols];
    for (size_t i_c = 0; i_c < m_nCols; i_c++)
      ret[i_c] = m_hostPtr[row + m_nRows * i_c];
    return ret;
  }

  void SetRow(size_t row, Var_t* input) {
    for (size_t i_c = 0; i_c < m_nCols; i_c++) {
      m_hostPtr[row + m_nRows * i_c] = input[i_c];
    }
  }

  void SetColumn(size_t col, Var_t* input) {
    m_hostPtr[col * m_nRows] = input[0];
  }

  void CopyD2H(Var_t* devPtr, size_t len, size_t offset) {
    cudaMemcpy(m_hostPtr + offset, devPtr, len * sizeof(Var_t),
               cudaMemcpyDeviceToHost);
  }

  void CopyD2H(Var_t* devPtr, size_t len, size_t offset, cudaStream_t* stream) {
    cudaMemcpyAsync(m_hostPtr + offset, devPtr, len * sizeof(Var_t),
                    cudaMemcpyDeviceToHost, *stream);
  }

  void Zeros() { memset(m_hostPtr, 0, m_size * sizeof(Var_t)); }

 private:
  Var_t* m_hostPtr;
  size_t m_nCols;
  size_t m_nRows;
  size_t m_size;
  bool m_pinned;
};

}  // namespace Acts
