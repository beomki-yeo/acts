
// Developer Note (Beomki Yeo):
// Including PlatformDef.h makes an error...

#include "Acts/Utilities/Platforms/CUDA/CUDABuffer.cu"
#include "Acts/Utilities/Platforms/CUDA/Kernels.cu"
#include <assert.h>
#include <boost/test/unit_test.hpp>

namespace Acts{
namespace Test{

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_CASE( CUDAOBJ_TEST ){

  //------------------------------------------------
  // Test Matrix backend
  //------------------------------------------------

  const int size = 9;
  float matA[size] = {1,2,3,4,5,6,7,8,9};
  float matB[size] = {9,8,7,6,5,4,3,2,1};
  
  CUDABuffer<float> cuMatA(size);
  CUDABuffer<float> cuMatB(size);

  cuMatA.SetData(matA,size);
  cuMatB.SetData(matB,size);

  CUDABuffer<float> cuMatC(size);
  
  int dimBlock = size;
  int dimGrid  = 1;
  
  ArraySum<<<dimGrid, dimBlock>>>(cuMatA.data(),cuMatB.data(),cuMatC.data());
  
  float* matC = cuMatC.dataHost(size);
  
  for (int i=0; i< size; i++){
    BOOST_REQUIRE( matC[i] == 10 ); 
  }   

}
BOOST_AUTO_TEST_SUITE_END()

}
}
