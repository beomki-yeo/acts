#include "Acts/Utilities/Platforms/Matrix/MatrixDef.h"
#include "Acts/Utilities/Platforms/Matrix/Kernels.cu"
#include <assert.h>
//#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

//using boost::test_tools::output_test_stream;

namespace Acts{
namespace Test{

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_CASE( CUDAOBJ_TEST ){
  //------------------------------------------------
  // Test Matrix backend
  //------------------------------------------------

  const int row = 3;
  const int col = 3;

  float matA[row*col] = {1,2,3,4,5,6,7,8,9};
  float matB[row*col] = {9,8,7,6,5,4,3,2,1};

  CUDA::MatrixX<float,row,col> cuMatA(matA);
  CUDA::MatrixX<float,row,col> cuMatB(matB);
  CUDA::MatrixX<float,row,col> cuMatC = CUDA::MatrixX<float,row,col>();
  
  int dimBlock = row*col;
  int dimGrid  = 1;
  
  ArraySum<<<dimGrid, dimBlock>>>(cuMatA.data(),cuMatB.data(),cuMatC.data());
  
  float* matC = cuMatC.dataHost();

  for (int i=0; i< row*col; i++){
    BOOST_REQUIRE( matC[i] == 10 ); 
  }   
}
BOOST_AUTO_TEST_SUITE_END()

}
}
