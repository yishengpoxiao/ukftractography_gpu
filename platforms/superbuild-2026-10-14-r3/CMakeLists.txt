cmake_minimum_required(VERSION 3.22)
project(UKFGPUDependencies LANGUAGES C CXX)
include(ExternalProject)
# This entry builds only the standalone backend dependencies, independently of
# the upstream Slicer SDK and the original SuperBuild configuration.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(UKF_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;75-virtual"
  CACHE STRING "CUDA real architectures and forward-compatible baseline PTX")
set(UKF_DEPENDENCIES "${CMAKE_BINARY_DIR}/dependencies")
set(common_args -DCMAKE_BUILD_TYPE:STRING=Release -DBUILD_TESTING:BOOL=OFF
  -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
  -DCMAKE_CXX_STANDARD:STRING=17 -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=ON
  -DCMAKE_INSTALL_PREFIX:PATH=${UKF_DEPENDENCIES}
  -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
  -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5)
if(WIN32)
  set(zlib_static "${UKF_DEPENDENCIES}/lib/zlibstatic.lib")
else()
  set(zlib_static "${UKF_DEPENDENCIES}/lib/libz.a")
endif()
ExternalProject_Add(zlib
  GIT_REPOSITORY https://github.com/madler/zlib.git GIT_TAG v1.3.1 GIT_SHALLOW TRUE
  CMAKE_CACHE_ARGS ${common_args} -DZLIB_BUILD_EXAMPLES:BOOL=OFF)
ExternalProject_Add(Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git GIT_TAG 3.4.0 GIT_SHALLOW TRUE
  CMAKE_CACHE_ARGS ${common_args} -DEIGEN_BUILD_DOC:BOOL=OFF -DEIGEN_BUILD_TESTING:BOOL=OFF
    -DEIGEN_BUILD_BLAS:BOOL=OFF -DEIGEN_BUILD_LAPACK:BOOL=OFF -DCMAKE_Fortran_COMPILER:FILEPATH=NOTFOUND)
ExternalProject_Add(ITK
  GIT_REPOSITORY https://github.com/InsightSoftwareConsortium/ITK.git
  GIT_TAG v5.4.2 GIT_SHALLOW TRUE
  CMAKE_CACHE_ARGS ${common_args} -DBUILD_EXAMPLES:BOOL=OFF
    -DITK_BUILD_DEFAULT_MODULES:BOOL=OFF -DModule_ITKCommon:BOOL=ON
    -DModule_ITKIOXML:BOOL=ON -DKWSYS_USE_MD5:BOOL=ON
    -DITK_USE_FFTWD:BOOL=OFF -DITK_USE_FFTWF:BOOL=OFF
  INSTALL_COMMAND "")
ExternalProject_Add(VTK
  GIT_REPOSITORY https://github.com/Kitware/VTK.git GIT_TAG v9.4.1 GIT_SHALLOW TRUE GIT_SUBMODULES ""
  GIT_CONFIG core.longpaths=true
  CMAKE_CACHE_ARGS ${common_args} -DVTK_WRAP_PYTHON:BOOL=OFF
    -DVTK_GROUP_ENABLE_StandAlone:STRING=DONT_WANT
    -DVTK_GROUP_ENABLE_Rendering:STRING=NO
    -DVTK_MODULE_ENABLE_VTK_CommonSystem:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_CommonCore:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_CommonMath:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_CommonMisc:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_CommonTransforms:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_IOLegacy:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_IOXML:STRING=YES
  INSTALL_COMMAND "")
ExternalProject_Add(teem
  GIT_REPOSITORY https://github.com/Slicer/teem.git
  GIT_TAG e4746083c0e1dc0c137124c41eca5d23adf73bfa
  DEPENDS zlib
  CMAKE_CACHE_ARGS ${common_args} -DTeem_PTHREAD:BOOL=OFF -DTeem_BZIP2:BOOL=OFF
    -DTeem_ZLIB:BOOL=ON -DTeem_PNG:BOOL=OFF
    -DZLIB_INCLUDE_DIR:PATH=${UKF_DEPENDENCIES}/include
    -DZLIB_LIBRARY:FILEPATH=${zlib_static}
  INSTALL_COMMAND "")
ExternalProject_Add(SlicerExecutionModel
  GIT_REPOSITORY https://github.com/Slicer/SlicerExecutionModel.git
  GIT_TAG 0e28e41aa167b477bc4889725ca08157c152e584
  DEPENDS ITK
  CMAKE_CACHE_ARGS ${common_args} -DITK_DIR:PATH=${CMAKE_BINARY_DIR}/ITK-prefix/src/ITK-build
  INSTALL_COMMAND "")
get_filename_component(UKF_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
# ExternalProject's cache args preserve semicolon-separated architecture lists.
ExternalProject_Add(backend
  SOURCE_DIR "${UKF_SOURCE}/platforms/standalone-2026-10-14"
  BINARY_DIR "${CMAKE_BINARY_DIR}/backend"
  DEPENDS zlib Eigen ITK VTK teem SlicerExecutionModel
  CMAKE_CACHE_ARGS ${common_args}
    -DCMAKE_CUDA_ARCHITECTURES:STRING=${UKF_CUDA_ARCHITECTURES}
    -DCMAKE_CUDA_COMPILER:FILEPATH=${CMAKE_CUDA_COMPILER}
    -DITK_DIR:PATH=${CMAKE_BINARY_DIR}/ITK-prefix/src/ITK-build
    -DVTK_DIR:PATH=${CMAKE_BINARY_DIR}/VTK-prefix/src/VTK-build
    -DTeem_DIR:PATH=${CMAKE_BINARY_DIR}/teem-prefix/src/teem-build
    -DSlicerExecutionModel_DIR:PATH=${CMAKE_BINARY_DIR}/SlicerExecutionModel-prefix/src/SlicerExecutionModel-build
    -DEigen3_DIR:PATH=${UKF_DEPENDENCIES}/share/eigen3/cmake
    -DZLIB_INCLUDE_DIR:PATH=${UKF_DEPENDENCIES}/include
    -DZLIB_LIBRARY:FILEPATH=${zlib_static}
  INSTALL_COMMAND "")
