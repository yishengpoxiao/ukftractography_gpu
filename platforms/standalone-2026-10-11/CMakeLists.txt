cmake_minimum_required(VERSION 3.22)
project(UKFGPUBackend LANGUAGES C CXX CUDA)
get_filename_component(UKF_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
find_package(CUDAToolkit 12.1 REQUIRED)
find_package(Threads REQUIRED)
find_package(ITK 5 REQUIRED)
include(${ITK_USE_FILE})
find_package(VTK 9 REQUIRED COMPONENTS CommonSystem CommonCore CommonMath CommonMisc
  CommonTransforms IOLegacy IOXML)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(ZLIB REQUIRED)
find_package(Teem REQUIRED)
find_package(SlicerExecutionModel REQUIRED)
include(${GenerateCLP_USE_FILE})

# Keep the original XML untouched. A build-tree copy gives GenerateCLP the
# expected UKFTractographyCLP.h basename with the current GPU CLI options.
configure_file("${UKF_SOURCE}/UKFTractography/UKFTractography-2026-10-10.xml"
  "${CMAKE_CURRENT_BINARY_DIR}/UKFTractography.xml" COPYONLY)
set(cli_sources "${UKF_SOURCE}/ukf/cli.cc")
generateclp(cli_sources "${CMAKE_CURRENT_BINARY_DIR}/UKFTractography.xml")
set(UKF_REVISION "source-archive")
if(EXISTS "${UKF_SOURCE}/.git")
  find_package(Git REQUIRED)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${UKF_SOURCE}" rev-parse HEAD
    OUTPUT_VARIABLE UKF_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/git_version.cc"
  "#include \"git_version.h\"\nconst char UKF_GIT_HASH[] = \"${UKF_REVISION}\";\n")

set(base_sources utilities.cc tractography.cc unscented_kalman_filter.cc seed.cc
  ukffiber.cc NrrdData.cc vtk_writer.cc dwi_normalize.cc thread.cc
  QuadProg++_Eigen.cc filter_model.cc filter_Full1T.cc filter_Full1T_FW.cc
  filter_Full2T.cc filter_Full2T_FW.cc filter_Full3T.cc filter_NODDI1F.cc
  filter_NODDI2F.cc filter_Simple1T.cc filter_Simple1T_FW.cc filter_Simple2T.cc
  filter_Simple2T_FW.cc filter_Simple3T.cc gpu/gpu_tractography.cc)
list(TRANSFORM base_sources PREPEND "${UKF_SOURCE}/ukf/")
add_library(ukf_gpu SHARED "${UKF_SOURCE}/ukf/gpu/gpu_kernels.cu")
target_compile_definitions(ukf_gpu PUBLIC UKF_GPU_SHARED PRIVATE UKF_GPU_BUILD)
target_compile_options(ukf_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
target_link_libraries(ukf_gpu PRIVATE CUDA::cudart)
add_library(UKFBase STATIC ${base_sources} ${cli_sources}
  "${CMAKE_CURRENT_BINARY_DIR}/git_version.cc" "${UKF_SOURCE}/common/BRAINSThreadControl.cxx")
target_compile_definitions(UKFBase PUBLIC UKF_STATIC UKF_USE_GPU)
target_include_directories(UKFBase PUBLIC "${UKF_SOURCE}/ukf" "${UKF_SOURCE}/common"
  "${CMAKE_CURRENT_BINARY_DIR}" ${Teem_INCLUDE_DIRS})
target_link_libraries(UKFBase PUBLIC ukf_gpu Eigen3::Eigen Threads::Threads
  ${ITK_LIBRARIES} ${VTK_LIBRARIES} ${SlicerExecutionModel_LIBRARIES} teem ZLIB::ZLIB)
add_executable(UKFTractography_gpu main.cc "${UKF_SOURCE}/UKFTractography/UKFTractography.cxx")
target_link_libraries(UKFTractography_gpu PRIVATE UKFBase)
if(MSVC)
  target_compile_definitions(UKFBase PUBLIC NOMINMAX _USE_MATH_DEFINES)
  target_compile_definitions(ukf_gpu PRIVATE NOMINMAX _USE_MATH_DEFINES)
  target_compile_options(UKFBase PRIVATE /bigobj /fp:strict /utf-8)
  target_compile_options(UKFTractography_gpu PRIVATE /fp:strict /utf-8)
  target_compile_options(ukf_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  target_sources(UKFTractography_gpu PRIVATE UKFGPU-2026-10-11.manifest)
endif()
set_target_properties(ukf_gpu UKFTractography_gpu PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
install(TARGETS UKFTractography_gpu ukf_gpu
  RUNTIME DESTINATION bin LIBRARY DESTINATION lib)
