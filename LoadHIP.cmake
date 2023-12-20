set(PYTORCH_FOUND_HIP TRUE)

torch_hip_get_arch_list(PYTORCH_ROCM_ARCH)
if(PYTORCH_ROCM_ARCH STREQUAL "")
  message(FATAL_ERROR "No GPU arch specified for ROCm build. Please use PYTORCH_ROCM_ARCH environment variable to specify GPU archs to build for.")
endif()


find_package(ROCM CONFIG REQUIRED)

set(CMAKE_MODULE_PATH "/usr/lib64/cmake/hip" ${CMAKE_MODULE_PATH})

# Find the HIP Package
find_package(HIP MODULE REQUIRED)
message("HIP Version: ${HIP_VERSION}")

math(EXPR TORCH_HIP_VERSION "(${HIP_VERSION_MAJOR} * 100) + ${HIP_VERSION_MINOR}")
math(EXPR ROCM_VERSION_DEV_INT "(${HIP_VERSION_MAJOR} * 1000) + ${HIP_VERSION_MINOR} * 100")


set(CMAKE_HCC_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_HCC_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
### Remove setting of Flags when FindHIP.CMake PR #558 is accepted.###

find_package(hsa-runtime64 REQUIRED)
find_package(amd_comgr REQUIRED)
find_package(rocrand REQUIRED)
find_package(hiprand REQUIRED)
find_package(rocblas REQUIRED)
find_package(hipblas REQUIRED)
find_package(miopen REQUIRED)
find_package(hipfft REQUIRED)
find_package(hipsparse REQUIRED)
#find_package(rccl)
find_package(rocprim REQUIRED)
find_package(hipcub REQUIRED)
find_package(rocthrust REQUIRED)
find_package(hipsolver REQUIRED)

set(hip_library_name amdhip64)
message("HIP library name: ${hip_library_name}")

# TODO: hip_hcc has an interface include flag "-hc" which is only
# recognizable by hcc, but not gcc and clang. Right now in our
# setup, hcc is only used for linking, but it should be used to
# compile the *_hip.cc files as well.
find_library(PYTORCH_HIP_HCC_LIBRARIES ${hip_library_name} HINTS /usr/lib64)
# TODO: miopen_LIBRARIES should return fullpath to the library file,
# however currently it's just the lib name
if(TARGET ${miopen_LIBRARIES})
  set(PYTORCH_MIOPEN_LIBRARIES ${miopen_LIBRARIES})
else()
  find_library(PYTORCH_MIOPEN_LIBRARIES ${miopen_LIBRARIES} HINTS /usr/lib64)
endif()
# TODO: rccl_LIBRARIES should return fullpath to the library file,
# however currently it's just the lib name
# if(TARGET ${rccl_LIBRARIES})
#   set(PYTORCH_RCCL_LIBRARIES ${rccl_LIBRARIES})
# else()
#   find_library(PYTORCH_RCCL_LIBRARIES ${rccl_LIBRARIES} HINTS /usr/lib64)
# endif()
# hiprtc is part of HIP
find_library(ROCM_HIPRTC_LIB ${hip_library_name} HINTS /usr/lib64)
# roctx is part of roctracer
# find_library(ROCM_ROCTX_LIB roctx64 HINTS /usr/lib64)

