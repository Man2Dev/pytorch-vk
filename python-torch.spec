%global pypi_name torch

# Where the src comes from
%global forgeurl https://github.com/pytorch/pytorch

# So pre releases can be tried
%bcond_with gitcommit
%if %{with gitcommit}
# git tag v2.3.0-rc12
%global commit0 97ff6cfd9c86c5c09d7ce775ab64ec5c99230f5d
%global shortcommit0 %(c=%{commit0}; echo ${c:0:7})
%global date0 20240408
%endif
%global pypi_version 2.3.0

# For -test subpackage
# suitable only for local testing
# Install and do something like
#   export LD_LIBRARY_PATH=/usr/lib64/python3.12/site-packages/torch/lib
#   /usr/lib64/python3.12/site-packages/torch/bin/test_api, test_lazy
%bcond_with test

%ifarch x86_64
# ROCm support came in F40
%if 0%{?fedora} > 39
%bcond_without rocm
%else
%bcond_with rocm
%endif
%endif
# hipblaslt is in development
%bcond_with hipblaslt
# Which families gpu build for
%global rocm_gpu_list gfx8 gfx9 gfx10 gfx11
%global rocm_default_gpu default
%bcond_without rocm_loop

# Caffe2 support came in F41
%if 0%{?fedora} > 40
%bcond_without caffe2
%else
%bcond_with caffe2
%endif

# Distributed support came in F41
%if 0%{?fedora} > 40
%bcond_without distributed
# For testing distributed+rccl etc.
%bcond_without rccl
%bcond_with gloo
%bcond_without mpi
%bcond_without tensorpipe
%else
%bcond_with distributed
%endif

# OpenCV support came in F41
%if 0%{?fedora} > 40
%bcond_without opencv
%else
%bcond_with opencv
%endif

# Do no confuse xnnpack versions
%if 0%{?fedora} > 40
%bcond_without xnnpack
%else
%bcond_with xnnpack
%endif

%if 0%{?fedora} > 39
%bcond_without pthreadpool
%else
%bcond_with pthreadpool
%endif

%if 0%{?fedora} > 39
%bcond_without pocketfft
%else
%bcond_with pocketfft
%endif

# For testing cuda
%ifarch x86_64
%bcond_with cuda
%endif

# Disable dwz with rocm because memory can be exhausted
%if %{with rocm}
%define _find_debuginfo_dwz_opts %{nil}
%endif

Name:           python-%{pypi_name}
%if %{with gitcommit}
Version:        %{pypi_version}^git%{date0}.%{shortcommit0}
%else
Version:        %{pypi_version}
%endif
Release:        %autorelease
Summary:        PyTorch AI/ML framework
# See license.txt for license details
License:        BSD-3-Clause AND BSD-2-Clause AND 0BSD AND Apache-2.0 AND MIT AND BSL-1.0 AND GPL-3.0-or-later AND Zlib

URL:            https://pytorch.org/
%if %{with gitcommit}
Source0:        %{forgeurl}/archive/%{commit0}/pytorch-%{shortcommit0}.tar.gz
Source100:        pyproject.toml
%else
Source0:        %{forgeurl}/releases/download/v%{version}/pytorch-v%{version}.tar.gz
%endif
Source1:        https://github.com/google/flatbuffers/archive/refs/tags/v23.3.3.tar.gz
Source2:        https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz

%if %{with cuda}
%global cuf_ver 1.1.2
Source10:       https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v%{cuf_ver}.tar.gz
%global cul_ver 3.4.1
Source11:       https://github.com/NVIDIA/cutlass/archive/refs/tags/v%{cul_ver}.tar.gz
%endif

%if %{with tensorpipe}
# Developement on tensorpipe has stopped, repo made read only July 1, 2023, this is the last commit
%global tp_commit 52791a2fd214b2a9dc5759d36725909c1daa7f2e
%global tp_scommit %(c=%{tp_commit}; echo ${c:0:7})
Source20:       https://github.com/pytorch/tensorpipe/archive/%{tp_commit}/tensorpipe-%{tp_scommit}.tar.gz
# The old libuv tensorpipe uses
Source21:       https://github.com/libuv/libuv/archive/refs/tags/v1.41.0.tar.gz
# Developement afaik on libnop has stopped, this is the last commit
%global nop_commit 910b55815be16109f04f4180e9adee14fb4ce281
%global nop_scommit %(c=%{nop_commit}; echo ${c:0:7})
Source22:       https://github.com/google/libnop/archive/%{nop_commit}/libnop-%{nop_scommit}.tar.gz
%endif

%if %{without xnnpack}
%global xnn_commit fcbf55af6cf28a4627bcd1f703ab7ad843f0f3a2
%global xnn_scommit %(c=%{xnn_commit}; echo ${c:0:7})
Source30:       https://github.com/google/xnnpack/archive/%{xnn_commit}/xnnpack-%{xnn_scommit}.tar.gz
%global fx_commit 63058eff77e11aa15bf531df5dd34395ec3017c8
%global fx_scommit %(c=%{fx_commit}; echo ${c:0:7})
Source31:       https://github.com/Maratyszcza/fxdiv/archive/%{fx_commit}/FXdiv-%{fx_scommit}.tar.gz
%global fp_commit 0a92994d729ff76a58f692d3028ca1b64b145d91
%global fp_scommit %(c=%{fp_commit}; echo ${c:0:7})
Source32:       https://github.com/Maratyszcza/FP16/archive/%{fp_commit}/FP16-%{fp_scommit}.tar.gz
%global ps_commit 072586a71b55b7f8c584153d223e95687148a900
%global ps_scommit %(c=%{ps_commit}; echo ${c:0:7})
Source33:       https://github.com/Maratyszcza/psimd/archive/%{ps_commit}/psimd-%{ps_scommit}.tar.gz
%endif

%if %{without pthreadpool}
%global pt_commit 4fe0e1e183925bf8cfa6aae24237e724a96479b8
%global pt_scommit %(c=%{pt_commit}; echo ${c:0:7})
Source40:       https://github.com/Maratyszcza/pthreadpool/archive/%{pt_commit}/pthreadpool-%{pt_scommit}.tar.gz
%endif

%if %{without pocketfft}
%global pf_commit 076cb3d2536b7c5d0629093ad886e10ac05f3623
%global pf_scommit %(c=%{pf_commit}; echo ${c:0:7})
Source50:       https://github.com/mreineck/pocketfft/archive/%{pf_commit}/pocketfft-%{pf_scommit}.tar.gz
%endif

Patch0:        0001-no-third_party-foxi.patch
Patch1:        0001-no-third_party-fmt.patch
Patch2:        0001-no-third_party-FXdiv.patch
Patch3:        0001-Stub-in-kineto-ActivityType.patch
Patch5:        0001-disable-submodule-search.patch

%if %{with caffe2}
Patch6:        0001-reenable-foxi-linking.patch
%endif

# https://github.com/pytorch/pytorch/pull/123384
Patch7:        0001-Reenable-dim-for-python-3.12.patch

# Dynamo/Inductor on 3.12
Patch8:        0001-dynamo-3.12-enable-dynamo-on-3.12-enable-most-dynamo.patch

# ROCm patches
# https://github.com/pytorch/pytorch/pull/120551
Patch100:      0001-Optionally-use-hipblaslt.patch
Patch101:      0001-cuda-hip-signatures.patch
Patch102:      0001-silence-an-assert.patch
Patch103:      0001-can-not-use-with-c-files.patch
Patch104:      0001-use-any-hip.patch
Patch105:      0001-disable-use-of-aotriton.patch

ExclusiveArch:  x86_64 aarch64
%global toolchain gcc
%global _lto_cflags %nil

BuildRequires:  cmake
BuildRequires:  cpuinfo-devel
BuildRequires:  eigen3-devel
BuildRequires:  fmt-devel
%if %{with caffe2}
BuildRequires:  foxi-devel
%endif
BuildRequires:  gcc-c++
BuildRequires:  gcc-gfortran
%if %{with distributed}
%if %{with gloo}
BuildRequires:  gloo-devel
%endif
%endif
BuildRequires:  ninja-build
BuildRequires:  onnx-devel
BuildRequires:  libomp-devel
%if %{with distributed}
%if %{with mpi}
BuildRequires:  openmpi-devel
%endif
%endif
BuildRequires:  openblas-devel
BuildRequires:  protobuf-devel
BuildRequires:  sleef-devel
BuildRequires:  valgrind-devel

%if %{with pocketfft}
BuildRequires:  pocketfft-devel
%endif

%if %{with pthreadpool}
BuildRequires:  pthreadpool-devel
%endif

%if %{with xnnpack}
BuildRequires:  FP16-devel
BuildRequires:  fxdiv-devel
BuildRequires:  psimd-devel
BuildRequires:  xnnpack-devel = 0.0^git20240229.fcbf55a
%endif

BuildRequires:  python3-devel
BuildRequires:  python3dist(filelock)
BuildRequires:  python3dist(jinja2)
BuildRequires:  python3dist(networkx)
BuildRequires:  python3dist(numpy)
BuildRequires:  python3dist(pyyaml)
BuildRequires:  python3dist(setuptools)
BuildRequires:  python3dist(sphinx)
BuildRequires:  python3dist(typing-extensions)

%if 0%{?fedora}
BuildRequires:  python3-pybind11
BuildRequires:  python3dist(fsspec)
BuildRequires:  python3dist(sympy)
%endif

%if %{with rocm}
BuildRequires:  hipblas-devel
%if %{with hipblaslt}
BuildRequires:  hipblaslt-devel
%endif
BuildRequires:  hipcub-devel
BuildRequires:  hipfft-devel
BuildRequires:  hiprand-devel
BuildRequires:  hipsparse-devel
BuildRequires:  hipsolver-devel
BuildRequires:  miopen-devel
BuildRequires:  rocblas-devel
BuildRequires:  rocrand-devel
BuildRequires:  rocfft-devel
%if %{with distributed}
%if %{with rccl}
BuildRequires:  rccl-devel
%endif
%endif
BuildRequires:  rocprim-devel
BuildRequires:  rocm-cmake
BuildRequires:  rocm-comgr-devel
BuildRequires:  rocm-core-devel
BuildRequires:  rocm-hip-devel
BuildRequires:  rocm-runtime-devel
BuildRequires:  rocm-rpm-macros
BuildRequires:  rocm-rpm-macros-modules
BuildRequires:  rocthrust-devel
BuildRequires:  roctracer-devel

Requires:       rocm-rpm-macros-modules
%endif

%if %{with opencv}
BuildRequires:  opencv-devel
%endif

%if %{with test}
BuildRequires:  google-benchmark-devel
%endif

Requires:       python3dist(dill)

# For convience
Provides:       pytorch

# Apache-2.0
Provides:       bundled(flatbuffers) = 22.3.3
# MIT
Provides:       bundled(miniz) = 2.1.0
Provides:       bundled(pybind11) = 2.11.1

%if %{with tensorpipe}
# BSD-3-Clause
Provides:       bundled(tensorpipe)
# Apache-2.0
Provides:       bundled(libnop)
# MIT AND CC-BY-4.0 AND ISC AND BSD-2-Clause
Provides:       bundled(libuv) = 1.41.0
%endif

# These are already in Fedora
%if %{without xnnpack}
# BSD-3-Clause
Provides:       bundled(xnnpack)
# MIT
Provides:       bundled(FP16)
# MIT
Provides:       bundled(fxdiv)
# MIT
Provides:       bundled(psimd)
%endif

%if %{without pthreadpool}
# BSD-2-Clause
Provides:       bundled(pthreadpool)
%endif

%if %{without pocketfft}
# BSD-3-Clause
Provides:       bundled(pocketfft)
%endif

%description
PyTorch is a Python package that provides two high-level features:

 * Tensor computation (like NumPy) with strong GPU acceleration
 * Deep neural networks built on a tape-based autograd system

You can reuse your favorite Python packages such as NumPy, SciPy,
and Cython to extend PyTorch when needed.

%package -n     python3-%{pypi_name}
Summary:        %{summary}

%description -n python3-%{pypi_name}
PyTorch is a Python package that provides two high-level features:

 * Tensor computation (like NumPy) with strong GPU acceleration
 * Deep neural networks built on a tape-based autograd system

You can reuse your favorite Python packages such as NumPy, SciPy,
and Cython to extend PyTorch when needed.

%if %{with rocm}
%package -n python3-%{pypi_name}-rocm-gfx8
Summary:        %{name} for ROCm gfx8

%description -n python3-%{pypi_name}-rocm-gfx8
%{summary}

%package -n python3-%{pypi_name}-rocm-gfx9
Summary:        %{name} for ROCm gfx9

%description -n python3-%{pypi_name}-rocm-gfx9
%{summary}

%package -n python3-%{pypi_name}-rocm-gfx10
Summary:        %{name} for ROCm gfx10

%description -n python3-%{pypi_name}-rocm-gfx10
%{summary}

%package -n python3-%{pypi_name}-rocm-gfx11
Summary:        %{name} for ROCm gfx11

%description -n python3-%{pypi_name}-rocm-gfx11
%{summary}

%endif

%if %{with test}
%package -n python3-%{pypi_name}-test
Summary:        Tests for %{name}
Requires:       python3-%{pypi_name}%{?_isa} = %{version}-%{release}

%description -n python3-%{pypi_name}-test
%{summary}
%endif


%prep

%if %{with gitcommit}
%autosetup -p1 -n pytorch-%{commit0}
# Overwrite with a git checkout of the pyproject.toml
cp %{SOURCE100} .
%else
%autosetup -p1 -n pytorch-v%{version}
%endif

# Remove bundled egg-info
rm -rf %{pypi_name}.egg-info

tar xf %{SOURCE1}
rm -rf third_party/flatbuffers/*
cp -r flatbuffers-23.3.3/* third_party/flatbuffers/

tar xf %{SOURCE2}
rm -rf third_party/pybind11/*
cp -r pybind11-2.11.1/* third_party/pybind11/

%if %{with cuda}
tar xf %{SOURCE10}
rm -rf third_party/cudnn_frontend/*
cp -r cudnn-frontend-%{cuf_ver}/* third_party/cudnn_frontend/
tar xf %{SOURCE11}
rm -rf third_party/cutlass/*
cp -r cutlass-%{cul_ver}/* third_party/cutlass/
%endif

%if %{with tensorpipe}
tar xf %{SOURCE20}
rm -rf third_party/tensorpipe/*
cp -r tensorpipe-*/* third_party/tensorpipe/
tar xf %{SOURCE21}
rm -rf third_party/tensorpipe/third_party/libuv/*
cp -r libuv-*/* third_party/tensorpipe/third_party/libuv/
tar xf %{SOURCE22}
rm -rf third_party/tensorpipe/third_party/libnop/*
cp -r libnop-*/* third_party/tensorpipe/third_party/libnop/
%endif

%if %{without xnnpack}
tar xf %{SOURCE30}
rm -rf third_party/XNNPACK/*
cp -r XNNPACK-*/* third_party/XNNPACK/
tar xf %{SOURCE31}
rm -rf third_party/FXdiv/*
cp -r FXdiv-*/* third_party/FXdiv/
tar xf %{SOURCE32}
rm -rf third_party/FP16/*
cp -r FP16-*/* third_party/FP16/
tar xf %{SOURCE33}
rm -rf third_party/psimd/*
cp -r psimd-*/* third_party/psimd/
%endif

%if %{without pthreadpool}
tar xf %{SOURCE40}
rm -rf third_party/pthreadpool/*
cp -r pthreadpool-*/* third_party/pthreadpool/
%endif

%if %{without pocketfft}
tar xf %{SOURCE50}
rm -rf third_party/pocketfft/*
cp -r pocketfft-*/* third_party/pocketfft/
%endif

%if %{with opencv}
# Reduce requirements, *FOUND is not set 
sed -i -e 's/USE_OPENCV AND OpenCV_FOUND AND USE_FFMPEG AND FFMPEG_FOUND/USE_OPENCV AND USE_FFMPEG/' caffe2/video/CMakeLists.txt
sed -i -e 's/USE_OPENCV AND OpenCV_FOUND/USE_OPENCV/' caffe2/image/CMakeLists.txt
sed -i -e 's/STATUS/FATAL/' caffe2/image/CMakeLists.txt
%endif

%if 0%{?rhel}
# In RHEL but too old
sed -i -e '/typing-extensions/d' setup.py
# Need to pip these
sed -i -e '/sympy/d' setup.py
sed -i -e '/fsspec/d' setup.py
%endif

# A new dependency
# Connected to USE_FLASH_ATTENTION, since this is off, do not need it
sed -i -e '/aotriton.cmake/d' cmake/Dependencies.cmake

# Release comes fully loaded with third party src
# Remove what we can
#
# For 2.1 this is all but miniz-2.1.0
# Instead of building as a library, caffe2 reaches into
# the third_party dir to compile the file.
# mimiz is licensed MIT
# https://github.com/richgel999/miniz/blob/master/LICENSE
mv third_party/miniz-2.1.0 .
#
# setup.py depends on this script
mv third_party/build_bundled.py .

# Need the just untarred flatbuffers/flatbuffers.h
mv third_party/flatbuffers .

mv third_party/pybind11 .

%if %{with cuda}
mv third_party/cudnn_frontend .
mv third_party/cutlass .
%endif

%if %{with tensorpipe}
mv third_party/tensorpipe .
%endif

%if %{without xnnpack}
mv third_party/XNNPACK .
mv third_party/FXdiv .
mv third_party/FP16 .
mv third_party/psimd .
%endif

%if %{without pthreadpool}
mv third_party/pthreadpool .
%endif

%if %{without pocketfft}
mv third_party/pocketfft .
%endif

%if %{with test}
mv third_party/googletest .
%endif

# Remove everything
rm -rf third_party/*
# Put stuff back
mv build_bundled.py third_party
mv miniz-2.1.0 third_party
mv flatbuffers third_party
mv pybind11 third_party

%if %{with cuda}
mv cudnn_frontend third_party
mv cutlass third_party
%endif

%if %{with tensorpipe}
mv tensorpipe third_party
%endif

%if %{without xnnpack}
mv XNNPACK third_party
mv FXdiv third_party
mv FP16 third_party
mv psimd third_party
%endif

%if %{without pthreadpool}
mv pthreadpool third_party
%endif

%if %{without pocketfft}
mv pocketfft third_party
%endif

%if %{with test}
mv googletest third_party
%endif

%if %{with pocketfft}
#
# Fake out pocketfft, and system header will be used
mkdir third_party/pocketfft
%endif

#
# Use the system valgrind headers
mkdir third_party/valgrind-headers
cp %{_includedir}/valgrind/* third_party/valgrind-headers

# Remove unneeded OpenCL files that confuse the lincense scanner
rm caffe2/contrib/opencl/OpenCL/cl.hpp
rm caffe2/mobile/contrib/libopencl-stub/include/CL/*.h
rm caffe2/mobile/contrib/libopencl-stub/include/CL/*.hpp

%if %{with rocm}
# hipify
./tools/amd_build/build_amd.py
# Fedora installs to /usr/include, not /usr/include/rocm-core
sed -i -e 's@rocm-core/rocm_version.h@rocm_version.h@' aten/src/ATen/hip/tunable/TunableGemm.h
%endif

%if %{with cuda}
# build complains about not being able to build -pie without -fPIC
sed -i -e 's@string(APPEND CMAKE_CUDA_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")@string(APPEND CMAKE_CUDA_FLAGS " -fPIC -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")@' CMakeLists.txt
%endif

%build

#
# Control the number of jobs
#
# The build can fail if too many threads exceed the physical memory
# So count core and and memory and increase the build memory util the build succeeds
#
# Real cores, No hyperthreading
COMPILE_JOBS=`cat /proc/cpuinfo | grep -m 1 'cpu cores' | awk '{ print $4 }'`
if [ ${COMPILE_JOBS}x = x ]; then
    COMPILE_JOBS=1
fi
# Take into account memmory usage per core, do not thrash real memory
%if %{with cuda}
BUILD_MEM=4
%else
BUILD_MEM=2
%endif
MEM_KB=0
MEM_KB=`cat /proc/meminfo | grep MemTotal | awk '{ print $2 }'`
MEM_MB=`eval "expr ${MEM_KB} / 1024"`
MEM_GB=`eval "expr ${MEM_MB} / 1024"`
COMPILE_JOBS_MEM=`eval "expr 1 + ${MEM_GB} / ${BUILD_MEM}"`
if [ "$COMPILE_JOBS_MEM" -lt "$COMPILE_JOBS" ]; then
    COMPILE_JOBS=$COMPILE_JOBS_MEM
fi
export MAX_JOBS=$COMPILE_JOBS

# For debugging setup.py
# export SETUPTOOLS_SCM_DEBUG=1

# For verbose cmake output
# export VERBOSE=ON
# For verbose linking
# export CMAKE_SHARED_LINKER_FLAGS=-Wl,--verbose

# Manually set this hardening flag
export CMAKE_EXE_LINKER_FLAGS=-pie

export BUILD_CUSTOM_PROTOBUF=OFF
export BUILD_NVFUSER=OFF
export BUILD_SHARED_LIBS=ON
export BUILD_TEST=OFF
export CMAKE_BUILD_TYPE=RelWithDebInfo
export CMAKE_FIND_PACKAGE_PREFER_CONFIG=ON
export CAFFE2_LINK_LOCAL_PROTOBUF=OFF
export INTERN_BUILD_MOBILE=OFF
export USE_DISTRIBUTED=OFF
export USE_CUDA=OFF
export USE_FBGEMM=OFF
export USE_FLASH_ATTENTION=OFF
export USE_GOLD_LINKER=OFF
export USE_GLOO=OFF
export USE_ITT=OFF
export USE_KINETO=OFF
export USE_LITE_INTERPRETER_PROFILER=OFF
export USE_LITE_PROTO=OFF
export USE_MAGMA=OFF
export USE_MKLDNN=OFF
export USE_MPI=OFF
export USE_NCCL=OFF
export USE_NNPACK=OFF
export USE_NUMPY=ON
export USE_OPENMP=ON
export USE_PYTORCH_QNNPACK=OFF
export USE_QNNPACK=OFF
export USE_ROCM=OFF
export USE_SYSTEM_CPUINFO=ON
export USE_SYSTEM_SLEEF=ON
export USE_SYSTEM_EIGEN_INSTALL=ON
export USE_SYSTEM_ONNX=ON
export USE_SYSTEM_PYBIND11=OFF
export USE_SYSTEM_LIBS=OFF
export USE_TENSORPIPE=OFF
export USE_XNNPACK=ON

%if %{with pthreadpool}
export USE_SYSTEM_PTHREADPOOL=ON
%endif

%if %{with xnnpack}
export USE_SYSTEM_FP16=ON
export USE_SYSTEM_FXDIV=ON
export USE_SYSTEM_PSIMD=ON
export USE_SYSTEM_XNNPACK=ON
%endif

%if %{with caffe2}
export BUILD_CAFFE2=ON
%endif

%if %{with cuda}
%if %{without rocm}
export CUDACXX=/usr/local/cuda/bin/nvcc
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include
export USE_CUDA=ON
# The arches to build for
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
%endif
%endif

%if %{with distributed}
export USE_DISTRIBUTED=ON
%if %{with tensorpipe}
export USE_TENSORPIPE=ON
export TP_BUILD_LIBUV=OFF
%endif

%if %{with gloo}
export USE_GLOO=ON
export USE_SYSTEM_GLOO=ON
%endif
%if %{with mpi}
export USE_MPI=ON
%endif
%endif

%if %{with opencv}
export USE_OPENCV=ON
%endif

%if %{with test}
export BUILD_TEST=ON
%endif

# Why we are using py3_ vs pyproject_
#
# current pyproject problem with mock
# + /usr/bin/python3 -Bs /usr/lib/rpm/redhat/pyproject_wheel.py /builddir/build/BUILD/pytorch-v2.1.0/pyproject-wheeldir
# /usr/bin/python3: No module named pip
# Adding pip to build requires does not fix
#
# See BZ 2244862


%if %{with rocm}

export USE_ROCM=ON
export HIP_PATH=`hipconfig -p`
export ROCM_PATH=`hipconfig -R`
export HIP_CLANG_PATH=`hipconfig -l`
RESOURCE_DIR=`${HIP_CLANG_PATH}/clang -print-resource-dir`
export DEVICE_LIB_PATH=${RESOURCE_DIR}/amdgcn/bitcode

gpu=%{rocm_default_gpu}
module load rocm/$gpu
export PYTORCH_ROCM_ARCH=$ROCM_GPUS
%py3_build
mv build build-${gpu}
module purge

%if %{with rocm_loop}
for gpu in %{rocm_gpu_list}
do
    module load rocm/$gpu
    export PYTORCH_ROCM_ARCH=$ROCM_GPUS
    %py3_build
    mv build build-${gpu}
    module purge
done
%endif

%else

%py3_build

%endif

%install

%if %{with rocm}

export USE_ROCM=ON
export HIP_PATH=`hipconfig -p`
export ROCM_PATH=`hipconfig -R`
export HIP_CLANG_PATH=`hipconfig -l`
RESOURCE_DIR=`${HIP_CLANG_PATH}/clang -print-resource-dir`
export DEVICE_LIB_PATH=${RESOURCE_DIR}/amdgcn/bitcode

gpu=%{rocm_default_gpu}
module load rocm/$gpu
export PYTORCH_ROCM_ARCH=$ROCM_GPUS
mv build-${gpu} build
%py3_install
mv build build-${gpu}
module purge

%if %{with rocm_loop}
for gpu in %{rocm_gpu_list}
do
    module load rocm/$gpu
    export PYTORCH_ROCM_ARCH=$ROCM_GPUS
    mv build-${gpu} build
    # need to customize the install location, so replace py3_install
    %{__python3} %{py_setup} %{?py_setup_args} install -O1 --skip-build --root %{buildroot} --prefix /usr/lib64/rocm/${gpu} %{?*}
    rm -rfv %{buildroot}/usr/lib/rocm/${gpu}/bin/__pycache__
    mv build build-${gpu}
    module purge
done
%endif

%else
%py3_install

%endif

# Do not remote the empty files


%files -n python3-%{pypi_name} 
%license LICENSE
%doc README.md 
%{_bindir}/convert-caffe2-to-onnx
%{_bindir}/convert-onnx-to-caffe2
%{_bindir}/torchrun
%{python3_sitearch}/%{pypi_name}
%{python3_sitearch}/%{pypi_name}-*.egg-info
%{python3_sitearch}/functorch
%{python3_sitearch}/torchgen
%if %{with caffe2}
%{python3_sitearch}/caffe2
%endif

%if %{with rocm}
%files -n python3-%{pypi_name}-rocm-gfx8
%{_libdir}/rocm/gfx8/bin/*
%{_libdir}/rocm/gfx8/lib64/*

%files -n python3-%{pypi_name}-rocm-gfx9
%{_libdir}/rocm/gfx9/bin/*
%{_libdir}/rocm/gfx9/lib64/*

%files -n python3-%{pypi_name}-rocm-gfx10
%{_libdir}/rocm/gfx10/bin/*
%{_libdir}/rocm/gfx10/lib64/*

%files -n python3-%{pypi_name}-rocm-gfx11
%{_libdir}/rocm/gfx11/bin/*
%{_libdir}/rocm/gfx11/lib64/*
%endif

%changelog
%autochangelog

