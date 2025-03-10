cmake_minimum_required(VERSION 3.20)

project(ollama-k80 CUDA CXX)

include(CheckLanguage)

find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GGML_BUILD ON)
set(GGML_SHARED ON)
set(GGML_CCACHE ON)
set(GGML_BACKEND_DL ON)
set(GGML_BACKEND_SHARED ON)
set(GGML_SCHED_MAX_COPIES 4)

set(GGML_LLAMAFILE ON)
set(GGML_CUDA_PEER_MAX_BATCH_SIZE 128)
set(GGML_CUDA_GRAPHS ON)
set(GGML_CUDA_FA ON)

if((CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    OR (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64|ARMv[0-9]+"))
    set(GGML_CPU_ALL_VARIANTS ON)
endif()

if (CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
    set(CMAKE_BUILD_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH "@loader_path")
endif()

set(OLLAMA_BUILD_DIR ${CMAKE_BINARY_DIR}/lib/ollama)
set(OLLAMA_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/ollama)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu/amx)

# 调整项目结构
set(GGML_CUDA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml)
set(GGML_CUDA_INCLUDE_DIRS
    ${GGML_CUDA_ROOT}/include
    ${GGML_CUDA_ROOT}/src
    ${GGML_CUDA_ROOT}/src/ggml-cuda
    ${GGML_CUDA_ROOT}/src/vendors
)

# 添加头文件搜索路径
include_directories(
    ${GGML_CUDA_ROOT}/src
    ${GGML_CUDA_ROOT}/include
    ${GGML_CUDA_ROOT}/src/ggml-cuda
    ${GGML_CUDA_ROOT}/src/vendors
    ${CUDA_INCLUDE_DIRS}
)

# 确保CUDA头文件可以被找到
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(GGML_CPU ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
set_property(TARGET ggml PROPERTY EXCLUDE_FROM_ALL TRUE)

get_target_property(CPU_VARIANTS ggml-cpu MANUALLY_ADDED_DEPENDENCIES)
if(NOT CPU_VARIANTS)
    set(CPU_VARIANTS "ggml-cpu")
endif()

install(TARGETS ggml-base ${CPU_VARIANTS}
    RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES ".*"
    RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
    LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
    FRAMEWORK DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24" AND NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()

    find_package(CUDAToolkit 6.5 REQUIRED)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cuda)
    set(OLLAMA_CUDA_INSTALL_DIR ${OLLAMA_INSTALL_DIR}/cuda_v${CUDAToolkit_VERSION_MAJOR})
    install(TARGETS ggml-cuda
        RUNTIME_DEPENDENCIES
            DIRECTORIES ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_LIBRARY_DIR}
            PRE_INCLUDE_REGEXES cublas cublasLt cudart
            PRE_EXCLUDE_REGEXES ".*"
        RUNTIME DESTINATION ${OLLAMA_CUDA_INSTALL_DIR} COMPONENT CUDA
        LIBRARY DESTINATION ${OLLAMA_CUDA_INSTALL_DIR} COMPONENT CUDA
    )

    # 检查CUDA版本是否兼容K80
    if(CUDAToolkit_VERSION VERSION_GREATER "11.4")
        message(FATAL_ERROR "Tesla K80 only supports CUDA version up to 11.4")
    endif()
    if(CUDAToolkit_VERSION VERSION_LESS "6.5")
        message(FATAL_ERROR "Tesla K80 requires CUDA version 6.5 or higher")
    endif()

    # 添加K80的compute capability支持
    set(CMAKE_CUDA_ARCHITECTURES "37" CACHE STRING "CUDA architectures to compile for")

    # 调整CUDA编译选项
    if (GGML_CUDA_FORCE_DMMV)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGGML_CUDA_FORCE_DMMV")
    endif()

    # 设置K80特定的CUDA编译选项
    set(K80_CUDA_FLAGS
        -arch=sm_37                     # K80的compute capability
        -gencode arch=sm_37,code=sm_37  # 生成特定架构代码
        --use_fast_math                 # 使用快速数学函数
        --maxrregcount=64              # 限制寄存器使用
        -Xptxas=-v,-O3                 # PTX汇编优化
        --compiler-options -fPIC        # 位置无关代码
        -D_FORCE_INLINES               # 强制内联
        -DGGML_CUDA_K80               # K80特定宏定义
        --diag-suppress=177           # 抑制某些警告
    )

    # 设置调试选项
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND K80_CUDA_FLAGS
            -G                        # 生成调试信息
            -DGGML_CUDA_DEBUG        # 启用调试宏
            --ptxas-options=-v       # 显示详细编译信息
        )
    endif()

    # 设置CUDA编译器选项
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${K80_CUDA_FLAGS}")

    # 添加预编译头文件支持
    if(MSVC)
        target_precompile_headers(ggml-cuda PRIVATE
            <cuda_runtime.h>
            <cuda_fp16.h>
            <cstdint>
            <climits>
        )
    endif()

    # 设置CUDA编译器选项
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G")
    endif()
endif()

set(WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX "^gfx(906|908|90a):xnack[+-]$"
    CACHE STRING
    "Regular expression describing AMDGPU_TARGETS not supported on Windows. Override to force building these targets. Default \"^gfx(906|908|90a):xnack[+-]$\"."
)

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    set(HIP_PLATFORM "amd")

    find_package(hip REQUIRED)
    if(NOT AMDGPU_TARGETS)
        list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(900|94[012]|101[02]|1030|110[012])$")
    elseif(WIN32 AND WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX)
        list(FILTER AMDGPU_TARGETS EXCLUDE REGEX ${WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX})
    endif()

    if(AMDGPU_TARGETS)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-hip)

        if (WIN32)
            target_compile_definitions(ggml-hip PRIVATE GGML_CUDA_NO_PEER_COPY)
        endif()

        target_compile_definitions(ggml-hip PRIVATE GGML_HIP_NO_VMM)

        set(OLLAMA_HIP_INSTALL_DIR ${OLLAMA_INSTALL_DIR}/rocm)
        install(TARGETS ggml-hip
            RUNTIME_DEPENDENCIES
                DIRECTORIES ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR}
                PRE_INCLUDE_REGEXES hipblas rocblas amdhip64 rocsolver amd_comgr hsa-runtime64 rocsparse tinfo rocprofiler-register drm drm_amdgpu numa elf
                PRE_EXCLUDE_REGEXES ".*"
                POST_EXCLUDE_REGEXES "system32"
            RUNTIME DESTINATION ${OLLAMA_HIP_INSTALL_DIR} COMPONENT HIP
            LIBRARY DESTINATION ${OLLAMA_HIP_INSTALL_DIR} COMPONENT HIP
        )

        foreach(HIP_LIB_BIN_INSTALL_DIR IN ITEMS ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR})
            if(EXISTS ${HIP_LIB_BIN_INSTALL_DIR}/rocblas)
                install(DIRECTORY ${HIP_LIB_BIN_INSTALL_DIR}/rocblas DESTINATION ${OLLAMA_HIP_INSTALL_DIR} COMPONENT HIP)
                break()
            endif()
        endforeach()
    endif()
endif()

# 添加编译选项控制
option(GGML_CUDA_K80 "Enable Tesla K80 support" ON)
option(GGML_CUDA_DEBUG "Enable CUDA debug info" OFF)

if(GGML_CUDA_K80)
    add_definitions(-DGGML_CUDA_K80)
    set(CMAKE_CUDA_ARCHITECTURES "37")
endif()

# 添加测试支持
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# CUDA 配置
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# 检查 CUDA 工具链
find_package(CUDA 11.4 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# K80 特定优化选项
option(K80_OPTIMIZATIONS "Enable K80 specific optimizations" ON)
if(K80_OPTIMIZATIONS)
    add_definitions(-DK80_OPTIMIZATIONS)
    # K80 特定的编译标志
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        --maxrregcount=64 \
        --use_fast_math \
        --restrict \
        -Xptxas=-v \
        -Xcompiler=-fPIC \
        -Xcompiler=-Wall")
endif()

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 37)  # Tesla K80 的架构
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "37")

# 检查 cuBLAS 和 cuDNN
if(USE_CUBLAS)
    find_package(BLAS REQUIRED)
    add_definitions(-DUSE_CUBLAS)
endif()

if(USE_CUDNN)
    find_package(CUDNN REQUIRED)
    add_definitions(-DUSE_CUDNN)
endif()

# 内存管理优化
add_definitions(-DUSE_MANAGED_MEMORY)
add_definitions(-DUSE_PINNED_MEMORY)

# 添加源文件
add_subdirectory(gpu)
add_subdirectory(cuda)

# 主要目标
add_executable(ollama-k80
    main.cpp
    gpu/detector.cpp
    gpu/monitor.cpp
    gpu/cache_manager.cpp
)

# 链接 CUDA 库
target_link_libraries(ollama-k80 PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cudnn
    ${CUDA_LIBRARIES}
)

# 安装规则
install(TARGETS ollama-k80
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 安装配置文件
install(FILES config/k80_config.yaml
    DESTINATION etc/ollama
)

# 测试配置
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
