cmake_minimum_required(VERSION 3.21)

# CUDA源文件
set(GGML_CUDA_SOURCES
    softmax.cu
    mmq.cu
    mmv.cu
    im2col.cu
)

# 创建CUDA库
add_library(ggml-cuda SHARED ${GGML_CUDA_SOURCES})

# 设置包含目录
target_include_directories(ggml-cuda
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 添加编译定义
target_compile_definitions(ggml-cuda
    PRIVATE
    GGML_CUDA_K80
    GGML_SHARED
)

# 设置K80特定的编译选项
set(K80_CUDA_FLAGS
    -arch=sm_37
    --use_fast_math
    --maxrregcount=64
    -Xptxas=-v,-O3
    -lineinfo
    --compiler-options -fPIC
    -D_FORCE_INLINES
)

# 设置编译选项
target_compile_options(ggml-cuda
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${K80_CUDA_FLAGS}>
)

# 链接依赖
target_link_libraries(ggml-cuda
    PUBLIC
    ggml
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)
