# K80 特定的优化设置
function(configure_k80_optimizations target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "Target ${target} does not exist")
    endif()

    # 设置 K80 特定的编译选项
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --maxrregcount=64
            --use_fast_math
            --restrict
            -Xptxas=-v
        >
    )

    # 设置 K80 特定的定义
    target_compile_definitions(${target} PRIVATE
        K80_GPU
        USE_MANAGED_MEMORY
        USE_PINNED_MEMORY
        MAX_BATCH_SIZE=256
        SHARED_MEMORY_SIZE=48
    )

    # 设置内存对齐
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

# 配置内存管理
function(configure_memory_management target)
    target_compile_definitions(${target} PRIVATE
        CUDA_MALLOC_ASYNC
        CUDA_MEMORY_POOL
        CUDA_STREAM_ORDER_MEMORY
    )
endfunction()

# 配置性能分析
function(configure_profiling target)
    if(ENABLE_PROFILING)
        target_compile_definitions(${target} PRIVATE
            ENABLE_NVTX
            ENABLE_CUDA_PROFILING
        )
        target_link_libraries(${target} PRIVATE
            ${CUDA_NVTX_LIBRARY}
        )
    endif()
endfunction() 
