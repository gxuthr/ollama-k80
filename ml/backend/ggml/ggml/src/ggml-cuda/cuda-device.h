// 添加K80设备检测与配置
struct ggml_cuda_device_config {
    int device_id;
    size_t max_mem;
    bool no_mul_mat_q;
    bool no_flash_attn;
    cudaStream_t stream;
};

static bool is_k80_device(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    return prop.major == 3 && prop.minor == 7;
}

// 添加CUDA版本检查
static bool check_cuda_version_for_k80() {
    int cuda_version;
    cudaRuntimeGetVersion(&cuda_version);
    int major = cuda_version / 1000;
    int minor = (cuda_version % 1000) / 10;
    
    if (major > 11 || (major == 11 && minor > 4)) {
        fprintf(stderr, "Warning: Tesla K80 only supports CUDA up to 11.4\n");
        return false;
    }
    if (major < 6 || (major == 6 && minor < 5)) {
        fprintf(stderr, "Warning: Tesla K80 requires CUDA 6.5 or higher\n");
        return false;
    }
    return true;
}

static void configure_k80_device(ggml_cuda_device_config & config) {
    if (is_k80_device(config.device_id)) {
        if (!check_cuda_version_for_k80()) {
            config.no_mul_mat_q = true;
            config.no_flash_attn = true;
            return;
        }
        // K80特定配置
        config.no_flash_attn = true;
        config.max_mem = 11*1024*1024*1024ull;
    }
}

// 添加运行时检查
static bool validate_k80_environment() {
    // 检查CUDA运行时
    cudaError_t runtime_error = cudaRuntimeGetVersion(nullptr);
    if (runtime_error != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error: %s\n", cudaGetErrorString(runtime_error));
        return false;
    }

    // 检查设备内存
    size_t free_mem, total_mem;
    cudaError_t mem_error = cudaMemGetInfo(&free_mem, &total_mem);
    if (mem_error != cudaSuccess) {
        fprintf(stderr, "Memory check error: %s\n", cudaGetErrorString(mem_error));
        return false;
    }

    // K80要求至少8GB可用内存
    if (free_mem < 8ULL*1024*1024*1024) {
        fprintf(stderr, "Warning: Less than 8GB available memory\n");
        return false;
    }

    return true;
}

// K80设备配置管理
struct k80_device_manager {
    static constexpr int WARP_SIZE = 32;
    static constexpr int MAX_THREADS_PER_BLOCK = 1024;
    static constexpr int MAX_SHARED_MEMORY = 48 * 1024;
    static constexpr int NUM_SMS = 13;
    
    struct config {
        bool use_tensor_cores;
        bool prefer_shared_memory;
        int max_registers_per_thread;
        size_t max_shared_memory_per_block;
    } current;
    
    void optimize_for_k80() {
        // 设置K80优化配置
        current.use_tensor_cores = false;  // K80不支持tensor cores
        current.prefer_shared_memory = true;
        current.max_registers_per_thread = 64;
        current.max_shared_memory_per_block = MAX_SHARED_MEMORY;
        
        apply_config();
    }
    
private:
    void apply_config() {
        // 应用缓存配置
        cudaFuncCache cache_config = 
            current.prefer_shared_memory ? 
            cudaFuncCachePreferShared : cudaFuncCachePreferL1;
        CUDA_CHECK_K80(cudaDeviceSetCacheConfig(cache_config),
                      "Failed to set cache config");
                      
        // 应用其他设备特定配置
        apply_memory_config();
        apply_compute_config();
    }
};

// K80性能优化配置
struct k80_perf_config {
    // 缓存配置
    static void optimize_cache() {
        cudaFuncCache pref = cudaFuncCachePreferL1;
        CUDA_CHECK_THROW(cudaDeviceSetCacheConfig(pref));
    }

    // 共享内存配置
    static void optimize_shared_memory() {
        cudaSharedMemConfig config = cudaSharedMemBankSizeEightByte;
        CUDA_CHECK_THROW(cudaDeviceSetSharedMemConfig(config));
    }
    
    // 内存访问优化
    static void optimize_memory_access() {
        // 设置最大L2缓存大小
        size_t size = 128*1024; // 128KB
        CUDA_CHECK_THROW(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, size));
    }

    // 应用所有优化
    static void apply_all() {
        optimize_cache();
        optimize_shared_memory();
        optimize_memory_access();
    }
};

// K80内存管理器
struct k80_memory_manager {
    static constexpr size_t K80_MEMORY_ALIGNMENT = 256;
    static constexpr size_t K80_MAX_MEMORY = 11ULL * 1024 * 1024 * 1024; // 11GB
    static constexpr size_t K80_MEMORY_RESERVE = 512ULL * 1024 * 1024;   // 512MB保留

    struct memory_block {
        void* ptr;
        size_t size;
        bool in_use;
    };

    std::vector<memory_block> blocks;

    void* allocate(size_t size) {
        size = align_size(size);
        
        // 尝试找到合适的空闲块
        for (auto& block : blocks) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }

        // 分配新块
        void* ptr = nullptr;
        CUDA_CHECK_K80(cudaMalloc(&ptr, size), "Failed to allocate memory");
        blocks.push_back({ptr, size, true});
        return ptr;
    }

    void free(void* ptr) {
        for (auto& block : blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }

    void cleanup() {
        for (const auto& block : blocks) {
            CUDA_CHECK_K80(cudaFree(block.ptr), "Failed to free memory");
        }
        blocks.clear();
    }

private:
    size_t align_size(size_t size) {
        return (size + K80_MEMORY_ALIGNMENT - 1) & ~(K80_MEMORY_ALIGNMENT - 1);
    }
};

// K80性能优化配置增强
struct k80_perf_optimizer {
    static void optimize_for_compute() {
        cudaFuncCache pref = cudaFuncCachePreferL1;
        CUDA_CHECK_K80_DETAILED(cudaDeviceSetCacheConfig(pref),
            "Failed to set cache preference to L1");
    }

    static void optimize_for_memory() {
        cudaFuncCache pref = cudaFuncCachePreferShared;
        CUDA_CHECK_K80_DETAILED(cudaDeviceSetCacheConfig(pref),
            "Failed to set cache preference to shared memory");
            
        cudaSharedMemConfig config = cudaSharedMemBankSizeEightByte;
        CUDA_CHECK_K80_DETAILED(cudaDeviceSetSharedMemConfig(config),
            "Failed to set shared memory bank size");
    }

    static void optimize_for_bandwidth() {
        // 设置最大L2缓存大小
        size_t size = 128*1024; // 128KB
        CUDA_CHECK_K80_DETAILED(cudaDeviceSetLimit(
            cudaLimitMaxL2FetchGranularity, size),
            "Failed to set L2 fetch granularity");
            
        // 启用异步预取
        CUDA_CHECK_K80_DETAILED(cudaDeviceSetLimit(
            cudaLimitPrefetchQueueSize, 8),
            "Failed to set prefetch queue size");
    }

    static void apply_optimal_config() {
        optimize_for_compute();
        optimize_for_memory();
        optimize_for_bandwidth();
        
        // 设置设备标志
        CUDA_CHECK_K80_DETAILED(cudaSetDeviceFlags(
            cudaDeviceMapHost |          // 启用统一内存
            cudaDeviceScheduleSpin |     // 使用自旋等待
            cudaDeviceLmemResizeToMax    // 最大化本地内存
        ), "Failed to set device flags");
    }
}; 
