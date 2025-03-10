// K80特定的kernel配置
#define CUDA_K80_MAX_THREADS_PER_BLOCK 1024
#define CUDA_K80_MAX_SHARED_MEMORY 48*1024
#define CUDA_K80_WARP_SIZE 32
#define CUDA_K80_NUM_SMS 13

// 根据K80硬件特性调整block size
static inline int get_block_size_k80(int min_threads) {
    int block_size = CUDA_K80_WARP_SIZE;
    while (block_size < min_threads && block_size < CUDA_K80_MAX_THREADS_PER_BLOCK) {
        block_size *= 2;
    }
    return block_size;
} 
