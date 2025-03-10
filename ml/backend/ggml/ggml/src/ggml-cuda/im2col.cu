#include "im2col.cuh"

template <typename T>
static  __global__ void im2col_kernel(
        const float * x, T * dst, int64_t batch_offset,
        int64_t offset_delta, int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH, int64_t pelements, int64_t CHW,
        int s0, int s1, int p0, int p1, int d0, int d1) {
    __shared__ float tile[32][32];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    if (tx < 32 && ty < 32) {
        tile[ty][tx] = x[ty * IW + tx];
    }
    __syncthreads();

    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= pelements) {
        return;
    }

    const int64_t  ksize = OW * (KH > 1 ? KW : 1);
    const int64_t  kx = i / ksize;
    const int64_t  kd = kx * ksize;
    const int64_t  ky = (i - kd) / OW;
    const int64_t  ix = i % OW;

    const int64_t  oh = blockIdx.y;
    const int64_t  batch = blockIdx.z / IC;
    const int64_t  ic = blockIdx.z % IC;

    const int64_t iiw = ix * s0 + kx * d0 - p0;
    const int64_t iih = oh * s1 + ky * d1 - p1;

    const int64_t offset_dst =
        ((batch * OH + oh) * OW + ix) * CHW +
        (ic * (KW * KH) + ky * KW + kx);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        dst[offset_dst] = 0.0f;
    } else {
        const int64_t offset_src = ic * offset_delta + batch * batch_offset;
        dst[offset_dst] = tile[iih][iiw];
    }
}

template <typename T>
static void im2col_cuda(const float * x, T* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t batch, int64_t batch_offset, int64_t offset_delta,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {
    const int parallel_elements = OW * KW * KH;
    const int num_blocks = (parallel_elements + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    dim3 block_nums(num_blocks, OH, batch * IC);
    im2col_kernel<<<block_nums, CUDA_IM2COL_BLOCK_SIZE, 0, stream>>>(x, dst, batch_offset, offset_delta, IC, IW, IH, OH, OW, KW, KH, parallel_elements, (IC * KH * KW), s0, s1, p0, p1, d0, d1);
}

static void im2col_cuda_f16(const float * x, half * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t batch, int64_t batch_offset, int64_t offset_delta,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<half>(x, dst, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, offset_delta, s0, s1, p0, p1, d0, d1, stream);
}

static void im2col_cuda_f32(const float * x, float * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t batch, int64_t batch_offset, int64_t offset_delta,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<float>(x, dst, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, offset_delta, s0, s1, p0, p1, d0, d1, stream);
}

// K80优化的im2col实现
template <typename T>
static __global__ void k80_optimized_im2col_kernel(
    const float* x,
    T* dst,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w) {
    
    // 使用共享内存缓存输入数据
    __shared__ float shared_input[32][32];
    
    const int64_t thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int64_t output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int64_t output_size = output_h * output_w;
    
    // 协作加载到共享内存
    if(threadIdx.x < 32 && threadIdx.y < 32) {
        int h = blockIdx.y * 32 + threadIdx.y;
        int w = blockIdx.x * 32 + threadIdx.x;
        if(h < height && w < width) {
            shared_input[threadIdx.y][threadIdx.x] = x[h * width + w];
        }
    }
    __syncthreads();
    
    // 计算输出
    if(thread_idx < output_size) {
        const int64_t out_h = thread_idx / output_w;
        const int64_t out_w = thread_idx % output_w;
        
        for(int64_t c = 0; c < channels; ++c) {
            for(int64_t kh = 0; kh < kernel_h; ++kh) {
                for(int64_t kw = 0; kw < kernel_w; ++kw) {
                    const int64_t h = out_h * stride_h - pad_h + kh * dilation_h;
                    const int64_t w = out_w * stride_w - pad_w + kw * dilation_w;
                    
                    if(h >= 0 && h < height && w >= 0 && w < width) {
                        dst[((c * kernel_h + kh) * kernel_w + kw) * output_size + thread_idx] = 
                            shared_input[h - blockIdx.y * 32][w - blockIdx.x * 32];
                    } else {
                        dst[((c * kernel_h + kh) * kernel_w + kw) * output_size + thread_idx] = 0;
                    }
                }
            }
        }
    }
}

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW =         src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW =         src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW =         dst->ne[1];

    const size_t  delta_offset = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const int64_t batch        = src1->ne[is_2D ? 3 : 2];
    const size_t  batch_offset = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    if(dst->type == GGML_TYPE_F16) {
        im2col_cuda_f16(src1_d, (half *) dst_d, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, delta_offset, s0, s1, p0, p1, d0, d1, stream);
    } else {
        im2col_cuda_f32(src1_d, (float *) dst_d, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, delta_offset, s0, s1, p0, p1, d0, d1, stream);
    }
}
