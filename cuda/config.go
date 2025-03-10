package cuda

import "runtime"

// K80CUDAConfig 定义 K80 的 CUDA 配置
type K80CUDAConfig struct {
    ArchitectureFlags []string
    OptimizationFlags []string
    MemoryFlags       []string
}

// NewK80CUDAConfig 返回针对 K80 优化的 CUDA 配置
func NewK80CUDAConfig() *K80CUDAConfig {
    return &K80CUDAConfig{
        ArchitectureFlags: []string{
            "-arch=sm_37",              // K80 的计算能力
            "-gencode=arch=compute_37", // 生成 PTX 代码
        },
        OptimizationFlags: []string{
            "--use_fast_math",          // 使用快速数学库
            "--maxrregcount=64",        // 限制寄存器使用
            "-O3",                      // 最高优化级别
        },
        MemoryFlags: []string{
            "--default-stream=per-thread", // 每线程流
            "--restrict",                  // 启用限制指针优化
        },
    }
}

// GetCompileFlags 返回编译标志
func (c *K80CUDAConfig) GetCompileFlags() []string {
    flags := make([]string, 0)
    flags = append(flags, c.ArchitectureFlags...)
    flags = append(flags, c.OptimizationFlags...)
    flags = append(flags, c.MemoryFlags...)
    return flags
} 
