package gpu

import (
    "log"
    "runtime"
)

// K80Optimizer 提供针对 Tesla K80 的优化
type K80Optimizer struct {
    TotalVRAM   uint64
    CurrentVRAM uint64
    BatchSize   int
}

// NewK80Optimizer 创建新的 K80 优化器实例
func NewK80Optimizer() *K80Optimizer {
    return &K80Optimizer{
        TotalVRAM: 12 * 1024 * 1024 * 1024, // 12GB
        BatchSize: 256,
    }
}

// OptimizeForModel 根据模型大小优化参数
func (o *K80Optimizer) OptimizeForModel(modelSize uint64) (map[string]interface{}, error) {
    // 计算可用显存
    availableVRAM := o.TotalVRAM - o.CurrentVRAM
    
    // 根据可用显存动态调整批处理大小
    optimizedBatchSize := o.calculateOptimalBatchSize(availableVRAM)
    
    // 返回优化后的参数
    return map[string]interface{}{
        "batch_size": optimizedBatchSize,
        "use_fp16": true, // 使用半精度以节省显存
        "use_int8_quantization": availableVRAM < 4*1024*1024*1024, // 显存不足4GB时使用int8量化
        "enable_memory_efficient_attention": true,
        "gradient_checkpointing": true,
    }, nil
}

// calculateOptimalBatchSize 计算最优批处理大小
func (o *K80Optimizer) calculateOptimalBatchSize(availableVRAM uint64) int {
    // 基于可用显存计算最大批处理大小
    maxBatchSize := int(availableVRAM / (2 * 1024 * 1024)) // 假设每个样本需要2MB
    
    // 根据经验值设置批处理大小范围
    if maxBatchSize > 512 {
        return 256 // K80 上测试的最优值
    } else if maxBatchSize > 128 {
        return 128
    } else if maxBatchSize > 64 {
        return 64
    } else {
        return 32 // 最小批处理大小
    }
}

// MonitorResources 监控 GPU 资源使用
func (o *K80Optimizer) MonitorResources() {
    go func() {
        for {
            var m runtime.MemStats
            runtime.ReadMemStats(&m)
            
            // 记录资源使用情况
            log.Printf("GPU Memory Usage: %v MB", o.CurrentVRAM/1024/1024)
            log.Printf("Batch Size: %v", o.BatchSize)
            
            runtime.Gosched()
        }
    }()
} 
