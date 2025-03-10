package gpu

import (
    "context"
    "sync"
    "time"
)

// 资源类型
type ResourceType int

const (
    ResourceMemory ResourceType = iota
    ResourceCompute
    ResourceBandwidth
)

// 资源限制
type ResourceLimits struct {
    MaxMemoryBytes     uint64
    MaxComputeThreads int
    MaxBandwidthMBps  uint64
    ReservationLimit  float64  // 预留资源比例
}

// 资源使用统计
type ResourceStats struct {
    Used      uint64
    Reserved  uint64
    Available uint64
    Peak      uint64
    LastGC    time.Time
}

// 资源管理器
type ResourceManager struct {
    mu          sync.RWMutex
    resources   map[ResourceType]*ResourceStats
    limits      ResourceLimits
    monitor     *PerformanceMonitor
    autoTuner   *AutoTuner
    gcInterval  time.Duration
}

func NewResourceManager(config *K80Config, monitor *PerformanceMonitor) *ResourceManager {
    rm := &ResourceManager{
        resources: make(map[ResourceType]*ResourceStats),
        limits: ResourceLimits{
            MaxMemoryBytes:    uint64(config.MaxVRAMBytes),
            MaxComputeThreads: config.Performance.MaxComputeThreads,
            MaxBandwidthMBps:  config.Performance.MaxBandwidthMBps,
            ReservationLimit:  0.2, // 预留20%资源
        },
        monitor:    monitor,
        gcInterval: 5 * time.Minute,
    }

    // 初始化资源统计
    rm.resources[ResourceMemory] = &ResourceStats{
        Available: rm.limits.MaxMemoryBytes,
    }
    
    return rm
}

// 资源分配请求
type AllocationRequest struct {
    Type     ResourceType
    Size     uint64
    Priority int
    Timeout  time.Duration
}

// 分配资源
func (rm *ResourceManager) Allocate(ctx context.Context, req AllocationRequest) error {
    rm.mu.Lock()
    defer rm.mu.Unlock()

    stats := rm.resources[req.Type]
    if stats == nil {
        return ErrInvalidResourceType
    }

    // 检查是否有足够资源
    if !rm.hasAvailableResource(req) {
        // 尝试回收资源
        if err := rm.garbageCollect(ctx, req.Type); err != nil {
            return err
        }
        
        // 再次检查资源
        if !rm.hasAvailableResource(req) {
            return ErrInsufficientResources
        }
    }

    // 分配资源
    stats.Used += req.Size
    stats.Peak = max(stats.Peak, stats.Used)

    return nil
}

// 释放资源
func (rm *ResourceManager) Release(ctx context.Context, req AllocationRequest) {
    rm.mu.Lock()
    defer rm.mu.Unlock()

    if stats := rm.resources[req.Type]; stats != nil {
        stats.Used = max(0, stats.Used-req.Size)
    }
}

// 资源回收
func (rm *ResourceManager) garbageCollect(ctx context.Context, resourceType ResourceType) error {
    stats := rm.resources[resourceType]
    if stats == nil {
        return ErrInvalidResourceType
    }

    switch resourceType {
    case ResourceMemory:
        return rm.collectMemory(ctx)
    case ResourceCompute:
        return rm.optimizeCompute(ctx)
    }

    return nil
}

// 内存回收
func (rm *ResourceManager) collectMemory(ctx context.Context) error {
    // 1. 清理缓存
    if err := rm.clearCache(); err != nil {
        return err
    }

    // 2. 压缩内存
    if err := rm.compressMemory(); err != nil {
        return err
    }

    // 3. 调整批处理大小
    if rm.autoTuner != nil {
        rm.autoTuner.AdjustBatchSize(0.8) // 减少20%
    }

    return nil
}

// 启动资源监控
func (rm *ResourceManager) StartMonitoring(ctx context.Context) {
    go func() {
        ticker := time.NewTicker(rm.gcInterval)
        defer ticker.Stop()

        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                rm.periodicGC(ctx)
            }
        }
    }()
} 
