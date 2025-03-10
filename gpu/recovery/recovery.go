package recovery

import (
    "context"
    "sync"
    "time"
)

// 故障类型
type FailureType int

const (
    FailureOOM FailureType = iota
    FailureHardware
    FailureDriver
    FailureCompute
)

// 恢复策略
type RecoveryStrategy interface {
    CanHandle(failure *Failure) bool
    Recover(ctx context.Context, failure *Failure) error
    Priority() int
}

// 故障信息
type Failure struct {
    Type      FailureType
    Error     error
    Timestamp time.Time
    Device    string
    Context   map[string]interface{}
}

// 恢复尝试记录
type RecoveryAttempt struct {
    Failure   *Failure
    Strategy  string
    Success   bool
    Duration  time.Duration
    Timestamp time.Time
}

// 故障恢复管理器
type RecoveryManager struct {
    mu         sync.RWMutex
    strategies []RecoveryStrategy
    history    []RecoveryAttempt
    maxRetries int
    monitor    *PerformanceMonitor
}

// OOM 恢复策略
type OOMRecoveryStrategy struct {
    resourceManager *ResourceManager
    autoTuner      *AutoTuner
}

func (s *OOMRecoveryStrategy) CanHandle(failure *Failure) bool {
    return failure.Type == FailureOOM
}

func (s *OOMRecoveryStrategy) Recover(ctx context.Context, failure *Failure) error {
    // 1. 清理缓存
    if err := s.resourceManager.ClearCache(); err != nil {
        return err
    }

    // 2. 降低批处理大小
    s.autoTuner.AdjustBatchSize(0.7) // 降低30%

    // 3. 压缩内存
    return s.resourceManager.CompressMemory()
}

func (s *OOMRecoveryStrategy) Priority() int {
    return 100 // 高优先级
}

// 硬件故障恢复策略
type HardwareRecoveryStrategy struct {
    deviceManager *DeviceManager
}

func (s *HardwareRecoveryStrategy) Recover(ctx context.Context, failure *Failure) error {
    // 1. 重置设备
    if err := s.deviceManager.ResetDevice(failure.Device); err != nil {
        return err
    }

    // 2. 重新初始化
    return s.deviceManager.ReinitializeDevice(failure.Device)
} 
