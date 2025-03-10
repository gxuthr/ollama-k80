package gpu

import (
    "fmt"
    "os/exec"
    "strconv"
    "strings"
    "time"
)

type GPUInfo struct {
    IsK80              bool
    ComputeCapability  string
    TotalMemory       uint64
    DriverVersion     string
    CUDAVersion      string
    ECCEnabled       bool
}

// DetectGPU 检测 GPU 信息并验证兼容性
func DetectGPU() (*GPUInfo, error) {
    // 使用 nvidia-smi 获取 GPU 信息
    cmd := exec.Command("nvidia-smi", "--query-gpu=gpu_name,memory.total,driver_version,compute_cap", "--format=csv,noheader")
    output, err := cmd.Output()
    if err != nil {
        return nil, fmt.Errorf("failed to detect GPU: %v", err)
    }

    info := parseGPUInfo(string(output))
    if !info.IsK80 {
        return nil, fmt.Errorf("GPU is not Tesla K80")
    }

    return info, nil
}

// ValidateGPURequirements 验证 GPU 是否满足运行要求
func ValidateGPURequirements(info *GPUInfo, config *K80Config) error {
    if info.TotalMemory < config.MinVRAMBytes {
        return fmt.Errorf("insufficient GPU memory: %d GB (required: %d GB)",
            info.TotalMemory/(1024*1024*1024), config.MinVRAMBytes/(1024*1024*1024))
    }

    if !isDriverCompatible(info.DriverVersion, config.RequiredDriverVersion) {
        return fmt.Errorf("incompatible driver version: %s (required: %s)",
            info.DriverVersion, config.RequiredDriverVersion)
    }

    return nil
}

// 添加错误恢复机制
type ErrorRecovery struct {
    MaxRetries      int
    CurrentRetry    int
    RetryDelay      time.Duration
    LastError       error
    FallbackToCPU   bool
}

func NewErrorRecovery(config *K80Config) *ErrorRecovery {
    return &ErrorRecovery{
        MaxRetries:    config.ErrorHandling.MaxRetries,
        RetryDelay:    time.Duration(config.ErrorHandling.RetryDelayMS) * time.Millisecond,
        FallbackToCPU: config.ErrorHandling.FallbackToCPU,
    }
}

func (er *ErrorRecovery) HandleError(err error) error {
    er.LastError = err
    if er.CurrentRetry >= er.MaxRetries {
        if er.FallbackToCPU {
            return er.switchToCPU()
        }
        return fmt.Errorf("max retries exceeded: %v", err)
    }

    er.CurrentRetry++
    time.Sleep(er.RetryDelay)
    return nil
}

func (er *ErrorRecovery) switchToCPU() error {
    // 实现 GPU 到 CPU 的切换逻辑
    return nil
} 
