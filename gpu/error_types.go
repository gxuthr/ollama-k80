package gpu

import "fmt"

// 错误严重程度
type ErrorSeverity int

const (
    SeverityLow ErrorSeverity = iota
    SeverityMedium
    SeverityHigh
    SeverityCritical
)

// 错误类型
type ErrorType int

const (
    ErrorTypeOOM ErrorType = iota
    ErrorTypeHardware
    ErrorTypeDriver
    ErrorTypeCompute
    ErrorTypeMemory
)

// GPU 错误
type GPUError struct {
    Type        ErrorType
    Severity    ErrorSeverity
    Message     string
    RetryCount  int
    DeviceID    int
    MemoryInfo  *MemoryInfo
}

type MemoryInfo struct {
    Total     uint64
    Used      uint64
    Free      uint64
    Processes []ProcessInfo
}

type ProcessInfo struct {
    PID        int
    MemoryUsed uint64
    Name       string
}

func (e *GPUError) Error() string {
    return fmt.Sprintf("GPU Error [Type: %v, Severity: %v]: %s", e.Type, e.Severity, e.Message)
} 
