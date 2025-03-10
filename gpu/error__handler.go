package gpu

import (
	"context"
	"time"
)

type ErrorHandler struct {
	maxRetries      int
	retryDelay      time.Duration
	fallbackEnabled bool
	onError         func(error)
	metrics         *MetricsCollector
}

type ErrorHandlerOption func(*ErrorHandler)

func WithMaxRetries(n int) ErrorHandlerOption {
	return func(h *ErrorHandler) {
		h.maxRetries = n
	}
}

func WithRetryDelay(d time.Duration) ErrorHandlerOption {
	return func(h *ErrorHandler) {
		h.retryDelay = d
	}
}

func NewErrorHandler(opts ...ErrorHandlerOption) *ErrorHandler {
	h := &ErrorHandler{
		maxRetries:      3,
		retryDelay:      time.Second,
		fallbackEnabled: true,
	}
	
	for _, opt := range opts {
		opt(h)
	}
	
	return h
}

func (h *ErrorHandler) Handle(ctx context.Context, err error) error {
	if gpuErr, ok := err.(*GPUError); ok {
		switch gpuErr.Type {
		case ErrorTypeOOM:
			return h.handleOOM(ctx, gpuErr)
		case ErrorTypeHardware:
			return h.handleHardwareError(ctx, gpuErr)
		case ErrorTypeDriver:
			return h.handleDriverError(ctx, gpuErr)
		}
	}
	return h.handleGenericError(ctx, err)
}

func (h *ErrorHandler) handleOOM(ctx context.Context, err *GPUError) error {
	// 1. 尝试释放缓存
	if err := h.releaseMemory(ctx); err != nil {
		return err
	}
	
	// 2. 降低批处理大小
	if err := h.reduceBatchSize(ctx); err != nil {
		return err
	}
	
	// 3. 如果还是失败，考虑切换到 CPU
	if h.fallbackEnabled {
		return h.switchToCPU(ctx)
	}
	
	return err
}

func (h *ErrorHandler) releaseMemory(ctx context.Context) error {
	// 实现内存释放逻辑
	return nil
}

func (h *ErrorHandler) reduceBatchSize(ctx context.Context) error {
	// 实现批处理大小调整
	return nil
}

func (h *ErrorHandler) switchToCPU(ctx context.Context) error {
	// 实现 GPU 到 CPU 的切换
	return nil
} 
