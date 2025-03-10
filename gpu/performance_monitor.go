package gpu

import (
	"context"
	"sync"
	"time"
)

type PerformanceMetrics struct {
	Timestamp       time.Time
	DeviceID       int
	GPUUtilization float64
	MemoryUsage    float64
	Temperature    float64
	PowerUsage     float64
	MemoryBandwidth float64
	ComputeEfficiency float64
	BatchSize      int
	Throughput     float64
	Latency        time.Duration
	ErrorCount     int
}

type PerformanceMonitor struct {
	mu              sync.RWMutex
	metrics         []PerformanceMetrics
	config          *K80Config
	alertThresholds map[string]float64
	callbacks       map[string][]func(PerformanceMetrics)
	stopChan        chan struct{}
}

func NewPerformanceMonitor(config *K80Config) *PerformanceMonitor {
	return &PerformanceMonitor{
		metrics:  make([]PerformanceMetrics, 0),
		config:   config,
		alertThresholds: map[string]float64{
			"temperature":     80.0,  // 80°C
			"memory_usage":    0.95,  // 95%
			"gpu_utilization": 0.95,  // 95%
		},
		callbacks: make(map[string][]func(PerformanceMetrics)),
		stopChan:  make(chan struct{}),
	}
}

func (pm *PerformanceMonitor) Start(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(time.Duration(pm.config.Monitoring.IntervalSeconds) * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				metrics, err := pm.collectMetrics()
				if err != nil {
					continue
				}

				pm.mu.Lock()
				pm.metrics = append(pm.metrics, metrics)
				pm.mu.Unlock()

				// 检查阈值并触发回调
				pm.checkThresholds(metrics)
			}
		}
	}()
}

func (pm *PerformanceMonitor) RegisterCallback(metric string, callback func(PerformanceMetrics)) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	
	if _, exists := pm.callbacks[metric]; !exists {
		pm.callbacks[metric] = make([]func(PerformanceMetrics), 0)
	}
	pm.callbacks[metric] = append(pm.callbacks[metric], callback)
}

func (pm *PerformanceMonitor) checkThresholds(metrics PerformanceMetrics) {
	// 检查温度
	if metrics.Temperature > pm.alertThresholds["temperature"] {
		pm.triggerCallbacks("temperature", metrics)
	}

	// 检查内存使用
	if metrics.MemoryUsage > pm.alertThresholds["memory_usage"] {
		pm.triggerCallbacks("memory_usage", metrics)
	}

	// 检查 GPU 利用率
	if metrics.GPUUtilization > pm.alertThresholds["gpu_utilization"] {
		pm.triggerCallbacks("gpu_utilization", metrics)
	}
}

func (pm *PerformanceMonitor) triggerCallbacks(metric string, metrics PerformanceMetrics) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	if callbacks, exists := pm.callbacks[metric]; exists {
		for _, callback := range callbacks {
			go callback(metrics)
		}
	}
} 
