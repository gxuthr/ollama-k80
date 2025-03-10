package gpu

import (
	"context"
	"math"
	"sync"
	"time"
)

type Parameter struct {
	Name    string
	Min     float64
	Max     float64
	Current float64
	Step    float64
}

type AutoTuner struct {
	mu          sync.RWMutex
	parameters  map[string]*Parameter
	history     []TuningResult
	monitor     *PerformanceMonitor
	config      *K80Config
}

type TuningResult struct {
	Params    map[string]float64
	Score     float64
	Timestamp time.Time
}

func NewAutoTuner(config *K80Config, monitor *PerformanceMonitor) *AutoTuner {
	return &AutoTuner{
		parameters: map[string]*Parameter{
			"batch_size": {
				Name:    "batch_size",
				Min:     32,
				Max:     512,
				Current: float64(config.DefaultBatchSize),
				Step:    32,
			},
			"memory_fraction": {
				Name:    "memory_fraction",
				Min:     0.5,
				Max:     0.95,
				Current: 0.8,
				Step:    0.05,
			},
		},
		monitor: monitor,
		config:  config,
	}
}

func (at *AutoTuner) Start(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if err := at.Tune(ctx); err != nil {
					continue
				}
			}
		}
	}()
}

func (at *AutoTuner) Tune(ctx context.Context) error {
	// 获取当前性能指标
	metrics := at.monitor.GetLatestMetrics()
	
	// 计算当前配置的得分
	currentScore := at.calculateScore(metrics)
	
	// 尝试优化每个参数
	for name, param := range at.parameters {
		// 尝试增加参数值
		if newScore := at.tryParameter(name, param.Current+param.Step); newScore > currentScore {
			at.updateParameter(name, param.Current+param.Step)
			currentScore = newScore
		} else if newScore := at.tryParameter(name, param.Current-param.Step); newScore > currentScore {
			// 尝试减少参数值
			at.updateParameter(name, param.Current-param.Step)
			currentScore = newScore
		}
	}
	
	return nil
}

func (at *AutoTuner) calculateScore(metrics PerformanceMetrics) float64 {
	// 计算性能得分
	// 考虑吞吐量、延迟、内存使用等因素
	throughputWeight := 0.4
	latencyWeight := 0.3
	memoryWeight := 0.3
	
	normalizedThroughput := metrics.Throughput / at.config.MaxThroughput
	normalizedLatency := math.Min(1.0, at.config.TargetLatency.Seconds()/metrics.Latency.Seconds())
	normalizedMemory := 1.0 - (metrics.MemoryUsage / float64(at.config.MaxVRAMBytes))
 
	return (normalizedThroughput * throughputWeight) +
		   (normalizedLatency * latencyWeight) +
		   (normalizedMemory * memoryWeight)
}

func (at *AutoTuner) updateParameter(name string, value float64) {
	at.mu.Lock()
	defer at.mu.Unlock()
	
	if param, exists := at.parameters[name]; exists {
		// 确保值在有效范围内
		value = math.Max(param.Min, math.Min(param.Max, value))
		param.Current = value
		
		// 更新配置
		at.applyParameter(name, value)
	}
}

func (at *AutoTuner) applyParameter(name string, value float64) {
	switch name {
	case "batch_size":
		at.config.DefaultBatchSize = int(value)
	case "memory_fraction":
		at.config.MemoryOptimization.MaxMemoryFraction = value
	}
} 
