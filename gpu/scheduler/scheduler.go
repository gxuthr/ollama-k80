package scheduler

import (
    "context"
    "sync"
    "time"
    "container/heap"
)

// 任务优先级
type Priority int

const (
    PriorityLow Priority = iota
    PriorityNormal
    PriorityHigh
    PriorityCritical
)

// 任务状态
type TaskStatus int

const (
    TaskPending TaskStatus = iota
    TaskRunning
    TaskCompleted
    TaskFailed
)

// GPU 任务
type Task struct {
    ID          string
    Priority    Priority
    Status      TaskStatus
    Resources   ResourceRequest
    StartTime   time.Time
    Deadline    time.Time
    Handler     func(context.Context) error
}

// 资源请求
type ResourceRequest struct {
    MemoryBytes uint64
    ComputeUnits int
    MaxDuration time.Duration
}

// 优先级队列实现
type PriorityQueue []*Task

func (pq PriorityQueue) Len() int { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool {
    return pq[i].Priority > pq[j].Priority
}
func (pq PriorityQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }
func (pq *PriorityQueue) Push(x interface{}) { *pq = append(*pq, x.(*Task)) }
func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    *pq = old[0 : n-1]
    return item
}

// 调度器
type Scheduler struct {
    mu              sync.RWMutex
    queue           PriorityQueue
    resourceManager *ResourceManager
    runningTasks    map[string]*Task
    maxConcurrent   int
    metrics         *SchedulerMetrics
}

type SchedulerMetrics struct {
    QueueLength     int
    RunningTasks    int
    CompletedTasks  int
    FailedTasks     int
    AverageWaitTime time.Duration
}

func NewScheduler(rm *ResourceManager, maxConcurrent int) *Scheduler {
    s := &Scheduler{
        queue:           make(PriorityQueue, 0),
        resourceManager: rm,
        runningTasks:    make(map[string]*Task),
        maxConcurrent:   maxConcurrent,
        metrics:         &SchedulerMetrics{},
    }
    heap.Init(&s.queue)
    return s
}

// 提交任务
func (s *Scheduler) Submit(task *Task) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    heap.Push(&s.queue, task)
    s.metrics.QueueLength++

    // 尝试调度任务
    return s.trySchedule()
}

// 尝试调度任务
func (s *Scheduler) trySchedule() error {
    for s.queue.Len() > 0 && len(s.runningTasks) < s.maxConcurrent {
        task := heap.Pop(&s.queue).(*Task)
        
        // 检查资源是否可用
        if !s.resourceManager.CanAllocate(task.Resources) {
            heap.Push(&s.queue, task)
            return nil
        }

        // 分配资源并启动任务
        if err := s.startTask(task); err != nil {
            heap.Push(&s.queue, task)
            return err
        }
    }
    return nil
}

// 启动任务
func (s *Scheduler) startTask(task *Task) error {
    ctx, cancel := context.WithTimeout(context.Background(), task.Resources.MaxDuration)
    
    go func() {
        defer cancel()
        defer s.completeTask(task)

        if err := task.Handler(ctx); err != nil {
            task.Status = TaskFailed
            s.metrics.FailedTasks++
            return
        }

        task.Status = TaskCompleted
        s.metrics.CompletedTasks++
    }()

    s.runningTasks[task.ID] = task
    s.metrics.RunningTasks++
    return nil
} 
