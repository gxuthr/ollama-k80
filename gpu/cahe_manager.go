package gpu

import (
	"sync"
	"container/list"
)

type CacheManager struct {
	mu           sync.RWMutex
	maxSizeBytes uint64
	currentSize  uint64
	cache        *list.List
	lookup       map[string]*list.Element
	config       *K80Config
}

type CacheEntry struct {
	key         string
	sizeBytes   uint64
	data        interface{}
}

func NewCacheManager(config *K80Config) *CacheManager {
	return &CacheManager{
		maxSizeBytes: uint64(config.Caching.MaxCacheSizeGB) * 1024 * 1024 * 1024,
		cache:        list.New(),
		lookup:       make(map[string]*list.Element),
		config:       config,
	}
}

func (cm *CacheManager) Add(key string, data interface{}, sizeBytes uint64) bool {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// 检查是否超过缓存大小限制
	if sizeBytes > cm.maxSizeBytes {
		return false
	}

	// 如果需要，清理空间
	for cm.currentSize+sizeBytes > cm.maxSizeBytes {
		if !cm.removeOldest() {
			return false
		}
	}

	// 添加新项
	entry := &CacheEntry{
		key:       key,
		sizeBytes: sizeBytes,
		data:      data,
	}
	element := cm.cache.PushFront(entry)
	cm.lookup[key] = element
	cm.currentSize += sizeBytes

	return true
}

func (cm *CacheManager) Get(key string) (interface{}, bool) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if element, ok := cm.lookup[key]; ok {
		cm.cache.MoveToFront(element)
		return element.Value.(*CacheEntry).data, true
	}

	return nil, false
}

func (cm *CacheManager) removeOldest() bool {
	if element := cm.cache.Back(); element != nil {
		entry := element.Value.(*CacheEntry)
		cm.cache.Remove(element)
		delete(cm.lookup, entry.key)
		cm.currentSize -= entry.sizeBytes
		return true
	}
	return false
} 
