"""
Performance Optimizer for Feature Mapping Pipeline
Implements caching, connection pooling, and optimization strategies
to meet sub-10ms mapping requirements.
"""

import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from fraudguard.entity.feature_mapping_entity import (
    UserTransactionInput, MappingResult, MerchantCategory, LocationRisk, SpendingPattern
)
from fraudguard.logger import fraud_logger


@dataclass
class CacheEntry:
    """Cache entry for mapping results"""
    result: Dict[str, Any]
    timestamp: float
    hit_count: int = 0
    
    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.timestamp > ttl_seconds


class MemoryCache:
    """In-memory LRU cache for mapping results"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _create_cache_key(self, user_input: UserTransactionInput, mapper_type: str) -> str:
        """Create cache key from user input and mapper type"""
        input_dict = {
            'amount': round(user_input.transaction_amount, 2),
            'merchant': user_input.merchant_category.value,
            'hour': user_input.time_context.hour_of_day,
            'day': user_input.time_context.day_of_week,
            'weekend': user_input.time_context.is_weekend,
            'location': user_input.location_risk.value,
            'spending': user_input.spending_pattern.value,
            'mapper': mapper_type
        }
        
        key_str = json.dumps(input_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, user_input: UserTransactionInput, mapper_type: str) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        cache_key = self._create_cache_key(user_input, mapper_type)
        
        with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check if expired
                if entry.is_expired(self.ttl_seconds):
                    del self.cache[cache_key]
                    if cache_key in self.access_order:
                        self.access_order.remove(cache_key)
                    self.misses += 1
                    return None
                
                # Update access order and hit count
                entry.hit_count += 1
                if cache_key in self.access_order:
                    self.access_order.remove(cache_key)
                self.access_order.append(cache_key)
                
                self.hits += 1
                return entry.result.copy()
            
            self.misses += 1
            return None
    
    def put(self, user_input: UserTransactionInput, mapper_type: str, result: Dict[str, Any]):
        """Store result in cache"""
        cache_key = self._create_cache_key(user_input, mapper_type)
        
        with self.lock:
            # Remove if already exists
            if cache_key in self.cache:
                if cache_key in self.access_order:
                    self.access_order.remove(cache_key)
            
            # Check if we need to evict
            while len(self.cache) >= self.max_size:
                if self.access_order:
                    oldest_key = self.access_order.pop(0)
                    if oldest_key in self.cache:
                        del self.cache[oldest_key]
                        self.evictions += 1
                else:
                    break
            
            # Add new entry
            entry = CacheEntry(
                result=result.copy(),
                timestamp=time.time()
            )
            self.cache[cache_key] = entry
            self.access_order.append(cache_key)
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate_percent': round(hit_rate, 2),
                'total_requests': total_requests
            }


class RedisCache:
    """Redis-based cache for distributed caching"""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 ttl_seconds: int = 3600):
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: pip install redis")
        
        self.ttl_seconds = ttl_seconds
        self.redis_client = None
        
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1
            )
            
            # Test connection
            self.redis_client.ping()
            fraud_logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            fraud_logger.warning(f"Redis cache initialization failed: {e}")
            self.redis_client = None
    
    def _create_cache_key(self, user_input: UserTransactionInput, mapper_type: str) -> str:
        """Create cache key from user input and mapper type"""
        input_dict = {
            'amount': round(user_input.transaction_amount, 2),
            'merchant': user_input.merchant_category.value,
            'hour': user_input.time_context.hour_of_day,
            'day': user_input.time_context.day_of_week,
            'weekend': user_input.time_context.is_weekend,
            'location': user_input.location_risk.value,
            'spending': user_input.spending_pattern.value,
            'mapper': mapper_type
        }
        
        key_str = json.dumps(input_dict, sort_keys=True)
        return f"fraudguard:mapping:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def get(self, user_input: UserTransactionInput, mapper_type: str) -> Optional[Dict[str, Any]]:
        """Get cached result from Redis"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._create_cache_key(user_input, mapper_type)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            fraud_logger.warning(f"Redis cache get error: {e}")
            return None
    
    def put(self, user_input: UserTransactionInput, mapper_type: str, result: Dict[str, Any]):
        """Store result in Redis cache"""
        if not self.redis_client:
            return
        
        try:
            cache_key = self._create_cache_key(user_input, mapper_type)
            cached_data = json.dumps(result, default=str)
            
            self.redis_client.setex(cache_key, self.ttl_seconds, cached_data)
            
        except Exception as e:
            fraud_logger.warning(f"Redis cache put error: {e}")
    
    def clear(self):
        """Clear all cache entries"""
        if not self.redis_client:
            return
        
        try:
            keys = self.redis_client.keys("fraudguard:mapping:*")
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            fraud_logger.warning(f"Redis cache clear error: {e}")


class RequestQueue:
    """Request queue for handling high-load scenarios"""
    
    def __init__(self, max_size: int = 1000, max_workers: int = 4):
        self.max_size = max_size
        self.max_workers = max_workers
        self.request_queue = queue.Queue(maxsize=max_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_requests = 0
        self.lock = threading.Lock()
        
        # Statistics
        self.processed_requests = 0
        self.failed_requests = 0
        self.queue_full_errors = 0
    
    def submit_request(self, func, *args, **kwargs) -> Optional[Any]:
        """Submit request to queue"""
        try:
            if self.request_queue.full():
                self.queue_full_errors += 1
                return None
            
            future = self.executor.submit(func, *args, **kwargs)
            
            with self.lock:
                self.active_requests += 1
            
            try:
                result = future.result(timeout=30)  # 30 second timeout
                self.processed_requests += 1
                return result
            except Exception as e:
                self.failed_requests += 1
                fraud_logger.error(f"Request processing failed: {e}")
                return None
            finally:
                with self.lock:
                    self.active_requests -= 1
                    
        except Exception as e:
            fraud_logger.error(f"Request submission failed: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            return {
                'queue_size': self.request_queue.qsize(),
                'max_queue_size': self.max_size,
                'active_requests': self.active_requests,
                'max_workers': self.max_workers,
                'processed_requests': self.processed_requests,
                'failed_requests': self.failed_requests,
                'queue_full_errors': self.queue_full_errors
            }


class PerformanceOptimizer:
    """Main performance optimizer for the feature mapping pipeline"""
    
    def __init__(self, 
                 enable_memory_cache: bool = True,
                 enable_redis_cache: bool = False,
                 enable_request_queue: bool = True,
                 memory_cache_size: int = 1000,
                 redis_config: Optional[Dict[str, Any]] = None):
        
        self.enable_memory_cache = enable_memory_cache
        self.enable_redis_cache = enable_redis_cache
        self.enable_request_queue = enable_request_queue
        
        # Initialize caches
        self.memory_cache = None
        self.redis_cache = None
        self.request_queue = None
        
        if enable_memory_cache:
            self.memory_cache = MemoryCache(max_size=memory_cache_size)
            fraud_logger.info("Memory cache initialized")
        
        if enable_redis_cache and REDIS_AVAILABLE:
            try:
                redis_config = redis_config or {}
                self.redis_cache = RedisCache(**redis_config)
            except Exception as e:
                fraud_logger.warning(f"Redis cache initialization failed: {e}")
                self.enable_redis_cache = False
        
        if enable_request_queue:
            self.request_queue = RequestQueue()
            fraud_logger.info("Request queue initialized")
        
        # Performance monitoring
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time_ms': 0,
            'response_times': []
        }
        self.stats_lock = threading.Lock()
    
    def get_cached_result(self, 
                         user_input: UserTransactionInput, 
                         mapper_type: str) -> Optional[Dict[str, Any]]:
        """Get result from cache (memory first, then Redis)"""
        
        # Try memory cache first
        if self.memory_cache:
            result = self.memory_cache.get(user_input, mapper_type)
            if result:
                with self.stats_lock:
                    self.performance_stats['cache_hits'] += 1
                return result
        
        # Try Redis cache
        if self.redis_cache:
            result = self.redis_cache.get(user_input, mapper_type)
            if result:
                # Store in memory cache for faster access
                if self.memory_cache:
                    self.memory_cache.put(user_input, mapper_type, result)
                
                with self.stats_lock:
                    self.performance_stats['cache_hits'] += 1
                return result
        
        with self.stats_lock:
            self.performance_stats['cache_misses'] += 1
        return None
    
    def cache_result(self, 
                    user_input: UserTransactionInput, 
                    mapper_type: str, 
                    result: Dict[str, Any]):
        """Store result in caches"""
        
        if self.memory_cache:
            self.memory_cache.put(user_input, mapper_type, result)
        
        if self.redis_cache:
            self.redis_cache.put(user_input, mapper_type, result)
    
    def optimize_prediction(self, 
                          prediction_func,
                          user_input: UserTransactionInput,
                          mapper_type: str,
                          **kwargs) -> Dict[str, Any]:
        """
        Optimized prediction with caching and performance monitoring
        
        Args:
            prediction_func: Function to call for prediction
            user_input: User input for prediction
            mapper_type: Type of mapper being used
            **kwargs: Additional arguments for prediction function
            
        Returns:
            Prediction result with performance metrics
        """
        start_time = time.time()
        
        with self.stats_lock:
            self.performance_stats['total_requests'] += 1
        
        # Check cache first
        cached_result = self.get_cached_result(user_input, mapper_type)
        if cached_result:
            processing_time_ms = (time.time() - start_time) * 1000
            cached_result['processing_time_ms'] = processing_time_ms
            cached_result['from_cache'] = True
            
            self._update_response_time(processing_time_ms)
            return cached_result
        
        # Execute prediction
        try:
            if self.request_queue and self.request_queue.active_requests > 10:
                # Use queue for high load
                result = self.request_queue.submit_request(
                    prediction_func, user_input, mapper_type, **kwargs
                )
            else:
                # Direct execution
                result = prediction_func(user_input, mapper_type, **kwargs)
            
            if result and not result.get('error'):
                # Cache successful result
                self.cache_result(user_input, mapper_type, result)
            
            processing_time_ms = (time.time() - start_time) * 1000
            if result:
                result['processing_time_ms'] = processing_time_ms
                result['from_cache'] = False
            
            self._update_response_time(processing_time_ms)
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_response_time(processing_time_ms)
            
            return {
                'error': str(e),
                'processing_time_ms': processing_time_ms,
                'from_cache': False
            }
    
    def _update_response_time(self, response_time_ms: float):
        """Update response time statistics"""
        with self.stats_lock:
            self.performance_stats['response_times'].append(response_time_ms)
            
            # Keep only last 1000 response times
            if len(self.performance_stats['response_times']) > 1000:
                self.performance_stats['response_times'] = \
                    self.performance_stats['response_times'][-1000:]
            
            # Update average
            self.performance_stats['average_response_time_ms'] = \
                np.mean(self.performance_stats['response_times'])
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self.stats_lock:
            stats = self.performance_stats.copy()
        
        # Add cache statistics
        if self.memory_cache:
            stats['memory_cache'] = self.memory_cache.get_stats()
        
        if self.request_queue:
            stats['request_queue'] = self.request_queue.get_stats()
        
        # Calculate additional metrics
        total_requests = stats['total_requests']
        if total_requests > 0:
            stats['cache_hit_rate_percent'] = round(
                (stats['cache_hits'] / total_requests) * 100, 2
            )
        else:
            stats['cache_hit_rate_percent'] = 0
        
        # Response time percentiles
        if stats['response_times']:
            response_times = np.array(stats['response_times'])
            stats['response_time_percentiles'] = {
                'p50': float(np.percentile(response_times, 50)),
                'p90': float(np.percentile(response_times, 90)),
                'p95': float(np.percentile(response_times, 95)),
                'p99': float(np.percentile(response_times, 99))
            }
        
        return stats
    
    def clear_caches(self):
        """Clear all caches"""
        if self.memory_cache:
            self.memory_cache.clear()
        
        if self.redis_cache:
            self.redis_cache.clear()
        
        fraud_logger.info("All caches cleared")
    
    def warm_up_cache(self, 
                     prediction_func,
                     warm_up_data: List[Tuple[UserTransactionInput, str]]):
        """
        Warm up cache with common prediction patterns
        
        Args:
            prediction_func: Prediction function to use
            warm_up_data: List of (user_input, mapper_type) tuples
        """
        fraud_logger.info(f"Warming up cache with {len(warm_up_data)} entries...")
        
        successful_warmups = 0
        for user_input, mapper_type in warm_up_data:
            try:
                result = prediction_func(user_input, mapper_type)
                if result and not result.get('error'):
                    self.cache_result(user_input, mapper_type, result)
                    successful_warmups += 1
            except Exception as e:
                fraud_logger.warning(f"Cache warm-up failed for entry: {e}")
        
        fraud_logger.info(f"Cache warm-up completed: {successful_warmups}/{len(warm_up_data)} successful")
    
    def create_warm_up_data(self) -> List[Tuple[UserTransactionInput, str]]:
        """Create common warm-up data patterns"""
        warm_up_data = []
        
        # Common transaction patterns
        common_patterns = [
            # Normal grocery transactions
            (50.0, MerchantCategory.GROCERY, 14, 2, LocationRisk.NORMAL, SpendingPattern.TYPICAL),
            (75.0, MerchantCategory.GROCERY, 18, 5, LocationRisk.NORMAL, SpendingPattern.TYPICAL),
            
            # Restaurant transactions
            (35.0, MerchantCategory.RESTAURANT, 19, 5, LocationRisk.NORMAL, SpendingPattern.TYPICAL),
            (85.0, MerchantCategory.RESTAURANT, 20, 6, LocationRisk.NORMAL, SpendingPattern.SLIGHTLY_HIGHER),
            
            # Gas station transactions
            (45.0, MerchantCategory.GAS_STATION, 8, 1, LocationRisk.NORMAL, SpendingPattern.TYPICAL),
            (65.0, MerchantCategory.GAS_STATION, 17, 4, LocationRisk.NORMAL, SpendingPattern.TYPICAL),
            
            # Online transactions
            (120.0, MerchantCategory.ONLINE_RETAIL, 21, 0, LocationRisk.NORMAL, SpendingPattern.TYPICAL),
            (250.0, MerchantCategory.ONLINE_RETAIL, 15, 6, LocationRisk.NORMAL, SpendingPattern.SLIGHTLY_HIGHER),
            
            # Suspicious patterns
            (1500.0, MerchantCategory.ONLINE_RETAIL, 2, 1, LocationRisk.SLIGHTLY_UNUSUAL, SpendingPattern.SUSPICIOUS),
            (2000.0, MerchantCategory.TRAVEL, 10, 0, LocationRisk.FOREIGN_COUNTRY, SpendingPattern.MUCH_HIGHER),
        ]
        
        mappers = ['random_forest', 'xgboost', 'ensemble']
        
        for amount, merchant, hour, day, location, spending in common_patterns:
            from fraudguard.entity.feature_mapping_entity import TimeContext
            
            time_context = TimeContext(
                hour_of_day=hour,
                day_of_week=day,
                is_weekend=day >= 5
            )
            
            user_input = UserTransactionInput(
                transaction_amount=amount,
                merchant_category=merchant,
                time_context=time_context,
                location_risk=location,
                spending_pattern=spending
            )
            
            for mapper_type in mappers:
                warm_up_data.append((user_input, mapper_type))
        
        return warm_up_data