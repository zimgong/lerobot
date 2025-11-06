#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Performance profiler for SAC algorithm components.

This module provides comprehensive timing and profiling capabilities for critical
components in the SAC learner and actor processes. It helps identify performance
bottlenecks and optimize the training pipeline.
"""

import logging
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import statistics
import json
import os
from pathlib import Path


@dataclass
class TimingStats:
    """Statistics for a timing measurement."""
    total_time: float = 0.0
    count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_time(self) -> float:
        """Average time for this measurement."""
        return self.total_time / max(self.count, 1)
    
    @property
    def recent_avg(self) -> float:
        """Average of recent measurements (last 100)."""
        if not self.recent_times:
            return 0.0
        return statistics.mean(self.recent_times)
    
    @property
    def recent_p90(self) -> float:
        """90th percentile of recent measurements."""
        if not self.recent_times:
            return 0.0
        try:
            res = statistics.quantiles(self.recent_times, n=10)[8]
        except Exception as e:
            logging.error(f"Error calculating 90th percentile: {e}")
            return 0.0
        return res
    
    def add_measurement(self, duration: float) -> None:
        """Add a new timing measurement."""
        self.total_time += duration
        self.count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.recent_times.append(duration)


class PerformanceProfiler:
    """
    Comprehensive performance profiler for SAC algorithm components.
    
    This profiler tracks timing for critical components and provides detailed
    statistics including averages, percentiles, and frequency analysis.
    """
    
    def __init__(self, log_frequency: int = 100, log_dir: Optional[str] = None):
        """
        Initialize the profiler.
        
        Args:
            log_frequency: How often to log statistics (in measurements)
            log_dir: Directory to save detailed profiling reports
        """
        self.timings: Dict[str, TimingStats] = defaultdict(TimingStats)
        self.log_frequency = log_frequency
        self.log_dir = log_dir
        self.measurement_count = 0
        
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
    
    @contextmanager
    def time_block(self, name: str, log_immediately: bool = False):
        """
        Context manager for timing a code block.
        
        Args:
            name: Name of the timing measurement
            log_immediately: Whether to log immediately after this measurement
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.add_timing(name, duration)
            
            if log_immediately:
                self._log_single_measurement(name, duration)
    
    def add_timing(self, name: str, duration: float) -> None:
        """
        Add a timing measurement.
        
        Args:
            name: Name of the timing measurement
            duration: Duration in seconds
        """
        self.timings[name].add_measurement(duration)
        self.measurement_count += 1
        
        # Log statistics periodically
        if self.measurement_count % self.log_frequency == 0:
            self._log_statistics()
    
    def _log_single_measurement(self, name: str, duration: float) -> None:
        """Log a single measurement immediately."""
        stats = self.timings[name]
        logging.info(
            f"[PROFILER] {name}: {duration*1000:.2f}ms "
            f"(avg: {stats.avg_time*1000:.2f}ms, count: {stats.count})"
        )
    
    def _log_statistics(self) -> None:
        """Log comprehensive statistics for all measurements."""
        if not self.timings:
            return
            
        logging.info("[PROFILER] === Performance Statistics ===")
        
        # Sort by total time to show most expensive operations first
        sorted_timings = sorted(
            self.timings.items(), 
            key=lambda x: x[1].total_time, 
            reverse=True
        )
        
        for name, stats in sorted_timings:
            if stats.count == 0:
                continue
                
            logging.info(
                f"[PROFILER] {name}:\n"
                f"  Total: {stats.total_time:.3f}s ({stats.count} calls)\n"
                f"  Average: {stats.avg_time*1000:.2f}ms\n"
                f"  Min/Max: {stats.min_time*1000:.2f}ms / {stats.max_time*1000:.2f}ms\n"
                f"  Recent avg: {stats.recent_avg*1000:.2f}ms\n"
                f"  Recent P90: {stats.recent_p90*1000:.2f}ms"
            )
        
        logging.info("[PROFILER] === End Performance Statistics ===")
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive statistics for all measurements.
        
        Returns:
            Dictionary mapping measurement names to their statistics
        """
        stats_dict = {}
        for name, stats in self.timings.items():
            if stats.count == 0:
                continue
                
            stats_dict[name] = {
                "total_time": stats.total_time,
                "count": stats.count,
                "avg_time": stats.avg_time,
                "min_time": stats.min_time,
                "max_time": stats.max_time,
                "recent_avg": stats.recent_avg,
                "recent_p90": stats.recent_p90,
            }
        
        return stats_dict
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """
        Save a detailed profiling report to file.
        
        Args:
            filename: Optional filename, defaults to timestamped name
            
        Returns:
            Path to the saved report file
        """
        if not self.log_dir:
            raise ValueError("log_dir must be set to save reports")
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"profiling_report_{timestamp}.json"
        
        report_path = os.path.join(self.log_dir, filename)
        
        report = {
            "timestamp": time.time(),
            "measurement_count": self.measurement_count,
            "statistics": self.get_statistics(),
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"[PROFILER] Report saved to {report_path}")
        return report_path
    
    def reset(self) -> None:
        """Reset all timing measurements."""
        self.timings.clear()
        self.measurement_count = 0
    
    def get_frequency_analysis(self) -> Dict[str, float]:
        """
        Get frequency analysis for all measurements.
        
        Returns:
            Dictionary mapping measurement names to their frequencies (Hz)
        """
        frequencies = {}
        for name, stats in self.timings.items():
            if stats.count > 0 and stats.total_time > 0:
                frequencies[name] = stats.count / stats.total_time
        
        return frequencies


class ComponentProfiler:
    """
    Specialized profiler for SAC algorithm components.
    
    This profiler provides pre-defined timing categories for common SAC operations
    and makes it easy to profile the critical path of the algorithm.
    """
    
    def __init__(self, log_frequency: int = 100, log_dir: Optional[str] = None):
        """
        Initialize the component profiler.
        
        Args:
            log_frequency: How often to log statistics
            log_dir: Directory to save profiling reports
        """
        self.profiler = PerformanceProfiler(log_frequency, log_dir)
        
        # Pre-defined component categories
        self.learner_components = [
            "data_processing",
            "observation_encoding", 
            "critic_forward",
            "critic_backward",
            "critic_optimization",
            "actor_forward",
            "actor_backward", 
            "actor_optimization",
            "temperature_forward",
            "temperature_backward",
            "temperature_optimization",
            "target_network_update",
            "batch_sampling",
            "gradient_clipping",
            "policy_parameter_push",
        ]
        
        self.actor_components = [
            "policy_inference",
            "environment_step",
            "observation_processing",
            "action_processing", 
            "transition_creation",
            "data_serialization",
            "network_communication",
            "parameter_loading",
            "fps_wait",
            "actor_forward",
            "build_packaged_image_features",
            "vla_model_common_process",
            "extract_last_layer_kv",
        ]
    
    def time_learner_component(self, component: str):
        """Context manager for timing learner components."""
        if component not in self.learner_components:
            logging.warning(f"Unknown learner component: {component}")
        return self.profiler.time_block(f"learner_{component}")
    
    def time_actor_component(self, component: str):
        """Context manager for timing actor components."""
        if component not in self.actor_components:
            logging.warning(f"Unknown actor component: {component}")
        return self.profiler.time_block(f"actor_{component}")
    
    def time_custom(self, name: str):
        """Context manager for timing custom components."""
        return self.profiler.time_block(name)
    
    def get_learner_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all learner components."""
        learner_stats = {}
        for component in self.learner_components:
            key = f"learner_{component}"
            if key in self.profiler.timings:
                learner_stats[component] = self.profiler.timings[key].__dict__.copy()
                if 'recent_times' in learner_stats[component]:
                    del learner_stats[component]['recent_times']
        return learner_stats
    
    def get_actor_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all actor components."""
        actor_stats = {}
        for component in self.actor_components:
            key = f"actor_{component}"
            if key in self.profiler.timings:
                actor_stats[component] = self.profiler.timings[key].__dict__.copy()
                if 'recent_times' in actor_stats[component]:
                    del actor_stats[component]['recent_times']
        return actor_stats
    
    def log_component_summary(self) -> None:
        """Log a summary of component performance."""
        logging.info("[PROFILER] === Component Performance Summary ===")
        
        # Learner components
        learner_stats = self.get_learner_stats()
        if learner_stats:
            logging.info("[PROFILER] Learner Components:")
            for component, stats in sorted(learner_stats.items(), 
                                        key=lambda x: x[1]['total_time'], 
                                        reverse=True):
                if 'avg_time' not in stats:
                    logging.warning(f"No avg_time for component: {component}")
                    logging.info(f"  {component}: {stats['total_time']:.3f}s total")
                else:
                    logging.info(
                        f"  {component}: {stats['avg_time']*1000:.2f}ms avg "
                        f"({stats['count']} calls, {stats['total_time']:.3f}s total)"
                    )
        
        # Actor components  
        actor_stats = self.get_actor_stats()
        if actor_stats:
            logging.info("[PROFILER] Actor Components:")
            for component, stats in sorted(actor_stats.items(),
                                        key=lambda x: x[1]['total_time'],
                                        reverse=True):
                if 'avg_time' not in stats:
                    logging.warning(f"No avg_time for component: {component}")
                    logging.info(f"  {component}: {stats['total_time']:.3f}s total")
                else:
                    logging.info(
                        f"  {component}: {stats['avg_time']*1000:.2f}ms avg "
                        f"({stats['count']} calls, {stats['total_time']:.3f}s total)"
                    )
        
        logging.info("[PROFILER] === End Component Summary ===")
    
    def save_component_report(self, filename: Optional[str] = None) -> str:
        """Save a component-specific profiling report."""
        if not self.profiler.log_dir:
            raise ValueError("log_dir must be set to save reports")
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"component_profiling_report_{timestamp}.json"
        
        report_path = os.path.join(self.profiler.log_dir, filename)
        
        report = {
            "timestamp": time.time(),
            "learner_components": self.get_learner_stats(),
            "actor_components": self.get_actor_stats(),
            "all_statistics": self.profiler.get_statistics(),
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"[PROFILER] Component report saved to {report_path}")
        return report_path


# Global profiler instance for easy access
_global_profiler: Optional[ComponentProfiler] = None


def get_profiler(log_frequency: int = 100, log_dir: Optional[str] = None) -> ComponentProfiler:
    """
    Get or create the global profiler instance.
    
    Args:
        log_frequency: How often to log statistics
        log_dir: Directory to save profiling reports
        
    Returns:
        The global ComponentProfiler instance
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = ComponentProfiler(log_frequency, log_dir)
    return _global_profiler


def reset_profiler() -> None:
    """Reset the global profiler."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.profiler.reset()
