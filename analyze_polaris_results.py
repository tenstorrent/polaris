# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
from collections import defaultdict

# Optional imports for visualization
HAS_MATPLOTLIB = False
HAS_PANDAS = False
HAS_SEABORN = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
    # Set style for better looking plots
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
except ImportError:
    pass

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pass

try:
    import seaborn as sns
    HAS_SEABORN = True
    if HAS_SEABORN:
        sns.set_palette("husl")
except ImportError:
    pass

try:
    import numpy as np
except ImportError:
    # Fallback numpy-like functions
    class FakeNumpy:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        
        @staticmethod
        def zeros(shape):
            if isinstance(shape, tuple) and len(shape) == 2:
                return [[0] * shape[1] for _ in range(shape[0])]
            return [0] * shape
        
        @staticmethod
        def arange(n):
            return list(range(n))
    
    np = FakeNumpy()

@dataclass
class SimulationMetric:
    """Data class to hold simulation metrics"""
    device: str
    workload_type: str
    workload_instance: str
    workload: str
    time_ms: float
    throughput: float
    ideal_throughput: float
    memory_GB: float
    fits_device: bool
    compute_util: float
    memory_util: float
    param_bytes: int
    efficiency: float
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.ideal_throughput > 0:
            self.efficiency = (self.throughput / self.ideal_throughput) * 100
        else:
            self.efficiency = 0


class PolarisResultsAnalyzer:
    """Main analyzer class for Polaris simulation results"""
    
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.metrics: List[SimulationMetric] = []
        self.devices = set()
        self.workloads = set()
        
        # Validate output path
        if not self.output_path.exists():
            raise FileNotFoundError(f"Output path does not exist: {output_path}")
        
        # Check for required subdirectories
        required_dirs = ['CONFIG', 'STATS', 'SUMMARY']
        for dir_name in required_dirs:
            if not (self.output_path / dir_name).exists():
                raise FileNotFoundError(f"Required directory missing: {dir_name}")
    
    def extract_metrics_from_summary(self) -> List[SimulationMetric]:
        """Extract metrics from study-summary.yaml file"""
        summary_file = self.output_path / 'SUMMARY' / 'study-summary.yaml'
        
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")
        
        with open(summary_file, 'r') as f:
            content = f.read()
        
        metrics = []
        
        # First try the new format with __dict__ structure
        dict_entries = re.split(r'- !!python/object:ttsim\.config\.validators\.TTSimHLRunSummaryRow\s+__dict__:', content)
        
        if len(dict_entries) > 1:
            # New format with __dict__
            for entry in dict_entries[1:]:
                # Extract fields using regex for the __dict__ format
                fields = {}
                
                patterns = {
                    'devname': r'devname:\s*([A-Z0-9_a-z]+)',
                    'wlname': r'wlname:\s*([A-Z_a-z]+)',
                    'wlinstance': r'wlinstance:\s*([a-z0-9_]+)',
                    'tot_msecs': r'tot_msecs:\s*([0-9.]+)',
                    'perf_projection': r'perf_projection:\s*([0-9.]+)',
                    'ideal_throughput': r'ideal_throughput:\s*([0-9.]+)',
                    'mem_size_GB': r'mem_size_GB:\s*([0-9.]+)',
                    'fits_device': r'fits_device:\s*(true|false)',
                    'rsrc_comp': r'rsrc_comp:\s*([0-9.]+)',
                    'rsrc_mem': r'rsrc_mem:\s*([0-9.]+)',
                    'inParamBytes': r'inParamBytes:\s*([0-9]+)'
                }
                
                for field, pattern in patterns.items():
                    match = re.search(pattern, entry)
                    if match:
                        if field in ['tot_msecs', 'perf_projection', 'ideal_throughput', 'mem_size_GB', 'rsrc_comp', 'rsrc_mem']:
                            fields[field] = float(match.group(1))
                        elif field in ['inParamBytes']:
                            fields[field] = int(match.group(1))
                        elif field == 'fits_device':
                            fields[field] = match.group(1) == 'true'
                        else:
                            fields[field] = match.group(1)
                
                # Only add if we have the essential fields
                if all(k in fields for k in ['devname', 'wlname', 'wlinstance', 'tot_msecs', 'perf_projection', 'ideal_throughput']):
                    metric = SimulationMetric(
                        device=fields['devname'],
                        workload_type=fields['wlname'],
                        workload_instance=fields['wlinstance'],
                        workload=f"{fields['wlname']}-{fields['wlinstance']}",
                        time_ms=fields['tot_msecs'],
                        throughput=fields['perf_projection'],
                        ideal_throughput=fields['ideal_throughput'],
                        memory_GB=fields.get('mem_size_GB', 0.0),
                        fits_device=fields.get('fits_device', True),
                        compute_util=fields.get('rsrc_comp', 0.0),
                        memory_util=fields.get('rsrc_mem', 0.0),
                        param_bytes=fields.get('inParamBytes', 0),
                        efficiency=0  # Will be calculated in __post_init__
                    )
                    metrics.append(metric)
                    self.devices.add(metric.device)
                    self.workloads.add(metric.workload)
        else:
            # Fallback to old format without __dict__
            old_entries = re.split(r'- !!python/object:ttsim\.config\.validators\.TTSimHLRunSummaryRow', content)
            
            for entry in old_entries[1:]:  # Skip first empty entry
                # Extract fields using regex
                fields = {}
                
                patterns = {
                    'devname': r'devname: ([A-Z0-9_]+)',
                    'wlname': r'wlname: ([A-Z_a-z]+)',
                    'wlinstance': r'wlinstance: ([a-z0-9_]+)',
                    'tot_msecs': r'tot_msecs: ([0-9.]+)',
                    'perf_projection': r'perf_projection: ([0-9.]+)',
                    'ideal_throughput': r'ideal_throughput: ([0-9.]+)',
                    'mem_size_GB': r'mem_size_GB: ([0-9.]+)',
                    'fits_device': r'fits_device: (true|false)',
                    'rsrc_comp': r'rsrc_comp: ([0-9.]+)',
                    'rsrc_mem': r'rsrc_mem: ([0-9.]+)',
                    'inParamBytes': r'inParamBytes: ([0-9]+)'
                }
                
                for field, pattern in patterns.items():
                    match = re.search(pattern, entry)
                    if match:
                        if field in ['tot_msecs', 'perf_projection', 'ideal_throughput', 'mem_size_GB', 'rsrc_comp', 'rsrc_mem']:
                            fields[field] = float(match.group(1))
                        elif field in ['inParamBytes']:
                            fields[field] = int(match.group(1))
                        elif field == 'fits_device':
                            fields[field] = match.group(1) == 'true'
                        else:
                            fields[field] = match.group(1)
                
                # Only add if we have the essential fields
                if all(k in fields for k in ['devname', 'wlname', 'wlinstance', 'tot_msecs', 'perf_projection', 'ideal_throughput']):
                    metric = SimulationMetric(
                        device=fields['devname'],
                        workload_type=fields['wlname'],
                        workload_instance=fields['wlinstance'],
                        workload=f"{fields['wlname']}-{fields['wlinstance']}",
                        time_ms=fields['tot_msecs'],
                        throughput=fields['perf_projection'],
                        ideal_throughput=fields['ideal_throughput'],
                        memory_GB=fields.get('mem_size_GB', 0.0),
                        fits_device=fields.get('fits_device', True),
                        compute_util=fields.get('rsrc_comp', 0.0),
                        memory_util=fields.get('rsrc_mem', 0.0),
                        param_bytes=fields.get('inParamBytes', 0),
                        efficiency=0  # Will be calculated in __post_init__
                    )
                    metrics.append(metric)
                    self.devices.add(metric.device)
                    self.workloads.add(metric.workload)
        
        return metrics
    
    def load_data(self):
        """Load all simulation data"""
        print(f"Loading data from {self.output_path}...")
        self.metrics = self.extract_metrics_from_summary()
        print(f"Loaded {len(self.metrics)} simulation results")
        print(f"Devices: {sorted(self.devices)}")
        print(f"Workloads: {len(self.workloads)}")
    
    def generate_summary_report(self) -> str:
        """Generate human-readable summary report"""
        report = []
        report.append("=" * 80)
        report.append("POLARIS SIMULATION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overview
        report.append("📊 SIMULATION OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total simulations: {len(self.metrics)}")
        report.append(f"Devices tested: {len(self.devices)} ({', '.join(sorted(self.devices))})")
        report.append(f"Workloads tested: {len(self.workloads)}")
        report.append(f"Output directory: {self.output_path}")
        report.append("")
        
        # Device comparison
        report.append("🖥️  DEVICE PERFORMANCE COMPARISON")
        report.append("-" * 40)
        device_stats = {}
        for device in sorted(self.devices):
            device_metrics = [m for m in self.metrics if m.device == device]
            avg_throughput = np.mean([m.throughput for m in device_metrics])
            avg_time = np.mean([m.time_ms for m in device_metrics])
            avg_efficiency = np.mean([m.efficiency for m in device_metrics])
            device_stats[device] = {
                'count': len(device_metrics),
                'avg_throughput': avg_throughput,
                'avg_time': avg_time,
                'avg_efficiency': avg_efficiency
            }
            
            report.append(f"{device}:")
            report.append(f"  - Workloads: {len(device_metrics)}")
            report.append(f"  - Avg Throughput: {avg_throughput:,.1f}")
            report.append(f"  - Avg Time: {avg_time:.3f} ms")
            report.append(f"  - Avg Efficiency: {avg_efficiency:.1f}%")
            report.append("")
        
        # Top performers
        report.append("🏆 TOP PERFORMING CONFIGURATIONS")
        report.append("-" * 40)
        sorted_metrics = sorted(self.metrics, key=lambda x: x.throughput, reverse=True)
        
        for i, metric in enumerate(sorted_metrics[:10]):
            report.append(f"{i+1:2d}. {metric.workload:20s} on {metric.device:8s}: "
                         f"{metric.throughput:10,.1f} throughput ({metric.efficiency:.1f}% eff)")
        report.append("")
        
        # Memory analysis
        report.append("💾 MEMORY USAGE ANALYSIS")
        report.append("-" * 40)
        memory_sorted = sorted(self.metrics, key=lambda x: x.memory_GB)
        
        report.append("Smallest memory footprint:")
        for metric in memory_sorted[:3]:
            report.append(f"  {metric.workload:20s}: {metric.memory_GB:.3f} GB (on {metric.device})")
        
        report.append("Largest memory footprint:")
        for metric in memory_sorted[-3:]:
            report.append(f"  {metric.workload:20s}: {metric.memory_GB:.3f} GB (on {metric.device})")
        report.append("")
        
        # Scaling insights
        report.append("📈 SCALING INSIGHTS")
        report.append("-" * 40)
        
        # GPT scaling - show best performance for each model across all devices
        gpt_models_by_instance = {}
        for m in self.metrics:
            if 'gpt' in m.workload_instance.lower():
                if m.workload_instance not in gpt_models_by_instance:
                    gpt_models_by_instance[m.workload_instance] = []
                gpt_models_by_instance[m.workload_instance].append(m)
        
        if gpt_models_by_instance:
            report.append("GPT Model Scaling (best performance per model):")
            # Get best performing device for each model
            best_models = []
            for instance, models in gpt_models_by_instance.items():
                best_model = max(models, key=lambda x: x.throughput)
                best_models.append(best_model)
            
            best_models.sort(key=lambda x: x.memory_GB)
            for model in best_models:
                params_mb = model.param_bytes / (1024*1024) if model.param_bytes > 0 else 0
                report.append(f"  {model.workload_instance:12s}: {model.memory_GB:6.3f} GB, "
                             f"{params_mb:7.1f} MB params, {model.time_ms:8.3f} ms, "
                             f"{model.throughput:10,.1f} throughput (on {model.device})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def create_visualizations(self, output_dir: Path):
        """Create various charts and visualizations"""
        if not HAS_MATPLOTLIB:
            print("⚠️  Matplotlib not available - skipping visualizations")
            print("   Install matplotlib with: pip install matplotlib pandas seaborn")
            return
        
        output_dir.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        
        plots = [
            ("Device Comparison", self._plot_device_comparison),
            ("Performance Matrix", self._plot_performance_matrix),
            ("Memory vs Performance", self._plot_memory_vs_performance),
            ("Efficiency Analysis", self._plot_efficiency_analysis),
            ("Scaling Analysis", self._plot_scaling_analysis),
            ("Resource Utilization", self._plot_resource_utilization),
        ]
        
        for plot_name, plot_func in plots:
            try:
                plot_func(output_dir)
            except Exception as e:
                print(f"⚠️  Warning: Failed to generate {plot_name} plot: {e}")
                continue
        
        print(f"📊 Visualizations saved to {output_dir}")
    
    def _plot_device_comparison(self, output_dir: Path):
        """Plot device performance comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        devices = sorted(self.devices)
        
        # Average throughput by device
        device_throughput = []
        device_time = []
        device_efficiency = []
        device_memory = []
        
        for device in devices:
            device_metrics = [m for m in self.metrics if m.device == device]
            device_throughput.append(np.mean([m.throughput for m in device_metrics]))
            device_time.append(np.mean([m.time_ms for m in device_metrics]))
            device_efficiency.append(np.mean([m.efficiency for m in device_metrics]))
            device_memory.append(np.mean([m.memory_GB for m in device_metrics]))
        
        # Throughput comparison
        bars1 = ax1.bar(devices, device_throughput, color='skyblue', alpha=0.7)
        ax1.set_title('Average Throughput by Device', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Throughput')
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Add value labels on bars
        for bar, value in zip(bars1, device_throughput):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                    f'{value:,.0f}', ha='center', va='bottom', fontsize=10)
        
        # Time comparison
        bars2 = ax2.bar(devices, device_time, color='lightcoral', alpha=0.7)
        ax2.set_title('Average Execution Time by Device', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (ms)')
        
        for bar, value in zip(bars2, device_time):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Efficiency comparison
        bars3 = ax3.bar(devices, device_efficiency, color='lightgreen', alpha=0.7)
        ax3.set_title('Average Efficiency by Device', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Efficiency (%)')
        
        for bar, value in zip(bars3, device_efficiency):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Memory usage comparison
        bars4 = ax4.bar(devices, device_memory, color='gold', alpha=0.7)
        ax4.set_title('Average Memory Usage by Device', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Memory (GB)')
        
        for bar, value in zip(bars4, device_memory):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'device_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_matrix(self, output_dir: Path):
        """Plot performance matrix heatmap"""
        # Create a pivot table for the heatmap
        devices = sorted(self.devices)
        workloads = sorted(self.workloads)
        
        # Create matrix for throughput
        if isinstance(np.zeros((len(workloads), len(devices))), list):
            # Using fake numpy
            throughput_matrix = [[0 for _ in range(len(devices))] for _ in range(len(workloads))]
            time_matrix = [[0 for _ in range(len(devices))] for _ in range(len(workloads))]
        else:
            # Using real numpy
            throughput_matrix = np.zeros((len(workloads), len(devices)))
            time_matrix = np.zeros((len(workloads), len(devices)))
        
        for i, workload in enumerate(workloads):
            for j, device in enumerate(devices):
                metric = next((m for m in self.metrics if m.workload == workload and m.device == device), None)
                if metric:
                    throughput_matrix[i][j] = metric.throughput
                    time_matrix[i][j] = metric.time_ms
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Throughput heatmap
        im1 = ax1.imshow(throughput_matrix, aspect='auto', cmap='YlOrRd')
        ax1.set_title('Throughput Performance Matrix', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(devices)))
        ax1.set_xticklabels(devices)
        ax1.set_yticks(range(len(workloads)))
        ax1.set_yticklabels([w.replace('basic_llm-', '') for w in workloads], fontsize=10)
        
        # Add text annotations
        for i in range(len(workloads)):
            for j in range(len(devices)):
                if throughput_matrix[i, j] > 0:
                    text = ax1.text(j, i, f'{throughput_matrix[i, j]:.0f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im1, ax=ax1, label='Throughput')
        
        # Time heatmap
        im2 = ax2.imshow(time_matrix, aspect='auto', cmap='YlGnBu')
        ax2.set_title('Execution Time Matrix', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(devices)))
        ax2.set_xticklabels(devices)
        ax2.set_yticks(range(len(workloads)))
        ax2.set_yticklabels([w.replace('basic_llm-', '') for w in workloads], fontsize=10)
        
        # Add text annotations
        for i in range(len(workloads)):
            for j in range(len(devices)):
                if time_matrix[i, j] > 0:
                    text = ax2.text(j, i, f'{time_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="white", fontsize=8)
        
        plt.colorbar(im2, ax=ax2, label='Time (ms)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_vs_performance(self, output_dir: Path):
        """Plot memory usage vs performance scatter plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        devices = sorted(self.devices)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Memory vs Throughput
        for i, device in enumerate(devices):
            device_metrics = [m for m in self.metrics if m.device == device]
            memory = [m.memory_GB for m in device_metrics]
            throughput = [m.throughput for m in device_metrics]
            ax1.scatter(memory, throughput, label=device, alpha=0.7, color=colors[i % len(colors)])
        
        ax1.set_xlabel('Memory Usage (GB)')
        ax1.set_ylabel('Throughput')
        ax1.set_title('Memory vs Throughput', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        
        # Memory vs Time
        for i, device in enumerate(devices):
            device_metrics = [m for m in self.metrics if m.device == device]
            memory = [m.memory_GB for m in device_metrics]
            time = [m.time_ms for m in device_metrics]
            ax2.scatter(memory, time, label=device, alpha=0.7, color=colors[i % len(colors)])
        
        ax2.set_xlabel('Memory Usage (GB)')
        ax2.set_ylabel('Execution Time (ms)')
        ax2.set_title('Memory vs Execution Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        
        # Parameters vs Throughput
        for i, device in enumerate(devices):
            device_metrics = [m for m in self.metrics if m.device == device and m.param_bytes > 0]
            params = [m.param_bytes / (1024*1024) for m in device_metrics]  # Convert to MB
            throughput = [m.throughput for m in device_metrics]
            ax3.scatter(params, throughput, label=device, alpha=0.7, color=colors[i % len(colors)])
        
        ax3.set_xlabel('Parameters (MB)')
        ax3.set_ylabel('Throughput')
        ax3.set_title('Model Size vs Throughput', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        
        # Efficiency vs Memory
        for i, device in enumerate(devices):
            device_metrics = [m for m in self.metrics if m.device == device]
            memory = [m.memory_GB for m in device_metrics]
            efficiency = [m.efficiency for m in device_metrics]
            ax4.scatter(memory, efficiency, label=device, alpha=0.7, color=colors[i % len(colors)])
        
        ax4.set_xlabel('Memory Usage (GB)')
        ax4.set_ylabel('Efficiency (%)')
        ax4.set_title('Memory vs Efficiency', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'memory_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_analysis(self, output_dir: Path):
        """Plot efficiency analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Efficiency distribution
        efficiencies = [m.efficiency for m in self.metrics]
        
        # Check if all efficiencies are the same (or very close)
        efficiency_range = max(efficiencies) - min(efficiencies)
        if efficiency_range < 1e-6:  # All values are essentially the same
            # Create a simple bar chart instead of histogram
            unique_eff = efficiencies[0]
            ax1.bar(['Efficiency'], [len(efficiencies)], alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_ylabel('Count')
            ax1.set_title(f'Efficiency Distribution (All simulations: {unique_eff:.1f}%)', fontsize=14, fontweight='bold')
            ax1.text(0, len(efficiencies)/2, f'{len(efficiencies)} simulations\nat {unique_eff:.1f}%', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            # Use adaptive binning
            num_bins = min(20, max(5, int(efficiency_range)))
            ax1.hist(efficiencies, bins=num_bins, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Efficiency (%)')
            ax1.set_ylabel('Count')
            ax1.set_title('Efficiency Distribution', fontsize=14, fontweight='bold')
            ax1.axvline(np.mean(efficiencies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(efficiencies):.1f}%')
            ax1.legend()
        
        # Efficiency by workload type
        workload_types = set(m.workload_type for m in self.metrics)
        for workload_type in workload_types:
            type_metrics = [m for m in self.metrics if m.workload_type == workload_type]
            type_efficiencies = [m.efficiency for m in type_metrics]
            ax2.scatter([workload_type] * len(type_efficiencies), type_efficiencies, 
                       alpha=0.7, s=50, label=workload_type)
        
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title('Efficiency by Workload Type', fontsize=14, fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scaling_analysis(self, output_dir: Path):
        """Plot scaling analysis for GPT models"""
        # Filter GPT models and sort by size
        gpt_metrics = [m for m in self.metrics if 'gpt' in m.workload_instance.lower()]
        gpt_metrics.sort(key=lambda x: x.memory_GB)
        
        if not gpt_metrics:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        devices = sorted(self.devices)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Group by device
        for i, device in enumerate(devices):
            device_gpt = [m for m in gpt_metrics if m.device == device]
            device_gpt.sort(key=lambda x: x.memory_GB)
            
            if not device_gpt:
                continue
            
            model_names = [m.workload_instance for m in device_gpt]
            memory = [m.memory_GB for m in device_gpt]
            throughput = [m.throughput for m in device_gpt]
            time = [m.time_ms for m in device_gpt]
            params = [m.param_bytes / (1024*1024) for m in device_gpt if m.param_bytes > 0]
            
            # Memory scaling
            ax1.plot(range(len(memory)), memory, 'o-', label=device, 
                    color=colors[i % len(colors)], linewidth=2, markersize=8)
            
            # Throughput scaling
            ax2.plot(range(len(throughput)), throughput, 'o-', label=device,
                    color=colors[i % len(colors)], linewidth=2, markersize=8)
            
            # Time scaling
            ax3.plot(range(len(time)), time, 'o-', label=device,
                    color=colors[i % len(colors)], linewidth=2, markersize=8)
            
            # Parameters scaling (if available)
            if params:
                ax4.plot(range(len(params)), params, 'o-', label=device,
                        color=colors[i % len(colors)], linewidth=2, markersize=8)
        
        # Set labels and titles
        model_labels = [m.workload_instance.replace('gpt_', '').replace('gpt', '') for m in device_gpt]
        
        ax1.set_xlabel('Model Size Progression')
        ax1.set_ylabel('Memory (GB)')
        ax1.set_title('GPT Model Memory Scaling', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(model_labels)))
        ax1.set_xticklabels(model_labels, rotation=45)
        ax1.legend()
        ax1.set_yscale('log')
        
        ax2.set_xlabel('Model Size Progression')
        ax2.set_ylabel('Throughput')
        ax2.set_title('GPT Model Throughput Scaling', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(model_labels)))
        ax2.set_xticklabels(model_labels, rotation=45)
        ax2.legend()
        ax2.set_yscale('log')
        
        ax3.set_xlabel('Model Size Progression')
        ax3.set_ylabel('Time (ms)')
        ax3.set_title('GPT Model Time Scaling', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(model_labels)))
        ax3.set_xticklabels(model_labels, rotation=45)
        ax3.legend()
        ax3.set_yscale('log')
        
        ax4.set_xlabel('Model Size Progression')
        ax4.set_ylabel('Parameters (MB)')
        ax4.set_title('GPT Model Parameter Scaling', fontsize=14, fontweight='bold')
        if params:
            ax4.set_xticks(range(len(model_labels)))
            ax4.set_xticklabels(model_labels, rotation=45)
        ax4.legend()
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_resource_utilization(self, output_dir: Path):
        """Plot resource utilization analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Compute vs Memory utilization scatter
        compute_util = [m.compute_util for m in self.metrics if m.compute_util > 0]
        memory_util = [m.memory_util for m in self.metrics if m.memory_util > 0]
        
        if compute_util and memory_util:
            ax1.scatter(compute_util, memory_util, alpha=0.6, s=50)
            ax1.set_xlabel('Compute Utilization')
            ax1.set_ylabel('Memory Utilization')
            ax1.set_title('Compute vs Memory Utilization', fontsize=14, fontweight='bold')
            ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal utilization')
            ax1.legend()
        
        # Resource utilization by device
        devices = sorted(self.devices)
        device_compute = []
        device_memory = []
        
        for device in devices:
            device_metrics = [m for m in self.metrics if m.device == device and m.compute_util > 0]
            if device_metrics:
                device_compute.append(np.mean([m.compute_util for m in device_metrics]))
                device_memory.append(np.mean([m.memory_util for m in device_metrics]))
            else:
                device_compute.append(0)
                device_memory.append(0)
        
        x = np.arange(len(devices))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, device_compute, width, label='Compute', alpha=0.7)
        bars2 = ax2.bar(x + width/2, device_memory, width, label='Memory', alpha=0.7)
        
        ax2.set_xlabel('Device')
        ax2.set_ylabel('Average Utilization')
        ax2.set_title('Resource Utilization by Device', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(devices)
        ax2.legend()
        
        # Throughput vs Compute utilization
        if compute_util:
            throughput = [m.throughput for m in self.metrics if m.compute_util > 0]
            ax3.scatter(compute_util, throughput, alpha=0.6, s=50)
            ax3.set_xlabel('Compute Utilization')
            ax3.set_ylabel('Throughput')
            ax3.set_title('Throughput vs Compute Utilization', fontsize=14, fontweight='bold')
            ax3.set_yscale('log')
        
        # Bottleneck analysis
        bottleneck_types = []
        for m in self.metrics:
            if m.compute_util > m.memory_util:
                bottleneck_types.append('Compute Bound')
            elif m.memory_util > m.compute_util:
                bottleneck_types.append('Memory Bound')
            else:
                bottleneck_types.append('Balanced')
        
        bottleneck_counts = {bt: bottleneck_types.count(bt) for bt in set(bottleneck_types)}
        
        ax4.pie(bottleneck_counts.values(), labels=bottleneck_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Resource Bottleneck Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'resource_utilization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_data(self, output_dir: Path):
        """Export data to CSV and JSON formats"""
        output_dir.mkdir(exist_ok=True)
        
        # Convert to list of dictionaries
        data = []
        for metric in self.metrics:
            data.append({
                'device': metric.device,
                'workload_type': metric.workload_type,
                'workload_instance': metric.workload_instance,
                'workload': metric.workload,
                'time_ms': metric.time_ms,
                'throughput': metric.throughput,
                'ideal_throughput': metric.ideal_throughput,
                'efficiency': metric.efficiency,
                'memory_GB': metric.memory_GB,
                'param_bytes': metric.param_bytes,
                'fits_device': metric.fits_device,
                'compute_util': metric.compute_util,
                'memory_util': metric.memory_util
            })
        
        # Export to CSV manually if pandas not available
        csv_file = output_dir / 'simulation_results.csv'
        if HAS_PANDAS:
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
        else:
            # Manual CSV export
            headers = list(data[0].keys()) if data else []
            with open(csv_file, 'w') as f:
                # Write header
                f.write(','.join(headers) + '\n')
                # Write data
                for row in data:
                    values = [str(row[header]) for header in headers]
                    f.write(','.join(values) + '\n')
        
        print(f"📊 Data exported to {csv_file}")
        
        # Export to JSON
        json_file = output_dir / 'simulation_results.json'
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"📊 Data exported to {json_file}")
        
        # Export summary statistics
        summary_stats = {
            'total_simulations': len(self.metrics),
            'devices': list(sorted(self.devices)),
            'workloads': list(sorted(self.workloads)),
            'device_stats': {},
            'workload_stats': {}
        }
        
        # Device statistics
        for device in sorted(self.devices):
            device_metrics = [m for m in self.metrics if m.device == device]
            summary_stats['device_stats'][device] = {
                'count': len(device_metrics),
                'avg_throughput': float(np.mean([m.throughput for m in device_metrics])),
                'avg_time_ms': float(np.mean([m.time_ms for m in device_metrics])),
                'avg_efficiency': float(np.mean([m.efficiency for m in device_metrics])),
                'avg_memory_GB': float(np.mean([m.memory_GB for m in device_metrics]))
            }
        
        # Workload statistics
        for workload in sorted(self.workloads):
            workload_metrics = [m for m in self.metrics if m.workload == workload]
            summary_stats['workload_stats'][workload] = {
                'count': len(workload_metrics),
                'avg_throughput': float(np.mean([m.throughput for m in workload_metrics])),
                'avg_time_ms': float(np.mean([m.time_ms for m in workload_metrics])),
                'avg_efficiency': float(np.mean([m.efficiency for m in workload_metrics])),
                'memory_GB': float(workload_metrics[0].memory_GB) if workload_metrics else 0
            }
        
        summary_file = output_dir / 'summary_statistics.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"📊 Summary statistics exported to {summary_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Analyze Polaris simulation results and generate reports and visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_polaris_results.py output/my_first_test
    python analyze_polaris_results.py /path/to/simulation/output
        """
    )
    
    parser.add_argument('output_path', type=str,
                       help='Path to the Polaris simulation output directory')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots and visualizations')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Directory to save analysis results (default: analysis_output)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = PolarisResultsAnalyzer(args.output_path)
        
        # Load data
        analyzer.load_data()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate summary report
        print("📝 Generating summary report...")
        report = analyzer.generate_summary_report()
        
        # Save report to file
        report_file = output_dir / 'analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Print report to console
        print(report)
        print(f"📝 Report saved to {report_file}")
        
        # Generate visualizations
        if not args.no_plots:
            print("📊 Generating visualizations...")
            analyzer.create_visualizations(output_dir / 'plots')
        
        # Export data
        print("💾 Exporting data...")
        analyzer.export_data(output_dir / 'data')
        
        print(f"\n✅ Analysis complete! Results saved to {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
