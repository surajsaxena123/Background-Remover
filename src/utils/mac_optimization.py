"""
Mac-specific optimizations and compatibility enhancements
Optimized for both Intel and Apple Silicon (M1/M2/M3) Macs
"""

import platform
import os
import sys
import logging
from typing import Optional, Dict, Any
import subprocess

def detect_mac_architecture() -> Dict[str, Any]:
    """Detect Mac architecture and optimization capabilities"""
    
    system_info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "is_mac": platform.system() == "Darwin",
        "is_apple_silicon": False,
        "is_intel": False,
        "supports_metal": False,
        "supports_mps": False,
        "cpu_count": os.cpu_count(),
        "memory_gb": None,
        "optimization_recommendations": []
    }
    
    if system_info["is_mac"]:
        # Detect Apple Silicon vs Intel
        if platform.machine() in ["arm64", "aarch64"]:
            system_info["is_apple_silicon"] = True
            system_info["optimization_recommendations"].extend([
                "Use MPS (Metal Performance Shaders) for GPU acceleration",
                "Enable ARM64-optimized libraries",
                "Use native Apple Silicon dependencies"
            ])
        elif platform.machine() in ["x86_64", "AMD64"]:
            system_info["is_intel"] = True
            system_info["optimization_recommendations"].extend([
                "Consider using Intel MKL optimizations",
                "Enable x86_64 optimized libraries"
            ])
        
        # Check for Metal Performance Shaders support
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                system_info["supports_mps"] = True
                system_info["supports_metal"] = True
                system_info["optimization_recommendations"].append(
                    "MPS acceleration available for PyTorch operations"
                )
        except ImportError:
            pass
        
        # Get memory information
        try:
            result = subprocess.run(['sysctl', 'hw.memsize'], 
                                  capture_output=True, text=True, check=True)
            memory_bytes = int(result.stdout.split(':')[1].strip())
            system_info["memory_gb"] = round(memory_bytes / (1024**3), 1)
            
            if system_info["memory_gb"] >= 16:
                system_info["optimization_recommendations"].append(
                    "Sufficient memory for large image processing"
                )
            elif system_info["memory_gb"] >= 8:
                system_info["optimization_recommendations"].append(
                    "Consider memory optimization for large images"
                )
            else:
                system_info["optimization_recommendations"].append(
                    "Limited memory - enable tiling for large images"
                )
        except (subprocess.CalledProcessError, ValueError, IndexError):
            pass
    
    return system_info

def setup_mac_environment() -> Dict[str, Any]:
    """Setup optimal environment for Mac"""
    
    mac_info = detect_mac_architecture()
    optimizations = {
        "opencv_threads": None,
        "numpy_threads": None,
        "device": "cpu",
        "memory_fraction": 0.8,
        "use_mps": False,
        "use_mkl": False
    }
    
    if not mac_info["is_mac"]:
        return optimizations
    
    # Set optimal thread counts
    cpu_count = mac_info.get("cpu_count", 4)
    optimizations["opencv_threads"] = min(cpu_count, 8)
    optimizations["numpy_threads"] = min(cpu_count, 4)
    
    # Configure OpenCV for Mac
    try:
        import cv2
        cv2.setNumThreads(optimizations["opencv_threads"])
        cv2.setUseOptimized(True)
    except ImportError:
        pass
    
    # Configure NumPy threading
    try:
        import numpy as np
        if hasattr(np, '__config__'):
            os.environ['OMP_NUM_THREADS'] = str(optimizations["numpy_threads"])
            os.environ['MKL_NUM_THREADS'] = str(optimizations["numpy_threads"])
    except ImportError:
        pass
    
    # Configure MPS (Metal Performance Shaders) for Apple Silicon
    if mac_info["supports_mps"]:
        optimizations["device"] = "mps"
        optimizations["use_mps"] = True
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Memory optimization based on available RAM
    memory_gb = mac_info.get("memory_gb", 8)
    if memory_gb >= 32:
        optimizations["memory_fraction"] = 0.9
    elif memory_gb >= 16:
        optimizations["memory_fraction"] = 0.8
    else:
        optimizations["memory_fraction"] = 0.6
    
    return optimizations

def get_optimal_device() -> str:
    """Get the optimal device for processing on Mac"""
    
    # Check for MPS (Apple Silicon GPU)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    
    # Fallback to CPU with optimizations
    return "cpu"

def optimize_for_mac_processing(image_size: tuple) -> Dict[str, Any]:
    """Get optimal processing parameters for Mac based on image size"""
    
    mac_info = detect_mac_architecture()
    h, w = image_size[:2]
    image_pixels = h * w
    
    params = {
        "tile_size": (1024, 1024),
        "batch_size": 1,
        "num_workers": 1,
        "memory_efficient": False,
        "use_half_precision": False,
        "enable_tiling": False
    }
    
    # Memory-based optimizations
    memory_gb = mac_info.get("memory_gb", 8)
    
    if image_pixels > 2048 * 2048:  # Large images
        params["enable_tiling"] = True
        if memory_gb >= 16:
            params["tile_size"] = (2048, 2048)
        else:
            params["tile_size"] = (1024, 1024)
    
    # Apple Silicon optimizations
    if mac_info["is_apple_silicon"]:
        params["use_half_precision"] = True  # Better performance on Apple Silicon
        if memory_gb >= 16:
            params["batch_size"] = 2
        
        # Optimize worker count for Apple Silicon
        cpu_count = mac_info.get("cpu_count", 8)
        params["num_workers"] = min(cpu_count // 2, 4)
    
    # Intel Mac optimizations
    elif mac_info["is_intel"]:
        if memory_gb >= 32:
            params["batch_size"] = 4
        elif memory_gb >= 16:
            params["batch_size"] = 2
        
        cpu_count = mac_info.get("cpu_count", 4)
        params["num_workers"] = min(cpu_count // 2, 6)
    
    return params

def setup_mac_opencv() -> None:
    """Configure OpenCV optimally for Mac"""
    
    try:
        import cv2
        
        # Enable optimizations
        cv2.setUseOptimized(True)
        
        # Set thread count based on CPU
        mac_info = detect_mac_architecture()
        if mac_info["is_mac"]:
            thread_count = min(mac_info.get("cpu_count", 4), 8)
            cv2.setNumThreads(thread_count)
        
        # Enable Intel IPP optimizations on Intel Macs
        if mac_info.get("is_intel", False):
            try:
                # Try to enable Intel Integrated Performance Primitives
                os.environ['OPENCV_IPP_ENABLE'] = '1'
            except:
                pass
        
        logging.info(f"OpenCV configured for Mac: threads={cv2.getNumThreads()}, optimized={cv2.useOptimized()}")
        
    except ImportError:
        logging.warning("OpenCV not available for Mac optimization")

def get_mac_memory_info() -> Dict[str, float]:
    """Get detailed memory information on Mac"""
    
    memory_info = {
        "total_gb": 0.0,
        "available_gb": 0.0,
        "used_percentage": 0.0,
        "recommended_limit_gb": 0.0
    }
    
    try:
        # Get total memory
        result = subprocess.run(['sysctl', 'hw.memsize'], 
                              capture_output=True, text=True, check=True)
        total_bytes = int(result.stdout.split(':')[1].strip())
        memory_info["total_gb"] = round(total_bytes / (1024**3), 1)
        
        # Get memory pressure (approximate available memory)
        try:
            result = subprocess.run(['vm_stat'], 
                                  capture_output=True, text=True, check=True)
            lines = result.stdout.split('\n')
            
            free_pages = 0
            inactive_pages = 0
            page_size = 4096  # Default page size
            
            for line in lines:
                if 'page size of' in line:
                    page_size = int(line.split()[-2])
                elif 'Pages free:' in line:
                    free_pages = int(line.split(':')[1].strip().rstrip('.'))
                elif 'Pages inactive:' in line:
                    inactive_pages = int(line.split(':')[1].strip().rstrip('.'))
            
            available_bytes = (free_pages + inactive_pages) * page_size
            memory_info["available_gb"] = round(available_bytes / (1024**3), 1)
            memory_info["used_percentage"] = round(
                (1 - memory_info["available_gb"] / memory_info["total_gb"]) * 100, 1
            )
            
        except (subprocess.CalledProcessError, ValueError, IndexError):
            # Fallback estimate
            memory_info["available_gb"] = memory_info["total_gb"] * 0.7
            memory_info["used_percentage"] = 30.0
        
        # Recommended processing limit (leave some memory for system)
        memory_info["recommended_limit_gb"] = memory_info["total_gb"] * 0.8
        
    except (subprocess.CalledProcessError, ValueError, IndexError):
        logging.warning("Could not get Mac memory information")
    
    return memory_info

def log_mac_system_info() -> None:
    """Log comprehensive Mac system information"""
    
    mac_info = detect_mac_architecture()
    memory_info = get_mac_memory_info()
    
    if mac_info["is_mac"]:
        logging.info("=== Mac System Information ===")
        logging.info(f"Architecture: {mac_info['machine']}")
        logging.info(f"Apple Silicon: {mac_info['is_apple_silicon']}")
        logging.info(f"Intel Mac: {mac_info['is_intel']}")
        logging.info(f"CPU Cores: {mac_info['cpu_count']}")
        logging.info(f"Total Memory: {memory_info['total_gb']:.1f} GB")
        logging.info(f"Available Memory: {memory_info['available_gb']:.1f} GB")
        logging.info(f"MPS Support: {mac_info['supports_mps']}")
        logging.info(f"Metal Support: {mac_info['supports_metal']}")
        
        logging.info("=== Optimization Recommendations ===")
        for rec in mac_info["optimization_recommendations"]:
            logging.info(f"• {rec}")
        
        logging.info("==============================")

def initialize_mac_optimizations() -> Dict[str, Any]:
    """Initialize all Mac-specific optimizations"""
    
    # Log system information
    log_mac_system_info()
    
    # Setup environment
    optimizations = setup_mac_environment()
    
    # Configure OpenCV
    setup_mac_opencv()
    
    # Set environment variables for optimal performance
    mac_info = detect_mac_architecture()
    
    if mac_info["is_apple_silicon"]:
        # Apple Silicon optimizations
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['TORCH_USE_RTLD_GLOBAL'] = 'YES'
    elif mac_info["is_intel"]:
        # Intel Mac optimizations
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['MKL_NUM_THREADS'] = str(optimizations["numpy_threads"])
    
    # General Mac optimizations
    os.environ['OMP_NUM_THREADS'] = str(optimizations["numpy_threads"])
    os.environ['NUMEXPR_MAX_THREADS'] = str(optimizations["numpy_threads"])
    
    logging.info("Mac optimizations initialized successfully")
    
    return {
        "system_info": mac_info,
        "optimizations": optimizations,
        "memory_info": get_mac_memory_info()
    }