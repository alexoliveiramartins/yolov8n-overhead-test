import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# ---- Fix for PyTorch 2.6 ----
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
torch.serialization.add_safe_globals([DetectionModel, Sequential])

# ---- GPU Metrics ----
try:
    import pynvml # acts as nvidia-ml-py
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_AVAILABLE = True
except Exception as e:
    GPU_AVAILABLE = False
    print(f"GPU Monitor skipped: {e}")

def get_gpu_stats():
    if not GPU_AVAILABLE:
        return {"util": 0, "mem": 0}
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return {"util": util.gpu, "mem": mem.used / 1e6} # MB

def benchmark(batch_size: int):
    print(f"\n=== Benchmark: batch={batch_size} ===")
    
    # 1. Load Model
    model = YOLO("yolov8m.pt")
    
    # 2. Prepare Data
    img = cv2.imread("test.jpg")
    if img is None: raise FileNotFoundError("test.jpg not found!")
    imgs = [img for _ in range(batch_size)]

    # 3. Warmup (Critical for GPU)
    # We run a few dummy passes so the driver initializes kernels
    print("Warming up...")
    for _ in range(3):
        model(imgs, verbose=False)

    # ---- CPU Benchmark ----
    start = time.perf_counter()
    model(imgs, device="cpu", verbose=False)
    end = time.perf_counter()
    cpu_time_ms = (end - start) * 1000
    cpu_fps = batch_size / (end - start)

    # ---- GPU Benchmark ----
    if torch.cuda.is_available():
        # Use CUDA Events for precise GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Force sync before starting
        torch.cuda.synchronize()
        
        start_event.record()
        # Explicitly set device=0 for GPU
        results = model(imgs, device=0, verbose=False) 
        end_event.record()
        
        # Wait for GPU to finish
        torch.cuda.synchronize()
        
        gpu_time_ms = start_event.elapsed_time(end_event) # Returns ms
        gpu_fps = batch_size / (gpu_time_ms / 1000)
        
        stats = get_gpu_stats()
    else:
        gpu_time_ms = 0
        gpu_fps = 0
        stats = {"util": 0, "mem": 0}

    print(f"CPU: {cpu_time_ms:.2f} ms | {cpu_fps:.2f} FPS")
    if torch.cuda.is_available():
        print(f"GPU: {gpu_time_ms:.2f} ms | {gpu_fps:.2f} FPS")
        print(f"VRAM: {stats['mem']:.1f} MB")

    return {
        "batch": batch_size,
        "cpu_ms": cpu_time_ms,
        "gpu_ms": gpu_time_ms,
        "gpu_fps": gpu_fps,
        "vram_mb": stats['mem']
    }

if __name__ == "__main__":
    # Create a dummy image if not exists
    if not cv2.imread("test.jpg") is not None:
        print("Creating dummy test.jpg...")
        cv2.imwrite("test.jpg", np.zeros((640, 640, 3), dtype=np.uint8))

    results = []
    # 2GB VRAM limit warning: Batch 8 might fail on GTX 1050 depending on image size
    for b in [1, 2, 4, 8]: 
        try:
            results.append(benchmark(batch_size=b))
        except torch.cuda.OutOfMemoryError:
            print(f"Batch {b} failed: CUDA Out Of Memory")
            break

    print("\n=== FINAL RESULTS (GPU) ===")
    print(f"{'Batch':<6} | {'Time (ms)':<10} | {'FPS':<10} | {'VRAM (MB)':<10}")
    print("-" * 45)
    for r in results:
        print(f"{r['batch']:<6} | {r['gpu_ms']:<10.2f} | {r['gpu_fps']:<10.2f} | {r['vram_mb']:<10.1f}")