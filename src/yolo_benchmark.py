import time
import torch
import cv2
import numpy as np
import argparse
import psutil
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# ---- Fix for PyTorch 2.6 ----
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
torch.serialization.add_safe_globals([DetectionModel, Sequential])

# ---- GPU Metrics (Hardware Layer) ----
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

def get_os_snapshot(proc):
    """
    Captures atomic snapshot of OS state.
    Relevant for: Process Scheduling, Memory Management, and Interrupts.
    """
    with proc.oneshot():
        cpu_times = proc.cpu_times()
        ctx = proc.num_ctx_switches()
        mem = proc.memory_info()
    
    snapshot = {
        # CPU Time (Kernel vs User)
        "cpu_user": cpu_times.user,
        "cpu_system": cpu_times.system,
        
        # Scheduler (Context Switches)
        "ctx_voluntary": ctx.voluntary,       # Waiting for I/O (PCIe/Disk)
        "ctx_involuntary": ctx.involuntary,   # CPU Contention
        
        # Memory (Virtual Memory)
        "page_faults": getattr(mem, 'pfaults', 0),
        "ram_used_mb": mem.rss / 1024**2,
        
        # GPU Hardware Telemetry
        "gpu_util": 0, "gpu_mem": 0, "gpu_watts": 0, "gpu_temp": 0
    }

    if GPU_AVAILABLE:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            snapshot["gpu_util"] = util.gpu
            snapshot["gpu_mem"] = mem_info.used / 1024**2
            snapshot["gpu_temp"] = pynvml.nvmlDeviceGetTemperature(gpu_handle, 0)
            try:
                snapshot["gpu_watts"] = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0
            except:
                snapshot["gpu_watts"] = 75.0 # Fallback if sensor missing
        except:
            pass
    return snapshot

def load_image(image_path):
    """Loads image from the specific CLI path provided"""
    img = cv2.imread(str(image_path))
    if img is None: 
        raise FileNotFoundError(f"Could not load image at: {image_path}")
    return img

def benchmark(batch_size, image_path, model_name, loops=10, save_output=False):
    print(f"\n=== Benchmark: {model_name} | Batch: {batch_size} | Image: {image_path} ===")
    
    model = YOLO(model_name)
    img = load_image(image_path)
    imgs = [img for _ in range(batch_size)]

    # 1. WARMUP (Pure Compute - No Disk I/O)
    print("   -> Warming up...")
    for _ in range(2):
        model(imgs, verbose=False)

    # 2. START MEASUREMENT
    proc = psutil.Process()
    start_stats = get_os_snapshot(proc)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    # The Benchmark Loop
    for _ in range(loops):
        model(imgs, device=0 if torch.cuda.is_available() else "cpu", verbose=False) 
    end_event.record()
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    # 3. END MEASUREMENT
    end_stats = get_os_snapshot(proc)
    
    # Timing Calculations
    if torch.cuda.is_available():
        total_time_ms = start_event.elapsed_time(end_event)
    else:
        # CPU Fallback timing would go here, but we assume GPU for this project
        total_time_ms = 0 
        
    avg_time_ms = total_time_ms / loops
    fps = (batch_size * 1000) / avg_time_ms if avg_time_ms > 0 else 0

    print(f"   -> FPS: {fps:.2f} | System Time: {end_stats['cpu_system'] - start_stats['cpu_system']:.3f}s")

# 4. VISUAL PROOF (Manual Save Fix)
    if save_output:
        print("   -> Generating visual proof...")
        # Run inference on just one image
        results = model(img, verbose=False) 
        annotated_frame = results[0].plot()
        filename = f"output_{model_name}_batch{batch_size}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"   -> IMAGE SAVED: {filename}")

    return {
        "model": model_name,
        "batch": batch_size,
        "fps": round(fps, 2),
        "latency_ms": round(avg_time_ms, 2),
        "gpu_mem_mb": round(end_stats['gpu_mem'], 1),
        "gpu_watts": end_stats['gpu_watts'],
        "gpu_temp_c": end_stats['gpu_temp'],
        # OS Metrics (Deltas)
        "cpu_system_s": round(end_stats['cpu_system'] - start_stats['cpu_system'], 3),
        "ctx_vol_delta": end_stats['ctx_voluntary'] - start_stats['ctx_voluntary'],
        "page_faults_delta": end_stats['page_faults'] - start_stats['page_faults'],
    }

def create_dummy_image(path: str):
    print(f"Creating dummy image at {path}...")
    cv2.imwrite(path, np.zeros((640, 640, 3), dtype=np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 OS Benchmark Tool")
    
    # --- ARGUMENTS ---
    parser.add_argument("-i", "--image", type=str, default="test.jpg", 
                        help="Path to input image (default: test.jpg)")
    parser.add_argument("-m", "--model", type=str, default="yolov8l.pt", 
                        help="Model size (n, s, m, l, x)")
    parser.add_argument("-b", "--batches", type=int, nargs="+", default=[1, 4, 8], 
                        help="List of batch sizes to test")
    parser.add_argument("-l", "--loops", type=int, default=10, 
                        help="Number of inference loops for averaging")
    parser.add_argument("--save", action="store_true", 
                        help="Save output image with bounding boxes (Slows down script)")
    parser.add_argument("--create-dummy", action="store_true", 
                        help="Create a black dummy image if file is missing")

    args = parser.parse_args()
    
    # Image Validation Logic
    image_path = Path(args.image)
    if not image_path.exists():
        if args.create_dummy: 
            create_dummy_image(str(image_path))
        else:
            print(f"Error: Image '{image_path}' not found.")
            print("Tip: Use --create-dummy to generate one, or download a real one.")
            exit(1)

    results = []
    
    for b in args.batches:
        try:
            res = benchmark(b, str(image_path), args.model, args.loops, args.save)
            results.append(res)
        except torch.cuda.OutOfMemoryError:
            print(f"   -> Batch {b} FAILED: CUDA Out Of Memory")
            results.append({"batch": b, "error": "OOM"})
            break 
        except Exception as e:
            print(f"   -> Batch {b} FAILED: {e}")
            break

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        csv_filename = f"results_{args.model.replace('.pt','')}_local.csv"
        df.to_csv(csv_filename, index=False)
        
        print("\n=== FINAL RESULTS ===")
        print(f"{'Batch':<6} | {'FPS':<10} | {'Sys Time':<10} | {'Pg Faults':<10}")
        print("-" * 45)
        for r in results:
            if "error" not in r:
                print(f"{r['batch']:<6} | {r['fps']:<10.2f} | {r['cpu_system_s']:<10.3f} | {r['page_faults_delta']:<10}")
            else:
                print(f"{r['batch']:<6} | {r['error']}")
                
        print(f"\nDetailed OS metrics saved to: {csv_filename}")
        if args.save:
            print(f"Visual outputs saved to: runs/benchmark/")