# 🚀 Inference & Optimization Guide (NVIDIA Jetson)

This guide explains how to convert the trained model (`.pt`) into a high-performance compute engine (`.engine`) to achieve **30 FPS** on the NVIDIA Jetson Orin Nano.

---

## 1. Jetson Setup

Before running inference, the board must be pushed to its maximum performance:

```bash
# Enable maximum power mode
sudo nvpmodel -m 0
# Force GPU clocks to maximum
sudo jetson_clocks
```

---

## 2. Model Conversion (Two-Step Process)

The conversion is done in **two separate steps**: first on your development machine, then on the Jetson. This approach avoids installing Ultralytics on the Jetson and eliminates frequent version conflicts.

### Step 2a — On your machine: `.pt` → `.onnx`

```bash
yolo export model=models/best.pt format=onnx half=True
```

A `best.onnx` file will be generated in `models/`. Copy it to the Jetson (via `scp`, USB drive, etc.).

### Step 2b — On the Jetson: `.onnx` → `.engine`

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/best.onnx \
  --saveEngine=models/best.engine \
  --fp16
```

| Parameter | Role |
|-----------|------|
| `--onnx` | Path to the transferred ONNX file |
| `--saveEngine` | Output path for the TensorRT engine |
| `--fp16` | Enables FP16 precision to accelerate GPU inference |

> ⚠️ **This `.engine` file is hardware-specific.** It will not work on another machine and should not be committed to the repository.

---

## 3. Running Real-Time Inference

To achieve maximum throughput (30 FPS), the `live_jetson.py` script leverages **NumPy vectorization** for detection box post-processing, eliminating CPU bottlenecks.

```bash
python3 src/live_jetson.py --engine models/best.engine --source 0
```

| Argument | Description |
|----------|-------------|
| `--engine` | Path to the generated TensorRT engine |
| `--source 0` | Jetson onboard camera (or video stream index) |

---

## 💡 Why This Optimization?

| Configuration | FPS | Latency |
|---------------|-----|---------|
| Without TensorRT (raw CPU/GPU) | ~8 FPS | ~120 ms |
| **TensorRT FP16 + NumPy (vectorized)** | **30 FPS** | **~33 ms** |

A latency below **35 ms** is essential for a drone in motion to detect and localize defects in real time, without motion blur or display lag.
