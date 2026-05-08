# 🌬️ AI-Powered Wind Turbine Blade Defect Detection (Edge Deployment)
> M1 Project – ISEN Brest (2026)
> **Nolan Nedelec** & **Cassandre Poyen** 

---

## 📋 Overview

Wind turbine blade inspection is a **dangerous and costly** operation, traditionally carried out by technicians working at height or via rope access systems. This project proposes an **autonomous embedded alternative**: an AI model deployed on a drone, capable of detecting structural blade defects in real time.

### Detected Defect Classes

| Class | Description |
|-------|-------------|
| `Crack` | Surface cracks |
| `Erosion` | Erosion / Corrosion |
| `Damage` | Various structural damage |

---

## ⚡ Performance

| Metric | Value |
|--------|-------|
| Accuracy (mAP50) | **82.1%** |
| Throughput | **30 FPS** |
| Total Latency | **31 – 33 ms** |
| Recommended Confidence Threshold | **0.35 – 0.40** |
| Target Hardware | NVIDIA Jetson Orin Nano |

---

## 🏗️ Technical Architecture

### Model
- **YOLOv8m** — selected after comparison with YOLOv11 and YOLOv26
- Dataset: **1,886** real-world images, cleaned and balanced

### Embedded Optimizations
- **TensorRT (FP16)** conversion via `trtexec` directly on the Jetson
- **NumPy** vectorization of post-processing to eliminate CPU bottlenecks

---

## 📁 Repository Structure

```
.
├── models/
│   ├── best.pt               # ⚠️ See Releases section (~50 MB)
│   └── JETSON.md             # Conversion and deployment guide for Jetson
│                             # best.onnx and best.engine NOT versioned (too large / hardware-specific)
│                             # → See Deployment section to generate them
│
├── src/
│   ├── app.py                # Gradio interface (local test / webcam)
│   └── live_jetson.py        # Real-time inference optimized for Jetson
│
├── data/
│   └── data_img/             # Test images (defect examples)
│
└── requirements.txt
```

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/nolannedelec/YOLOv8-WindTurbine-Inspection.git
cd YOLOv8-WindTurbine-Inspection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> Dependencies: `ultralytics`, `gradio`, `numpy`, `opencv-python`

### 3. Download the model
The `best.pt` file is ~50 MB and **cannot be stored directly in the repository**. It is available in the **[Releases](https://github.com/nolannedelec/YOLOv8-WindTurbine-Inspection/releases)** section of this repository.

Place it in the `models/` folder once downloaded.

### 4. Launch the Gradio interface (local test)
```bash
python src/app.py
```
The interface allows you to:
- Choose the input source (uploaded image or webcam stream)
- Dynamically adjust the confidence threshold
- Visualize the annotated image along with a summary of detected defects

---

## 🤖 Deployment on NVIDIA Jetson Orin Nano

The complete conversion and deployment guide (Jetson setup, `.pt` → `.onnx` → `.engine` pipeline, running inference, performance comparison) is available here:

📖 **[models/JETSON.md](./models/JETSON.md)**

---

## 📦 requirements.txt

```
ultralytics
gradio
numpy
opencv-python
```

---

*Project developed as part of the Master 1 program at **ISEN Brest** – 2026*
