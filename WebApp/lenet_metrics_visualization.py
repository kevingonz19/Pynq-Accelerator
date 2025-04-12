import requests
import base64
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import os
import csv
import math
from datetime import datetime


def generate_metrics_visualization(csv_path="layer_comparison_metrics.csv", output_root="run_metrics"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, f"metrics_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df.to_csv(os.path.join(output_dir, "layer_comparison_metrics.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["cpu_time_ms"], label="CPU Time (ms)")
    plt.plot(df["timestamp"], df["fpga_time_ms"], label="FPGA Time (ms)")
    plt.title("Inference Time Comparison")
    plt.xlabel("Timestamp")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_time_comparison.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["cosine_similarity"], color='purple')
    plt.title("Logits Cosine Similarity Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cosine_similarity.png"))
    plt.close()

    if "cpu_memory_kb" in df and "fpga_memory_kb" in df:
        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["cpu_memory_kb"], label="CPU Memory (KB)")
        plt.plot(df["timestamp"], df["fpga_memory_kb"], label="FPGA Memory (KB)")
        plt.title("Memory Usage Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Memory (KB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "memory_usage.png"))
        plt.close()

    layers = ["conv1", "conv2", "fc1", "pool1", "pool2"]
    for layer in layers:
        plt.figure(figsize=(10, 4))
        plt.plot(df["timestamp"], df[f"{layer}_mean_abs_diff"], label=f"{layer.upper()} Mean Diff")
        plt.plot(df["timestamp"], df[f"{layer}_max_abs_diff"], label=f"{layer.upper()} Max Diff", linestyle='--')
        plt.title(f"{layer.upper()} Layer Difference Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Absolute Difference")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{layer}_diffs.png"))
        plt.close()

    print(f"Metrics visualized and saved in: {output_dir}")

def plot_layer_grid(cpu_layer, fpga_layer, layer_name, save_dir="heatmap_outputs"):
    os.makedirs(save_dir, exist_ok=True)
    cpu_layer = np.array(cpu_layer)
    fpga_layer = np.array(fpga_layer)

    if cpu_layer.ndim == 1:
        side = int(math.ceil(math.sqrt(cpu_layer.shape[0])))
        cpu_layer = np.pad(cpu_layer, (0, side**2 - cpu_layer.shape[0]), mode='constant').reshape((side, side))
        fpga_layer = np.pad(fpga_layer, (0, side**2 - fpga_layer.shape[0]), mode='constant').reshape((side, side))
        diff_layer = np.abs(cpu_layer - fpga_layer)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        titles = [f"CPU {layer_name}", f"FPGA {layer_name}", f"Diff {layer_name}"]
        maps = [cpu_layer, fpga_layer, diff_layer]
        for ax, data, title in zip(axs, maps, titles):
            im = ax.imshow(data, cmap='hot')
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{layer_name}_fc.png"))
        plt.close()
        return

    channels = cpu_layer.shape[0]
    cols = 5
    rows = math.ceil(channels / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.suptitle(f"{layer_name} Channel-wise Diff", fontsize=16)
    axs = axs.flatten()
    for i in range(rows * cols):
        ax = axs[i]
        if i < channels:
            diff = np.abs(cpu_layer[i] - fpga_layer[i])
            im = ax.imshow(diff, cmap='hot')
            ax.set_title(f"Ch {i}")
            fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{layer_name}_diff_grid.png"))
    plt.close()

def image_to_base64(image_path):
    img = Image.open(image_path).resize((28, 28)).convert('L')
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def request_prediction(base64_img, process_type):
    payload = {
        "net": "lenet",
        "type": process_type,
        "img": f"data:image/png;base64,{base64_img}"
    }
    response = requests.post("http://192.168.137.147:5000/predict_with_metrics", data=payload)
    print(f"[{process_type.upper()}] Status Code:", response.status_code)
    print(f"[{process_type.upper()}] Response Text:", response.text[:300])
    try:
        return response.json()
    except Exception as e:
        print("Error decoding JSON:", e)
        return None

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_layer_diffs(cpu_maps, fpga_maps):
    diffs = {}
    for layer in cpu_maps:
        if layer in fpga_maps:
            cpu = np.array(cpu_maps[layer])
            fpga = np.array(fpga_maps[layer])
            if cpu.shape == fpga.shape:
                abs_diff = np.abs(cpu - fpga)
                diffs[layer] = {
                    "mean_abs_diff": np.mean(abs_diff),
                    "max_abs_diff": np.max(abs_diff)
                }
    return diffs

def log_all_metrics(csv_path, meta, diffs):
    file_exists = os.path.exists(csv_path)
    fieldnames = list(meta.keys()) + [f"{layer}_{k}" for layer in diffs for k in diffs[layer]]
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = dict(meta)
        for layer, stats in diffs.items():
            for k, v in stats.items():
                row[f"{layer}_{k}"] = v
        writer.writerow(row)

def visualize_layer_times(cpu_timers, fpga_timers, save_path="layer_timings.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    layers = list(cpu_timers.keys())
    cpu_times = [cpu_timers[layer] for layer in layers]
    fpga_times = [fpga_timers[layer] for layer in layers]

    x = np.arange(len(layers))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, cpu_times, width, label='CPU')
    plt.bar(x + width/2, fpga_times, width, label='FPGA')
    plt.xticks(x, layers, rotation=45)
    plt.ylabel("Execution Time (ms)")
    plt.title("Layer-wise Execution Time: CPU vs FPGA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"🕒 Layer timings saved to {save_path}")

# Run
image_path = "test_digit.png"
b64_img = image_to_base64(image_path)

cpu_result = request_prediction(b64_img, "cpu")
fpga_result = request_prediction(b64_img, "fpga")

if cpu_result and fpga_result:
    cpu_logits = np.array(cpu_result["res"])
    fpga_logits = np.array(fpga_result["res"])

    cpu_conf = np.max(softmax(cpu_logits))
    fpga_conf = np.max(softmax(fpga_logits))
    prediction_match = np.argmax(cpu_logits) == np.argmax(fpga_logits)
    cosine_sim = cosine_similarity(cpu_logits, fpga_logits)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = f"run_outputs/outputs_{timestamp}"
    os.makedirs(root_dir, exist_ok=True)

    csv_path = os.path.join(root_dir, "layer_comparison_metrics.csv")
    meta = {
        "timestamp": datetime.now().isoformat(),
        "cpu_time_ms": cpu_result["process_time"],
        "fpga_time_ms": fpga_result["process_time"],
        "prediction_match": prediction_match,
        "cpu_confidence": cpu_conf,
        "fpga_confidence": fpga_conf,
        "cosine_similarity": cosine_sim,
        "cpu_memory_kb": cpu_result.get("memory_kb", 0),
        "fpga_memory_kb": fpga_result.get("memory_kb", 0)
    }
    diffs = compute_layer_diffs(cpu_result["feature_maps"], fpga_result["feature_maps"])
    log_all_metrics(csv_path, meta, diffs)

    heatmap_dir = os.path.join(root_dir, "heatmaps")
    for layer in cpu_result["feature_maps"]:
        if layer in fpga_result["feature_maps"]:
            plot_layer_grid(cpu_result["feature_maps"][layer], fpga_result["feature_maps"][layer], layer, save_dir=heatmap_dir)

    generate_metrics_visualization(csv_path=csv_path, output_root=root_dir)

    visualize_layer_times(
        cpu_result.get("timers", {}),
        fpga_result.get("timers", {}),
        save_path=os.path.join(root_dir, "layer_timings.png")
    )
