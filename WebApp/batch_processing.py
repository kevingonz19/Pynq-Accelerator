import os
import requests
import base64
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
import statistics


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
    return response.json() if response.ok else None

def softmax(logits):
    scaled = np.array(logits) / 100.0  # Match JavaScript behavior
    exp_logits = np.exp(scaled)
    return exp_logits / np.sum(exp_logits)

# Setup
image_dir = "batch_images"
output_dir = "batch_graphs"
os.makedirs(output_dir, exist_ok=True)

# Storage for results
filenames = []
cpu_acc = []
fpga_acc = []
cpu_time = []
fpga_time = []
cpu_conf_gap = []
fpga_conf_gap = []

# Inference loop
for filename in sorted(os.listdir(image_dir)):
    if not filename.endswith(".png"):
        continue

    path = os.path.join(image_dir, filename)
    true_label = int(filename.split('_')[0])
    b64_img = image_to_base64(path)

    cpu_result = request_prediction(b64_img, "cpu")
    fpga_result = request_prediction(b64_img, "fpga")

    if not cpu_result or not fpga_result:
        continue

    cpu_logits = np.array(cpu_result["res"])
    fpga_logits = np.array(fpga_result["res"])
    cpu_soft = softmax(cpu_logits)
    fpga_soft = softmax(fpga_logits)

    filenames.append(filename)
    cpu_acc.append(1 if np.argmax(cpu_logits) == true_label else 0)
    fpga_acc.append(1 if np.argmax(fpga_logits) == true_label else 0)
    cpu_time.append(cpu_result["process_time"])
    fpga_time.append(fpga_result["process_time"])
    cpu_conf_gap.append(1.0 - cpu_soft[true_label])
    fpga_conf_gap.append(1.0 - fpga_soft[true_label])

# Accuracy Plot
plt.figure(figsize=(12, 5))
plt.plot(filenames, cpu_acc, marker='o', label="CPU Accuracy")
plt.plot(filenames, fpga_acc, marker='x', label="FPGA Accuracy")
plt.xticks(ticks=range(len(filenames)), labels=range(len(filenames)))  # show index instead of filenames
plt.title("Prediction Accuracy Per Image")
plt.ylabel("Correct (1) or Incorrect (0)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_per_image.png"))
plt.close()

# Inference Time Plot
plt.figure(figsize=(12, 5))
plt.plot(filenames, cpu_time, marker='o', label="CPU Inference Time (ms)")
plt.plot(filenames, fpga_time, marker='x', label="FPGA Inference Time (ms)")
plt.xticks(ticks=range(len(filenames)), labels=range(len(filenames)))  # show index instead of filenames
plt.title("Inference Time Per Image")
plt.ylabel("Time (ms)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "inference_time_per_image.png"))
plt.close()

# True Label Confidence Gap Plot
plt.figure(figsize=(12, 5))
plt.plot(filenames, cpu_conf_gap, marker='o', label="CPU Confidence Gap (1 - P[true])")
plt.plot(filenames, fpga_conf_gap, marker='x', label="FPGA Confidence Gap (1 - P[true])")
plt.xticks(ticks=range(len(filenames)), labels=range(len(filenames)))  # show index instead of filenames
plt.title("True Label Confidence Gap Per Image")
plt.ylabel("Gap (1 - softmax[true])")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "true_confidence_gap_per_image.png"))
plt.close()

# Optional: Debugging Visualization (values on bars)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure(figsize=(12, 5))
plt.plot(filenames, cpu_conf_gap, marker='o', label="CPU")
plt.plot(filenames, fpga_conf_gap, marker='x', label="FPGA")
for i, val in enumerate(cpu_conf_gap):
    plt.text(i, val + 0.01, f"{val:.2f}", ha='center', fontsize=8)
for i, val in enumerate(fpga_conf_gap):
    plt.text(i, val - 0.04, f"{val:.2f}", ha='center', fontsize=8)
plt.xticks(ticks=range(len(filenames)), labels=range(len(filenames)))  # show index instead of filenames
plt.title("True Label Confidence Gap Per Image (with values)")
plt.ylabel("Gap")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"true_confidence_gap_debug_{timestamp}.png"))
plt.close()

summary_stats = {
    "cpu_accuracy_mean": statistics.mean(cpu_acc),
    "cpu_accuracy_median": statistics.median(cpu_acc),
    "fpga_accuracy_mean": statistics.mean(fpga_acc),
    "fpga_accuracy_median": statistics.median(fpga_acc),

    "cpu_time_mean": statistics.mean(cpu_time),
    "cpu_time_median": statistics.median(cpu_time),
    "fpga_time_mean": statistics.mean(fpga_time),
    "fpga_time_median": statistics.median(fpga_time),

    "cpu_conf_gap_mean": statistics.mean(cpu_conf_gap),
    "cpu_conf_gap_median": statistics.median(cpu_conf_gap),
    "fpga_conf_gap_mean": statistics.mean(fpga_conf_gap),
    "fpga_conf_gap_median": statistics.median(fpga_conf_gap),
}

summary_path = os.path.join(output_dir, f"summary_stats_{timestamp}.txt")
with open(summary_path, "w") as f:
    for k, v in summary_stats.items():
        f.write(f"{k}: {v:.4f}\n")
print(f"📄 Summary stats saved to: {summary_path}")

print(f"✅ All batch graphs saved to: {output_dir}")
