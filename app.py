from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO
from LeNet import lenet_cpu, lenet_fpga
import cv2
from twilio.rest import Client
import os

# New ones
import time
import numpy as np

app = Flask(__name__)
CORS(app)

# TODO API KEYS

def get_memory_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if "VmRSS:" in line:
                return int(line.split()[1])  # in KB
    return 0

@app.route('/')
def hello_world():
    return "Hello world!"

@app.route('/predict', methods=['POST'])
def get_predict_cpu_lenet():
    net = request.form.get('net')
    process_type = request.form.get('type')
    base64_data = request.form.get('img').split(';base64,')[1]
    im = Image.open(BytesIO(base64.b64decode(base64_data))).resize((28,28)).convert('L')

    if net=="lenet" and process_type=="cpu":
        res, process_time = lenet_cpu.predict(im)
    elif net=="lenet" and process_type=="fpga":
        res, process_time = lenet_fpga.predict(im)
    else:
        res, process_time = [], 0.0

    return jsonify({"res": tuple(res), "process_time": process_time})

@app.route('/capture', methods=['POST'])
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": "Failed to capture image"}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    bw = cv2.bitwise_not(bw)
    success, buffer = cv2.imencode('.jpg', bw)
    if not success:
        return jsonify({"error": "Failed to encode image"}), 500

    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"img": "data:image/jpeg;base64," + jpg_as_text})

@app.route('/call', methods=['POST'])
def call_phone():
    to_number = request.form.get('to')
    if not to_number:
        return jsonify({"error": "Missing phone number"}), 400

    call = twilio_client.calls.create(
        to=to_number,
        from_=TWILIO_FROM,
        twiml="<Response><Say>This is a test call from PYNQ Accelerator demo.</Say></Response>"
    )
    return jsonify({"sid": call.sid}), 202

@app.route('/predict_with_metrics', methods=['POST'])
def predict_with_metrics():
    net = request.form.get('net')
    process_type = request.form.get('type')
    base64_data = request.form.get('img').split(';base64,')[1]
    im = Image.open(BytesIO(base64.b64decode(base64_data))).resize((28, 28)).convert('L')

    start = time.time()

    # Predict with feature maps and timers
    if net == "lenet" and process_type == "cpu":
        res, process_time, feature_maps, timers, memory_kb = lenet_cpu.predict(im, return_feature_map=True)
    elif net == "lenet" and process_type == "fpga":
        res, process_time, feature_maps, timers, memory_kb = lenet_fpga.predict(im, return_feature_map=True)
    else:
        res, process_time, feature_maps, timers, memory_kb = [], 0.0, {}, {}, 0

    end = time.time()
    inference_time = round((end - start) * 1000, 2)

    # Serialize feature maps
    serialized_maps = {}
    try:
        for name, fmap in feature_maps.items():
            serialized_maps[name] = np.array(fmap).tolist()
    except Exception as e:
        print("Serialization error:", e)
        serialized_maps = {}

    # Get current memory usage in KB
    memory_kb = get_memory_kb()

    return jsonify({
        "res": res.tolist() if isinstance(res, np.ndarray) else res,
        "process_time": inference_time,
        "feature_maps": serialized_maps,
        "timers": timers,
        "memory_kb": memory_kb  # Memory usage
    })

if __name__ == '__main__':
    app.run(host='192.168.2.99', port=5000)
