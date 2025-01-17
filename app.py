from flask import Flask, jsonify
import torch

app = Flask(__name__)

@app.route('/gpu-info', methods=['GET'])
def get_gpu_info():
    if torch.cuda.is_available():
        gpu_info = {
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "pytorch_version": torch.__version__,
            "cudnn_version": torch.backends.cudnn.version()
        }
    else:
        gpu_info = {"error": "No GPU available or CUDA is not configured."}

    return jsonify(gpu_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
