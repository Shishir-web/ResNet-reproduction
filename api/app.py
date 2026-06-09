from flask import Flask, request, jsonify, make_response
import onnxruntime as ort
import numpy as np
from PIL import Image
import io, base64, json

app = Flask(__name__)

session = ort.InferenceSession('models/resnet18.onnx')
with open('models/classes.json') as f:
    CLASSES = json.load(f)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def corsify(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    return response

def preprocess(image_b64):
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert('RGB')
    img = img.resize((64, 64), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1).reshape(1, 3, 64, 64)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return corsify(make_response('', 204))
    try:
        data  = request.json.get('image')
        inp   = preprocess(data)
        out   = session.run(['output'], {'input': inp})[0][0]
        probs = np.exp(out) / np.exp(out).sum()
        top5_idx = probs.argsort()[::-1][:5]
        results  = [
            {'class': CLASSES[i], 'confidence': float(probs[i])}
            for i in top5_idx
        ]
        print(f"Top prediction: {results[0]['class']} "
              f"({results[0]['confidence']:.2%})")
        return corsify(jsonify({'predictions': results}))
    except Exception as e:
        print(f"Error: {e}")
        return corsify(jsonify({'error': str(e)})), 500

if __name__ == '__main__':
    app.run(port=5001, debug=False)