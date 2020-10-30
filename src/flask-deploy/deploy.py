from flask import Flask, request
from model import predict_base64
from flask_cors import CORS
import cv2

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['POST'])
def hello_world():
    data = request.json

    result = predict_base64(data['img'])

    return {"result": result}


if __name__ == '__main__':
    app.run()
