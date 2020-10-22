from flask import Flask, request
from model import predict_base64

app = Flask(__name__)


# file_name = 'test.jpg'
# with open(file_name, "rb") as f:
#     base64_str = base64.b64encode(f.read()).decode()

@app.route('/', methods=['POST'])
def hello_world():
    data = request.json

    result = predict_base64(data['img'])

    return {"result": result}


if __name__ == '__main__':
    app.run()
