from flask import Flask, render_template, request
import io
import base64
import cv2
import numpy as np
from keras.models import load_model

init_Base64 = 21  # data:image/png;base64, jpeg => 22
app = Flask(__name__)


@app.route('/')
def home():
    return render_template("mnist.html")


@app.route('/upload', methods=['POST'])
def upload():
    draw = request.form['url']
    draw = draw[init_Base64:]
    draw_decoded = base64.b64decode(draw)
    image = np.asarray(bytearray(draw_decoded), dtype="uint8")

    mode = request.form.get("mode", "digit")

    if mode == "digit":
        model = load_model('mnist_cnn_tiny.h5')
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        cv2.imwrite('test.png', image)
        image = image.reshape(1, 28, 28, 1)

        p = model.predict(image)
        p = np.argmax(p)

    else:
        model_H = load_model('hand_written_korean_classification.hdf5')
        labels_file = io.open("label.txt", 'r', encoding="UTF-8").read().split()
        label = [str for str in labels_file]
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_AREA)
        image = (255 - image) / 255
        cv2.imwrite('test.png', image)
        image = image.reshape(1, 32, 32, 3)

        p = model_H.predict(image)
        p = label[np.argmax(p)]

    return f"result : {p} <a href=javascript:history.back()>뒤로</a>"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
