from flask import Flask, request, Response
import json
import pytesseract
import cv2
import numpy as np

app = Flask(__name__)

characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def run_ocr(image, psm):
    tesseract_conf = f"-c tessedit_char_whitelist={characters} --psm {psm}"
    number = pytesseract.image_to_string(image, config=tesseract_conf)
    print(f"Number (psm: {psm}): {number}")

    return number


@app.route("/ocr", methods=["POST"])
def ocr():
    img = request.files["file"]
    np_img = np.fromstring(img.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    numbers = []
    numbers.append(run_ocr(img, "9"))
    numbers.append(run_ocr(img, "11"))

    return Response(json.dumps(numbers), mimetype="application/json")
