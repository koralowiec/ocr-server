from flask import Flask, request, Response
import json
import pytesseract
import cv2
import numpy as np
import base64

app = Flask(__name__)

characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def run_ocr(image, psm):
    tesseract_conf = f"-c tessedit_char_whitelist={characters} --psm {psm}"
    number = pytesseract.image_to_string(image, config=tesseract_conf)
    print(f"Number (psm: {psm}): {number}")

    return clean_up_number(number)


def clean_up_number(number):
    bad_chars = ["\n\x0c", "\x0c"]

    for char in bad_chars:
        number = number.replace(char, "")

    return number


@app.route("/ocr", methods=["POST"])
def ocr():
    is_image_in_base64 = request.args.get("base64", default=False)

    if is_image_in_base64:
        img = request.json["img"]
        img_bytes = base64.b64decode(img)
        np_img = np.frombuffer(img_bytes, dtype=np.uint8)
    else:
        img = request.files["file"]
        np_img = np.fromstring(img.read(), np.uint8)

    img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

    numbers = []
    numbers.append(run_ocr(img, "9"))
    numbers.append(run_ocr(img, "11"))

    return Response(json.dumps(numbers), mimetype="application/json")
