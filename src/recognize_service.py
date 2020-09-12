from typing import List
import pytesseract
import numpy as np
import cv2


class RecognizeService:
    characters: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    bad_chars: List[str] = ["\n\x0c", "\x0c"]
    psms: List[str] = ["9", "11"]

    def get_numbers(self, img_bytes: bytes) -> List[str]:
        np_img = np.frombuffer(img_bytes, dtype=np.uint8)

        img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

        numbers = []
        for psm in self.psms:
            numbers.append(self.run_ocr(img, psm))

        return numbers

    def run_ocr(self, image: np.ndarray, psm: int) -> str:
        tesseract_conf = (
            f"-c tessedit_char_whitelist={self.characters} --psm {psm}"
        )
        number = pytesseract.image_to_string(image, config=tesseract_conf)

        return self.clean_up_number(number)

    def clean_up_number(self, number: str) -> str:
        for char in self.bad_chars:
            number = number.replace(char, "")

        return number
