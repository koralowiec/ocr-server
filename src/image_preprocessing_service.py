import cv2
import numpy as np
from PIL import Image


class ImagePreprocessingService:
    def get_separate_characters_from_image(self, img_bytes: bytes):
        image_org = self.from_bytes_to_image(img_bytes)
        image = self.prepare_for_segmentation(image_org)
        cont = self.get_contours(image)
        characters = self.get_rects(cont, image_org)

        print("chars: ", len(characters))

        # saving doesn't work
        for i in range(len(characters)):
            print(i)
            cv2.imwrite(f"numbers/{i}.jpeg", characters[i])

        return characters

    def from_bytes_to_image(self, img_bytes: bytes):
        np_img = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

    def prepare_for_segmentation(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.threshold(
            blur, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        return binary

    def get_contours(self, img):
        cont, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cont

    def sort_contours(self, cnts, reverse=False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(
            *sorted(
                zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse
            )
        )
        return cnts

    def get_rects(self, cont, img):
        crop_characters = []

        for c in self.sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1 <= ratio <= 3.5:
                if h / img.shape[0] >= 0.4:
                    # curr_num = img[y : y + h, x : x + w]
                    curr_num = img[y - 3 : y + h + 3, x - 3 : x + w + 3]
                    crop_characters.append(curr_num)

        return crop_characters
