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

        img_height = img.shape[0]
        img_width = img.shape[1]

        for c in self.sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1 <= ratio <= 10:
                if h / img_height >= 0.2:
                    bias = 3

                    y_1 = 0 if y - bias < 0 else y - bias
                    y_2 = (
                        img_height
                        if y + h + bias > img_height
                        else y + h + bias
                    )
                    x_1 = 0 if x - bias < 0 else x - bias
                    x_2 = (
                        img_width if x + w + bias > img_width else x + w + bias
                    )

                    curr_num = img[y_1:y_2, x_1:x_2]
                    crop_characters.append(curr_num)

        return crop_characters
