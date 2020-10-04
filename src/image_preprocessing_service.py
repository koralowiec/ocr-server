import cv2
import numpy as np
from typing import List
from functools import reduce
from statistics import mode, median


class ImagePreprocessingService:
    min_height_to_width_ratio: int = 1
    max_height_to_width_ratio: int = 10
    min_rect_height_to_image_height_ratio: float = 0.2

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

    def prepare_for_segmentation(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.threshold(
            blur, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        return binary

    def get_contours(self, img: np.ndarray):
        cont, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cont

    def sort_contours(self, cnts, reverse: bool = False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(
            *sorted(
                zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse
            )
        )
        return cnts

    def filter_contours(self, cnts, img_height: int):
        filtered_cnts = []

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if (
                self.min_height_to_width_ratio
                <= ratio
                <= self.max_height_to_width_ratio
            ):
                if h / img_height >= self.min_rect_height_to_image_height_ratio:
                    filtered_cnts.append(c)

        return filtered_cnts

    def get_rects(self, cont, img: np.ndarray):
        crop_characters: List[np.ndarray] = []
        img_height = img.shape[0]
        img_width = img.shape[1]

        sorted_contours = self.sort_contours(cont)
        filtered_contours = self.filter_contours(sorted_contours, img_height)

        bias = 3
        for character in filtered_contours:
            (x, y, w, h) = cv2.boundingRect(character)

            y_1 = 0 if y - bias < 0 else y - bias
            y_2 = img_height if y + h + bias > img_height else y + h + bias
            x_1 = 0 if x - bias < 0 else x - bias
            x_2 = img_width if x + w + bias > img_width else x + w + bias

            curr_num = img[y_1:y_2, x_1:x_2]
            crop_characters.append(curr_num)

        heights = [character.shape[0] for character in crop_characters]

        height_mode = mode(heights)
        height_median = median(heights)

        height_factor = 0.95 * height_mode
        crop_characters_filtered = list(
            filter(
                lambda character: character.shape[0] > height_factor,
                crop_characters,
            ),
        )

        print("Number of cropped_characters: ", len(crop_characters))
        print(
            "Number of filtered cropped_characters: ",
            len(crop_characters_filtered),
        )

        return crop_characters_filtered

    def get_cropped_image_from_countours(self):
        pass

    def crop_image_using_contours(self, cont, img: np.ndarray):
        img_height = img.shape[0]
        img_width = img.shape[1]

        x_min_area = 0
        x_max_area = img_width

        sorted_contours = self.sort_contours(cont)
        filtered_contours = self.filter_contours(sorted_contours, img_height)

        bias = 3

        (x, _, _, _) = cv2.boundingRect(filtered_contours[0])
        if x - bias > x_min_area:
            x_min_area = x - bias

        (x, _, _, _) = cv2.boundingRect(
            filtered_contours[len(filtered_contours) - 1]
        )
        if x + bias < x_max_area:
            x_max_area = x + bias

        return img[:, x_min_area, x_max_area]

