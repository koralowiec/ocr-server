import cv2
import numpy as np
from typing import List, Optional, Tuple
from statistics import mode, median


class ImagePreprocessingService:
    min_height_to_width_ratio: int = 1
    max_height_to_width_ratio: int = 10
    min_rect_height_to_image_height_ratio: float = 0.2
    white_color = [255, 255, 255]

    @classmethod
    def get_separate_characters_from_image(cls, img_bytes: bytes):
        image_org = cls.from_bytes_to_image(img_bytes)
        image = cls.prepare_for_segmentation(image_org)
        cont = cls.get_contours(image)
        characters = cls.get_rects(cont, image_org)

        cls.save_images(characters)

        concatenated_characters = cls.concatenate_characters(characters)
        cls.save_image(concatenated_characters, filename_prefix="conc")

        border_size = 3
        concatenated_characters_bordered = cv2.copyMakeBorder(
            concatenated_characters,
            border_size,
            border_size,
            border_size,
            border_size,
            cv2.BORDER_CONSTANT,
            value=cls.white_color,
        )
        cls.save_image(image_org, filename_prefix="org")
        cls.save_image(
            concatenated_characters_bordered, filename_prefix="conc-border"
        )

        contours = cls.prepare_for_roi_from_biggest_countour(image_org)
        roi = cls.get_roi_from_the_biggest_countour(image_org, contours)
        cls.save_image(roi, filename_prefix="roi")

        return (
            characters,
            concatenated_characters,
            concatenated_characters_bordered,
            roi,
        )

    @classmethod
    def from_bytes_to_image(cls, img_bytes: bytes):
        np_img = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

    @classmethod
    def prepare_for_segmentation(cls, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.threshold(
            blur, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        return binary

    @classmethod
    def get_contours(cls, img: np.ndarray):
        cont, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cont

    @classmethod
    def sort_contours(cls, cnts, reverse: bool = False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(
            *sorted(
                zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse
            )
        )
        return cnts

    @classmethod
    def filter_contours_by_ratios(cls, cnts, img_height: int):
        filtered_cnts = []

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if (
                cls.min_height_to_width_ratio
                <= ratio
                <= cls.max_height_to_width_ratio
            ):
                if h / img_height >= cls.min_rect_height_to_image_height_ratio:
                    filtered_cnts.append(c)

        return filtered_cnts

    @staticmethod
    def contours_to_characters(contours, image):
        characters = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            character = image[y : (y + h), x : (x + w)]
            characters.append(character)

        return characters

    @staticmethod
    def save_image(image: np.ndarray, filename_prefix: str = ""):
        cv2.imwrite(f"/img/{filename_prefix}.jpeg", image)

    @staticmethod
    def save_images(images: List[np.ndarray], filename_prefix: str = ""):
        print("Number of images: ", len(images))

        for i in range(len(images)):
            cv2.imwrite(f"/numbers/{filename_prefix}{i}.jpeg", images[i])

    @staticmethod
    def is_rectangle1_in_rectangle2(
        rectangle1: Tuple[int], rectangle2: Tuple[int]
    ) -> bool:
        (x1, y1, w1, h1) = rectangle1
        (x2, y2, w2, h2) = rectangle2

        # centre of 1st rectangle
        x_central = x1 + w1 / 2
        y_central = y1 + h1 / 2

        return (
            x2 < x_central
            and x_central < x2 + w2
            and y2 < y_central
            and y_central < y2 + h2
        )

    @classmethod
    def get_rects(
        cls, cont, img: np.ndarray, boundary_rect: Optional[Tuple[int]] = None
    ):
        cropped_characters: List[np.ndarray] = []
        img_height = img.shape[0]
        img_width = img.shape[1]

        sorted_contours = cls.sort_contours(cont)
        sorted_characters = cls.contours_to_characters(sorted_contours, img)
        cls.save_images(sorted_characters, filename_prefix="sort")
        filtered_contours = cls.filter_contours_by_ratios(
            sorted_contours, img_height
        )
        filtered_characters = cls.contours_to_characters(filtered_contours, img)
        cls.save_images(filtered_characters, filename_prefix="fil")

        # adaptive bias?
        bias = 3
        for i in range(len(filtered_contours)):
            character = filtered_contours[i]
            (x, y, w, h) = cv2.boundingRect(character)

            if boundary_rect is not None:
                if cls.is_rectangle1_in_rectangle2((x, y, w, h), boundary_rect):
                    continue

            if i > 0:
                prev_character = filtered_contours[i - 1]
                prev_rect = cv2.boundingRect(prev_character)

                if cls.is_rectangle1_in_rectangle2((x, y, w, h), prev_rect):
                    continue

            y_1 = 0 if y - bias < 0 else y - bias
            y_2 = img_height if y + h + bias > img_height else y + h + bias
            x_1 = 0 if x - bias < 0 else x - bias
            x_2 = img_width if x + w + bias > img_width else x + w + bias

            curr_num = img[y_1:y_2, x_1:x_2]
            cropped_characters.append(curr_num)

        cls.save_images(cropped_characters, filename_prefix="cropped")

        heights = [character.shape[0] for character in cropped_characters]

        height_mode = mode(heights)
        height_median = median(heights)

        height_factor_min = 0.9 * height_mode
        height_factor_max = 1.1 * height_mode
        print(heights)
        print(height_factor_min)
        crop_characters_filtered = list(
            filter(
                lambda character: character.shape[0] > height_factor_min
                and character.shape[0] < height_factor_max,
                cropped_characters,
            ),
        )

        print("Number of cropped_characters: ", len(cropped_characters))
        print(
            "Number of filtered cropped_characters: ",
            len(crop_characters_filtered),
        )

        return crop_characters_filtered

    @staticmethod
    def concatenate_characters(characters: List[np.ndarray]) -> np.ndarray:
        heights = [character.shape[0] for character in characters]
        height_mode = mode(heights)

        resized_characters = []
        for c in characters:
            if height_mode == c.shape[1]:
                continue

            dim = (c.shape[1], height_mode)
            resized = cv2.resize(c, dim, interpolation=cv2.INTER_AREA)
            resized_characters.append(resized)

        return cv2.hconcat(resized_characters)

    @staticmethod
    def get_the_biggest_contour(contours):
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        return cnts[0]

    @classmethod
    def get_roi_from_the_biggest_countour(cls, image, contours):
        countour = cls.get_the_biggest_contour(contours)
        (x, y, w, h) = cv2.boundingRect(countour)
        result = image[y : y + h, x : x + w]
        return result

    @classmethod
    def prepare_for_roi_from_biggest_countour(cls, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.float32) / 15
        filtered = cv2.filter2D(gray, -1, kernel)
        ret, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        return contours
