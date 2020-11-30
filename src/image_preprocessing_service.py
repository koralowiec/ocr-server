import cv2
import numpy as np
from typing import List, Tuple, Union
from statistics import mode, median
import io
import uuid
from minio.error import ResponseError

from minio_setup import minio_client, bucket_name


class ImagePreprocessingService:
    min_height_to_width_ratio: int = 1
    max_height_to_width_ratio: int = 10
    min_rect_height_to_image_height_ratio: float = 0.2
    white_color = (255, 255, 255)
    green_color = (0, 255, 0)

    @classmethod
    def get_images_with_characters(cls, image: np.ndarray):
        image = cls.resize_image(image)

        contours, thres = cls.prepare_for_roi_from_biggest_countour(image)
        img_contours_roi = image.copy()
        cv2.drawContours(img_contours_roi, contours, -1, cls.green_color, 3)
        roi, boundary_rectangle = cls.get_roi_from_the_biggest_countour(image, contours)

        roi = cls.resize_image(roi)

        image_for_segmentation = cls.prepare_for_segmentation(roi)
        cont = cls.get_contours(image_for_segmentation)

        img_contours_characters = np.zeros_like(roi)
        for i in range(3):
            img_contours_characters[:, :, i] = image_for_segmentation
        cv2.drawContours(img_contours_characters, cont, -1, cls.green_color, 3)

        characters, rects = cls.get_rects(cont, roi)
        rectangle_characters_image = cls.draw_rectangles_on_image(roi, rects)

        concatenated_characters = cls.concatenate_characters(characters)

        # concatenated_characters_height = concatenated_characters.shape[0]
        # concatenated_characters_width = concatenated_characters.shape[1]
        # border_size = int(concatenated_characters_height * 0.3)
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

        images = {}
        images["characters"] = characters
        images["concatenated"] = concatenated_characters
        images["concatenated_bordered"] = concatenated_characters_bordered
        images["roi"] = roi
        images["binary_seg"] = image_for_segmentation
        images["thresh_roi"] = thres
        images["countours_roi"] = img_contours_roi
        images["image_for_segmentation"] = image_for_segmentation
        images["countours_characters"] = img_contours_characters
        images["rectangles_characters"] = rectangle_characters_image

        return images

    @staticmethod
    def draw_rectangles_on_image(
        image: np.ndarray, rectangles: List[Tuple[int, int, int, int]]
    ):
        image_with_rectangles = image.copy()
        for rect in rectangles:
            cv2.rectangle(
                image_with_rectangles,
                (rect[0], rect[1]),
                (rect[0] + rect[2], rect[1] + rect[3]),
                ImagePreprocessingService.green_color,
                2,
            )
        return image_with_rectangles

    @classmethod
    def from_bytes_to_image(cls, img_bytes: bytes):
        np_img = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

    @classmethod
    def prepare_for_segmentation(cls, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
            1
        ]

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
            *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
        )
        return cnts

    @classmethod
    def filter_contours_by_ratios(cls, cnts, img_height: int):
        filtered_cnts = []

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if cls.min_height_to_width_ratio <= ratio <= cls.max_height_to_width_ratio:
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
    def save_images_to_minio(images: dict):
        paths = {}
        for key, image in images.items():
            if isinstance(image, list):
                paths_list = {}
                for i in range(len(image)):
                    path = ImagePreprocessingService.save_image_minio(
                        image[i], filename_sufix=f"{key}{i}"
                    )
                    paths_list[f"{key}{i}"] = path
                paths[key] = paths_list
            else:
                saved_image_path = ImagePreprocessingService.save_image_minio(
                    image, key
                )
                paths[key] = saved_image_path
        return paths

    @staticmethod
    def save_image_minio(
        image: Union[bytes, np.ndarray], filename_sufix: str = ""
    ) -> str:
        if image is None:
            return ""

        if isinstance(image, np.ndarray):
            success, encoded_image = cv2.imencode(".jpeg", image)
            image = encoded_image.tobytes()

        uid = uuid.uuid1()
        if filename_sufix != "":
            filename_sufix = f"-{filename_sufix}"
        filename = f"{uid}{filename_sufix}.jpeg"
        path = f"{bucket_name}/{filename}"

        try:
            with io.BytesIO(image) as data:
                _ = minio_client.put_object(bucket_name, filename, data, len(image))
        except ResponseError as e:
            print(e)

        return path

    @staticmethod
    def save_image(image: np.ndarray, filename_prefix: str = ""):
        try:
            cv2.imwrite(f"/img/{filename_prefix}.jpeg", image)
        except Exception as e:
            print(e)

    @staticmethod
    def save_images(
        images: List[np.ndarray], filename_prefix: str = "", dir="/numbers"
    ):
        print("Number of images: ", len(images))

        for i in range(len(images)):
            cv2.imwrite(f"{dir}/{filename_prefix}{i}.jpeg", images[i])

    @staticmethod
    def save_images_with_key_as_prefix(images: dict):
        for key, image in images.items():
            if isinstance(image, list):
                ImagePreprocessingService.save_images(
                    image, filename_prefix=key, dir="/img/numbers"
                )
            else:
                ImagePreprocessingService.save_image(image, filename_prefix=key)

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
    def get_rects(cls, cont, img: np.ndarray):
        cropped_characters: List[np.ndarray] = []
        img_height = img.shape[0]
        img_width = img.shape[1]

        sorted_contours = cls.sort_contours(cont)
        sorted_characters = cls.contours_to_characters(sorted_contours, img)
        cls.save_images(sorted_characters, filename_prefix="sort")

        # filtered_contours = sorted_contours
        filtered_contours = cls.filter_contours_by_ratios(sorted_contours, img_height)
        filtered_characters = cls.contours_to_characters(filtered_contours, img)
        cls.save_images(filtered_characters, filename_prefix="fil")
        print("Number of filtered_contours", len(filtered_contours))

        characters = []

        for i in range(len(filtered_contours)):
            character = filtered_contours[i]
            (x, y, w, h) = cv2.boundingRect(character)

            characters_length = len(characters)
            if characters_length > 0:
                index = characters[characters_length - 1][1]
                prev_character = filtered_contours[index]
                prev_rect = cv2.boundingRect(prev_character)

                if cls.is_rectangle1_in_rectangle2((x, y, w, h), prev_rect):
                    continue

            curr_num = (x, y, w, h)
            entry = (curr_num, i)
            characters.append(entry)

        characters = [c[0] for c in characters]

        heights = [c[3] for c in characters]
        height_mode = mode(heights)
        height_median = median(heights)
        widths = [c[2] for c in characters]
        width_mode = mode(widths)

        print("Height mode:", height_mode)
        print("Height median:", height_median)

        for i in range(len(characters)):
            print("h", characters[i][3])

        height_factor_min = 0.9 * height_median
        height_factor_max = 2 * height_median
        cropped_characters_filtered = list(
            filter(
                lambda c: c[3] > height_factor_min and c[3] < height_factor_max,
                characters,
            ),
        )

        print(
            "Number of filtered cropped_characters: ",
            len(cropped_characters_filtered),
        )

        bias_h = int(0.1 * height_mode)
        bias_w = int(0.15 * width_mode)
        for cords in cropped_characters_filtered:
            (x, y, w, h) = cords

            y_1 = 0 if y - bias_h < 0 else y - bias_h
            y_2 = img_height if y + h + bias_h > img_height else y + h + bias_h
            x_1 = 0 if x - bias_w < 0 else x - bias_w
            x_2 = img_width if x + w + bias_w > img_width else x + w + bias_w

            curr_num = img[y_1:y_2, x_1:x_2]
            cropped_characters.append(curr_num)

        print("Number of cropped_characters: ", len(cropped_characters))

        return cropped_characters, cropped_characters_filtered

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
    def get_roi_from_the_biggest_countour(
        cls, image, contours
    ) -> Tuple[np.ndarray, Tuple[int]]:
        countour = cls.get_the_biggest_contour(contours)
        rect = cv2.boundingRect(countour)
        (x, y, w, h) = rect
        result = image[y : y + h, x : x + w]
        return result, rect

    @classmethod
    def prepare_for_roi_from_biggest_countour(cls, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.float32) / 15
        filtered = cv2.filter2D(gray, -1, kernel)
        _, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours, thresh

    @staticmethod
    def resize_image(image: np.ndarray, scale: float = 4.0):
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized
