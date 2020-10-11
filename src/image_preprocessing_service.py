import cv2
import numpy as np
from typing import List
from statistics import mode, median


class ImagePreprocessingService:
    min_height_to_width_ratio: int = 1
    max_height_to_width_ratio: int = 10
    min_rect_height_to_image_height_ratio: float = 0.2

    @classmethod
    def get_separate_characters_from_image(cls, img_bytes: bytes):
        image_org = cls.from_bytes_to_image(img_bytes)
        image = cls.prepare_for_segmentation(image_org)
        print("Type:", type(image))
        cont = cls.get_contours(image)
        characters = cls.get_rects(cont, image_org)

        cls.save_images(characters)

        concatenated_characters = cls.concatenate_characters(characters)
        cls.save_image(concatenated_characters, filename_prefix="conc")

        white = [255, 255, 255]
        concatenated_characters_bordered = cv2.copyMakeBorder(
            concatenated_characters,
            3,
            3,
            3,
            3,
            cv2.BORDER_CONSTANT,
            value=white,
        )
        cls.save_image(
            concatenated_characters_bordered, filename_prefix="conc-border"
        )

        return (
            characters,
            concatenated_characters,
            concatenated_characters_bordered,
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

    @classmethod
    def get_rects(cls, cont, img: np.ndarray):
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

            print(i, (x, y, w, h))

            if i > 0:
                prev_character = filtered_contours[i - 1]
                (x_p, y_p, w_p, h_p) = cv2.boundingRect(prev_character)

                x_central = x + w / 2
                y_central = y + h / 2

                print(i, x_central, y_central)

                if (
                    x_p < x_central
                    and x_central < x_p + w_p
                    and y_p < y_central
                    and y_central < y_p + h_p
                ):
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

        height_factor = 0.9 * height_mode
        print(heights)
        print(height_factor)
        crop_characters_filtered = list(
            filter(
                lambda character: character.shape[0] > height_factor,
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

    @classmethod
    def get_cropped_image_from_countours(cls):
        pass

    @classmethod
    def crop_image_using_contours(cls, cont, img: np.ndarray):
        img_height = img.shape[0]
        img_width = img.shape[1]

        x_min_area = 0
        x_max_area = img_width

        sorted_contours = cls.sort_contours(cont)
        filtered_contours = cls.filter_contours_by_ratios(
            sorted_contours, img_height
        )

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

