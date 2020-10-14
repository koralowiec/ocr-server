from typing import List, Union, Tuple
import pytesseract
import numpy as np
from statistics import mode

from image_preprocessing_service import ImagePreprocessingService


class RecognizeService:
    characters: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    bad_chars: List[str] = ["\n\x0c", "\x0c", "\n"]
    psms: List[str] = [
        "7",
        "8",
        "9",
        "11",
    ]

    def get_license_plate_number(
        self, image: Union[bytes, np.ndarray]
    ) -> Tuple[str, dict]:
        if isinstance(image, bytes):
            image = ImagePreprocessingService.from_bytes_to_image(image)

        image_paths: dict = {}

        ImagePreprocessingService.save_image(image, filename_prefix="source")
        image_paths["source"] = ImagePreprocessingService.save_image_minio(
            image, filename_sufix="source"
        )

        (
            characters,
            characters_in_one_image,
            characters_in_one_image_bordered,
            roi,
        ) = ImagePreprocessingService.get_images_with_characters(image)

        ImagePreprocessingService.save_images(
            characters, filename_prefix="characters"
        )
        ImagePreprocessingService.save_image(
            characters_in_one_image, filename_prefix="concatenated"
        )
        image_paths[
            "concatenated"
        ] = ImagePreprocessingService.save_image_minio(
            characters_in_one_image, filename_sufix="concatenated"
        )
        ImagePreprocessingService.save_image(
            characters_in_one_image_bordered,
            filename_prefix="concatenated-bordered",
        )
        ImagePreprocessingService.save_image(roi, filename_prefix="roi")
        image_paths["roi"] = ImagePreprocessingService.save_image_minio(
            roi, filename_sufix="roi"
        )

        potential_length_of_lp_number = len(characters)

        lp_numbers_from_concatenated_image = self.get_numbers(
            characters_in_one_image
        )
        lp_numbers_from_bordered_image = self.get_numbers(
            characters_in_one_image_bordered
        )

        lp_numbers = [
            *lp_numbers_from_concatenated_image,
            *lp_numbers_from_bordered_image,
        ]

        print(lp_numbers)

        lp_numbers = [n for n in lp_numbers if len(n) > 0]
        lp_numbers_with_potentialy_correct_length = [
            n for n in lp_numbers if len(n) == potential_length_of_lp_number
        ]
        lp_numbers_with_different_length = [
            n
            for n in lp_numbers
            if n not in lp_numbers_with_potentialy_correct_length
        ]

        print(lp_numbers_with_potentialy_correct_length)
        print(lp_numbers_with_different_length)

        return (mode(lp_numbers_with_potentialy_correct_length), image_paths)

    def get_numbers(self, image: np.ndarray) -> List[str]:
        numbers = []
        for psm in self.psms:
            numbers.append(self.run_ocr(image, psm))

        return numbers

    def run_ocr(self, image: np.ndarray, psm: int) -> str:
        tesseract_conf = (
            f"-c tessedit_char_whitelist={self.characters} --psm {psm}"
        )
        number = ""
        try:
            number = pytesseract.image_to_string(image, config=tesseract_conf)
        except:
            print(f"OCR could not recognize number for --psm {psm}")

        return self.clean_up_number(number)

    def clean_up_number(self, number: str) -> str:
        for char in self.bad_chars:
            number = number.replace(char, "")

        return number

    def get_numbers_from_separate_characters(
        self, characters: List[np.ndarray]
    ):
        number = ""
        for c in characters:
            recognized_character = self.run_ocr(c, 10)
            if len(recognized_character) != 1:
                recognized_character = "?"

            number += recognized_character

        return number

