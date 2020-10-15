from typing import List, Union, Tuple
import pytesseract
import numpy as np
from statistics import mode
from collections import Counter

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

        ImagePreprocessingService.save_image(image, filename_prefix="source")
        image_paths["source"] = ImagePreprocessingService.save_image_minio(
            image, filename_sufix="source"
        )

        images = ImagePreprocessingService.get_images_with_characters(image)
        characters = images["characters"]
        characters_in_one_image = images["concatenated"]
        characters_in_one_image_bordered = images["concatenated_bordered"]
        roi = images["roi"]

        ImagePreprocessingService.save_images_with_key_as_prefix(images)

        image_paths: dict = {}
        image_paths[
            "concatenated"
        ] = ImagePreprocessingService.save_image_minio(
            characters_in_one_image, filename_sufix="concatenated"
        )
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

        print(
            "numbers pc length", lp_numbers_with_potentialy_correct_length,
        )
        print("numbers d length", lp_numbers_with_different_length)

        if len(lp_numbers_with_potentialy_correct_length) == 0:
            potential_length_of_lp_number = len(
                mode(lp_numbers_with_different_length)
            )
            lp_numbers_with_potentialy_correct_length = [
                n
                for n in lp_numbers_with_different_length
                if len(n) == potential_length_of_lp_number
            ]

        most_common_lp = Counter(
            lp_numbers_with_potentialy_correct_length
        ).most_common()
        print("Most common", most_common_lp)

        potential_lp_number = most_common_lp[0][0]

        if len(most_common_lp) != 1:
            for i in range(potential_length_of_lp_number):
                cnt = Counter()
                for lp in most_common_lp:
                    char = lp[0][i]
                    quantity_of_char = lp[1]
                    cnt[char] += quantity_of_char

                ch = cnt.most_common()
                ch_set = set([char[0] for char in ch])

                if len(ch_set) > 1:
                    print("ch", ch)
                    print("ch set", ch_set)
                    character = characters[i]
                    new_ch = self.run_ocr(character, 10)
                    print("ocr psm 10:", new_ch)

                    if new_ch not in ch_set:
                        x = most_common_lp[0][1]
                        most_common_chars = list(
                            filter(lambda c: c[1] >= x, ch)
                        )
                        print(most_common_chars)
                        if len(most_common_chars) == 1:
                            new_ch = most_common_chars[0][0]
                        else:
                            new_ch = "?"

                    print("chosen char:", new_ch)

                    potential_lp_number = self.change_str_at_index(
                        potential_lp_number, i, new_ch
                    )

        return (potential_lp_number, image_paths)

    def change_str_at_index(
        self, string: str, index: int, change_str: str
    ) -> str:
        return string[:index] + change_str + string[index + 1 :]

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
