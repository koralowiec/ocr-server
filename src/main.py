from fastapi import FastAPI, File, Depends
from typing import Optional
from pydantic import BaseModel, Field
import base64

from recognize_service import RecognizeService
from image_preprocessing_service import ImagePreprocessingService

app = FastAPI()


class Base64Body(BaseModel):
    b64Encoded: str = Field(..., title="Image encoded in Base64")


def process_image_raw(
    file: bytes = File(...),
    recognize_service: RecognizeService = Depends(),
    img_prep_service: ImagePreprocessingService = Depends(),
):
    characters = img_prep_service.get_separate_characters_from_image(file)
    number_from_separate_chars = recognize_service.get_numbers_from_separate_characters(
        characters
    )

    numbers_from_ocr = recognize_service.get_numbers(file)

    return {
        "numberFromSeparateChars": number_from_separate_chars,
        "numbersFromOCR": numbers_from_ocr,
    }


def process_image_base64(
    b64: Base64Body,
    recognize_service: RecognizeService = Depends(),
    img_prep_service: ImagePreprocessingService = Depends(),
):
    file = base64.b64decode(b64.b64Encoded)
    characters = img_prep_service.get_separate_characters_from_image(file)
    number_from_separate_chars = recognize_service.get_numbers_from_separate_characters(
        characters
    )

    numbers_from_ocr = recognize_service.get_numbers(file)

    return {
        "numberFromSeparateChars": number_from_separate_chars,
        "numbersFromOCR": numbers_from_ocr,
    }


@app.post("/ocr/base64")
async def recognize_characters_from_base64(
    process_image: dict = Depends(process_image_base64),
):
    return process_image


@app.post("/ocr/raw")
async def recognize_characters_from_raw_image(
    process_image: dict = Depends(process_image_raw),
):
    return process_image


@app.get("/")
async def healthcheck():
    return "ok"

