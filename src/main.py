from fastapi import FastAPI, File, Depends
from typing import Optional
from pydantic import BaseModel, Field
import base64

from recognize_service import RecognizeService

app = FastAPI()


class Base64Body(BaseModel):
    b64Encoded: str = Field(..., title="Image encoded in Base64")


@app.post("/ocr/base64")
async def recognize_characters_from_base64(
    b64: Base64Body, recognize_service: RecognizeService = Depends()
):
    img_bytes = base64.b64decode(b64.b64Encoded)

    return recognize_service.get_numbers(img_bytes)


@app.post("/ocr/raw")
async def recognize_characters_from_raw_image(
    file: bytes = File(...), recognize_service: RecognizeService = Depends()
):
    return recognize_service.get_numbers(file)


@app.get("/")
async def healthcheck():
    return "ok"

