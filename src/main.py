from fastapi import FastAPI, File, Depends
from pydantic import BaseModel, Field
import base64
import time

from recognize_service import RecognizeService

app = FastAPI()


class Base64Body(BaseModel):
    b64Encoded: str = Field(..., title="Image encoded in Base64")


def process_image_raw(
    file: bytes = File(...), recognize_service: RecognizeService = Depends(),
):
    start_time = time.time()
    lp_number, links = recognize_service.get_license_plate_number(file)
    end_time = time.time()
    print("Time: ", end_time - start_time)

    return {"licensePlateNumber": lp_number, "links": links}


def process_image_base64(
    b64: Base64Body, recognize_service: RecognizeService = Depends(),
):
    file = base64.b64decode(b64.b64Encoded)

    start_time = time.time()
    lp_number, links = recognize_service.get_license_plate_number(file)
    end_time = time.time()
    print("Time: ", end_time - start_time)

    return {"licensePlateNumber": lp_number, "links": links}


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

