from fastapi import FastAPI, File, Depends, HTTPException
from pydantic import BaseModel, Field
import base64
import time

from recognize_service import RecognizeService

app = FastAPI()


class Base64Body(BaseModel):
    b64Encoded: str = Field(..., title="Image encoded in Base64")


def get_numbers(file: bytes, recognize_service: RecognizeService):
    start_time = time.time()
    try:
        lp_number, links = recognize_service.get_license_plate_number(file)
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=422, detail="Could not get license plate number"
        )
    end_time = time.time()
    print("Time: ", end_time - start_time)

    return {"licensePlateNumber": lp_number, "links": links}


def process_image_raw(
    file: bytes = File(...),
    recognize_service: RecognizeService = Depends(),
):
    return get_numbers(file, recognize_service)


def process_image_base64(
    b64: Base64Body,
    recognize_service: RecognizeService = Depends(),
):
    file = base64.b64decode(b64.b64Encoded)
    return get_numbers(file, recognize_service)


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
