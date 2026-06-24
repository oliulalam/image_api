from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from rembg import remove, new_session
from PIL import Image
import cv2
import numpy as np
import io
import base64

app = FastAPI(
    title="Image Processing API",
    description="Background Remove & Image Upscale API - Built with Python",
    version="1.0.0"
)

# CORS — Flutter থেকে call করার জন্য
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# rembg session একবার load করো (fast হবে)
rembg_session = new_session("u2net")


# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────

def read_image(file_bytes: bytes) -> Image.Image:
    """Bytes থেকে PIL Image বানাও"""
    return Image.open(io.BytesIO(file_bytes)).convert("RGBA")


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """PIL Image কে base64 string এ convert করো"""
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def upscale_image_cv2(img: Image.Image, scale: int = 2) -> Image.Image:
    """OpenCV দিয়ে image upscale করো"""
    # PIL → numpy array
    img_array = np.array(img.convert("RGB"))

    # Upscale using INTER_LANCZOS4 (best quality)
    h, w = img_array.shape[:2]
    new_h, new_w = h * scale, w * scale
    upscaled = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # numpy array → PIL Image
    return Image.fromarray(upscaled)


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Image Processing API চালু আছে! ✅",
        "endpoints": {
            "background_remove": "/remove-bg",
            "upscale": "/upscale",
            "both": "/remove-bg-and-upscale",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    """
    Image এর background remove করো।
    - Input: যেকোনো image (jpg, png, webp)
    - Output: PNG with transparent background (base64)
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        raise HTTPException(status_code=400, detail="শুধু JPG, PNG, WEBP file দাও")

    try:
        file_bytes = await file.read()

        # rembg দিয়ে background remove
        output_bytes = remove(file_bytes, session=rembg_session)

        # PIL Image বানাও
        result_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

        # Base64 encode করো
        result_b64 = image_to_base64(result_img, "PNG")

        return JSONResponse({
            "success": True,
            "message": "Background সফলভাবে remove হয়েছে!",
            "image_base64": result_b64,
            "format": "PNG",
            "original_size": {
                "width": result_img.width,
                "height": result_img.height
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    scale: int = 2
):
    """
    Image upscale করো।
    - Input: যেকোনো image
    - scale: 2 বা 4 (default: 2)
    - Output: Upscaled image (base64)
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        raise HTTPException(status_code=400, detail="শুধু JPG, PNG, WEBP file দাও")

    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale শুধু 2 বা 4 হতে পারে")

    try:
        file_bytes = await file.read()
        original_img = Image.open(io.BytesIO(file_bytes))

        original_w, original_h = original_img.size

        # Upscale করো
        upscaled_img = upscale_image_cv2(original_img, scale)

        # Base64 encode
        result_b64 = image_to_base64(upscaled_img, "PNG")

        return JSONResponse({
            "success": True,
            "message": f"Image {scale}x upscale সফল!",
            "image_base64": result_b64,
            "format": "PNG",
            "original_size": {"width": original_w, "height": original_h},
            "new_size": {"width": upscaled_img.width, "height": upscaled_img.height},
            "scale": scale
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/remove-bg-and-upscale")
async def remove_bg_and_upscale(
    file: UploadFile = File(...),
    scale: int = 2
):
    """
    Background remove করো তারপর upscale করো — একসাথে!
    - Input: যেকোনো image
    - Output: Background removed + upscaled image (base64)
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        raise HTTPException(status_code=400, detail="শুধু JPG, PNG, WEBP file দাও")

    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale শুধু 2 বা 4 হতে পারে")

    try:
        file_bytes = await file.read()

        # Step 1: Background remove
        bg_removed_bytes = remove(file_bytes, session=rembg_session)
        bg_removed_img = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")

        original_w, original_h = bg_removed_img.size

        # Step 2: Upscale
        final_img = upscale_image_cv2(bg_removed_img, scale)

        # Base64 encode
        result_b64 = image_to_base64(final_img, "PNG")

        return JSONResponse({
            "success": True,
            "message": f"Background remove + {scale}x upscale সফল!",
            "image_base64": result_b64,
            "format": "PNG",
            "original_size": {"width": original_w, "height": original_h},
            "new_size": {"width": final_img.width, "height": final_img.height},
            "scale": scale
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
