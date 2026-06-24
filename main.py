from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from rembg import remove, new_session
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np
import io
import base64

app = FastAPI(
    title="Image Processing API",
    description="Background Remove & Image Upscale API - Built with Python",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rembg_session = new_session("u2net")

ALLOWED_TYPES = [
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/heic", "image/heif",
    "image/bmp", "image/tiff", "image/gif"
]


# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────

def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_bytes(img: Image.Image) -> bytes:
    """PIL Image কে PNG bytes এ convert করো"""
    buf = io.BytesIO()
    # RGBA হলে সরাসরি PNG, নইলে RGB করে PNG
    if img.mode in ("RGBA", "LA"):
        img.save(buf, format="PNG")
    else:
        img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def resize_if_large(img: Image.Image, max_size: int = 1024) -> Image.Image:
    """Image বড় হলে resize করো"""
    w, h = img.size
    if w <= max_size and h <= max_size:
        return img
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def advanced_upscale(img: Image.Image, scale: int = 2) -> Image.Image:
    """Advanced multi-step upscaling pipeline"""
    original_mode = img.mode

    w, h = img.size
    upscaled = img.resize((w * scale, h * scale), Image.LANCZOS)

    sharpened = upscaled.filter(
        ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3)
    )

    img_array = np.array(sharpened.convert("RGB"))
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    sharpened_cv = cv2.filter2D(img_array, -1, kernel)
    blended = cv2.addWeighted(img_array, 0.3, sharpened_cv, 0.7, 0)
    denoised = cv2.fastNlMeansDenoisingColored(blended, None, 3, 3, 7, 21)

    result = Image.fromarray(denoised)
    result = ImageEnhance.Contrast(result).enhance(1.08)
    result = ImageEnhance.Color(result).enhance(1.05)
    result = ImageEnhance.Sharpness(result).enhance(1.2)

    if original_mode == "RGBA":
        result = result.convert("RGBA")

    return result


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Image Processing API চালু আছে! ✅",
        "version": "2.0.0",
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
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    try:
        file_bytes = await file.read()

        # Image open করো
        original_img = Image.open(io.BytesIO(file_bytes))

        # বড় হলে resize করো
        resized = resize_if_large(original_img, max_size=1024)

        # bytes এ convert করো
        resized_bytes = pil_to_bytes(resized)

        # Background remove
        output_bytes = remove(resized_bytes, session=rembg_session)

        result_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        result_b64 = image_to_base64(result_img, "PNG")

        return JSONResponse({
            "success": True,
            "message": "Background সফলভাবে remove হয়েছে!",
            "image_base64": result_b64,
            "format": "PNG",
            "size": {"width": result_img.width, "height": result_img.height}
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...), scale: int = 2):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale শুধু 2 বা 4 হতে পারে")
    try:
        file_bytes = await file.read()
        original_img = Image.open(io.BytesIO(file_bytes))

        resized = resize_if_large(original_img, max_size=800)
        original_w, original_h = resized.size

        result_img = advanced_upscale(resized, scale)
        result_b64 = image_to_base64(result_img, "PNG")

        return JSONResponse({
            "success": True,
            "message": f"Image {scale}x upscale সফল!",
            "image_base64": result_b64,
            "format": "PNG",
            "original_size": {"width": original_w, "height": original_h},
            "new_size": {"width": result_img.width, "height": result_img.height},
            "scale": scale
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/remove-bg-and-upscale")
async def remove_bg_and_upscale(file: UploadFile = File(...), scale: int = 2):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale শুধু 2 বা 4 হতে পারে")
    try:
        file_bytes = await file.read()

        original_img = Image.open(io.BytesIO(file_bytes))
        resized = resize_if_large(original_img, max_size=800)
        resized_bytes = pil_to_bytes(resized)

        # Step 1: Background remove
        bg_removed_bytes = remove(resized_bytes, session=rembg_session)
        bg_removed_img = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")
        original_w, original_h = bg_removed_img.size

        # Step 2: Advanced upscale
        final_img = advanced_upscale(bg_removed_img, scale)
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)