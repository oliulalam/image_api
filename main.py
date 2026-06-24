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

# CORS — Flutter থেকে call করার জন্য
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# rembg session একবার load করো (fast হবে)
rembg_session = new_session("u2net")

# সব supported image format
ALLOWED_TYPES = [
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/heic", "image/heif",
    "image/bmp", "image/tiff", "image/gif"
]


# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────

def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """PIL Image কে base64 string এ convert করো"""
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def advanced_upscale(img: Image.Image, scale: int = 2) -> Image.Image:
    """
    Advanced multi-step upscaling:
    Step 1: LANCZOS resize (best quality resize algorithm)
    Step 2: Unsharp Mask (detail enhancement)
    Step 3: Edge sharpening via OpenCV
    Step 4: Contrast + Color enhancement
    Step 5: Noise reduction
    """
    original_mode = img.mode

    # Step 1: LANCZOS upscale (PIL এর সবচেয়ে ভালো algorithm)
    w, h = img.size
    upscaled = img.resize((w * scale, h * scale), Image.LANCZOS)

    # Step 2: Unsharp Mask — detail sharp করে
    # radius=2, percent=120, threshold=3 — best balance
    sharpened = upscaled.filter(
        ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3)
    )

    # Step 3: OpenCV দিয়ে edge sharpening
    img_array = np.array(sharpened.convert("RGB"))

    # Sharpening kernel
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    sharpened_cv = cv2.filter2D(img_array, -1, kernel)

    # Blend original upscaled + sharpened (70% sharp + 30% smooth)
    blended = cv2.addWeighted(img_array, 0.3, sharpened_cv, 0.7, 0)

    # Step 4: Noise reduction — detail রেখে noise কমায়
    denoised = cv2.fastNlMeansDenoisingColored(blended, None, 3, 3, 7, 21)

    # Step 5: PIL এ ফিরে যাও
    result = Image.fromarray(denoised)

    # Step 6: Contrast সামান্য বাড়াও
    contrast = ImageEnhance.Contrast(result)
    result = contrast.enhance(1.08)

    # Step 7: Color একটু vivid করো
    color = ImageEnhance.Color(result)
    result = color.enhance(1.05)

    # Step 8: Sharpness একটু বাড়াও
    sharpness = ImageEnhance.Sharpness(result)
    result = sharpness.enhance(1.2)

    # Original mode restore করো (RGBA হলে)
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
    """
    Image এর background remove করো।
    - Input: যেকোনো image format
    - Output: PNG with transparent background (base64)
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    try:
        file_bytes = await file.read()

        # rembg দিয়ে background remove
        output_bytes = remove(file_bytes, session=rembg_session)

        result_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        result_b64 = image_to_base64(result_img, "PNG")

        return JSONResponse({
            "success": True,
            "message": "Background সফলভাবে remove হয়েছে!",
            "image_base64": result_b64,
            "format": "PNG",
            "size": {
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
    Image upscale করো — Advanced multi-step pipeline।
    - scale: 2 বা 4
    - Output: High quality upscaled image (base64)
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale শুধু 2 বা 4 হতে পারে")

    try:
        file_bytes = await file.read()
        original_img = Image.open(io.BytesIO(file_bytes))
        original_w, original_h = original_img.size

        # Advanced upscale
        result_img = advanced_upscale(original_img, scale)
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/remove-bg-and-upscale")
async def remove_bg_and_upscale(
    file: UploadFile = File(...),
    scale: int = 2
):
    """
    Background remove + Advanced upscale — একসাথে!
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale শুধু 2 বা 4 হতে পারে")

    try:
        file_bytes = await file.read()

        # Step 1: Background remove
        bg_removed_bytes = remove(file_bytes, session=rembg_session)
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)