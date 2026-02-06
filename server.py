import base64
import io
import os
from datetime import datetime

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, WebSocket
from PIL import Image

# -----------------------------
# CUDA-safe process setup
# -----------------------------
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# -----------------------------
# Model load (ONCE)
# -----------------------------
pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2.5-1024px-aesthetic",
    torch_dtype=torch.float16,
    variant="fp16",
)

pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass

OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()


@app.websocket("/ws")
async def generate_image(ws: WebSocket):
    await ws.accept()

    while True:
        data = await ws.receive_json()

        prompt = data["prompt"]
        steps = int(data.get("steps", 30))
        guidance = float(data.get("guidance", 4.5))
        width = int(data.get("width", 1536))
        height = int(data.get("height", 864))

        await ws.send_json({"status": "generating"})

        
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
        ).images[0]

        # Save image
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{ts}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        image.save(path)

        # Encode for websocket
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()

        await ws.send_json({
            "status": "done",
            "image": encoded,
            "path": path,
        })
