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
# Cache for loaded models
# -----------------------------
loaded_models = {}

def load_model(model_name):
    if model_name not in loaded_models:
        print(f"Loading model: {model_name}")
        loaded_models[model_name] = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        loaded_models[model_name].enable_attention_slicing()
        loaded_models[model_name].vae.enable_tiling()
        loaded_models[model_name].vae.enable_slicing()
        try:
            loaded_models[model_name].enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Warning: XFormers not enabled for {model_name}: {e}")
    return loaded_models[model_name]

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

        model_name = data.get("model", "playgroundai/playground-v2.5-1024px-aesthetic")
        prompt = data["prompt"]
        steps = int(data.get("steps", 50))
        guidance = float(data.get("guidance", 4.5))
        width = int(data.get("width", 1920))
        height = int(data.get("height", 1080))

        await ws.send_json({"status": "generating"})

        # Dynamically load the selected model
        pipe = load_model(model_name)

        # Generate image
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
        ).images[0]

        # Encode image for websocket
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()

        await ws.send_json({
            "status": "done",
            "image": encoded,
        })

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
