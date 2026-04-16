import base64
import io
import os
import gc

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

import uvicorn

# -----------------------------
# CUDA-safe process setup
# -----------------------------
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# -----------------------------
# Cache for currently loaded model
# -----------------------------
current_model = None
current_model_name = None

def unload_model():
    global current_model, current_model_name
    if current_model is not None:
        print(f"Unloading model: {current_model_name}")
        del current_model
        current_model = None
        current_model_name = None
        gc.collect()
        torch.cuda.empty_cache()

def load_model(model_name):
    global current_model, current_model_name
    
    if current_model_name == model_name:
        return current_model
    
    unload_model()
    
    print(f"Loading model: {model_name}")
    current_model = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to("cuda")
    current_model.enable_attention_slicing()
    current_model.vae.enable_tiling()
    current_model.vae.enable_slicing()
    try:
        current_model.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"Warning: XFormers not enabled for {model_name}: {e}")
    
    current_model_name = model_name
    return current_model

OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

@app.get("/")
async def root():
    """Serve the frontend index.html for browser access."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": "Index not found. Use websocket endpoints at /ws and /ws/load-model"}

@app.websocket("/ws/load-model")
async def load_model_ws(ws: WebSocket):
    await ws.accept()
    
    while True:
        data = await ws.receive_json()
        model_name = data.get("model")
        
        try:
            print(f"Loading model: {model_name}")
            load_model(model_name)
            await ws.send_json({"status": "loaded", "model": model_name})
        except Exception as e:
            print(f"Error loading model: {e}")
            await ws.send_json({"status": "error", "message": str(e)})

@app.websocket("/ws")
async def generate_image(ws: WebSocket):
    await ws.accept()

    while True:
        data = await ws.receive_json()

        prompt = data["prompt"]
        steps = int(data.get("steps", 50))
        guidance = float(data.get("guidance", 4.5))
        width = int(data.get("width", 1920))
        height = int(data.get("height", 1080))

        await ws.send_json({"status": "generating"})

        try:
            # Use the currently loaded model
            if current_model is None:
                await ws.send_json({"status": "error", "message": "No model loaded"})
                continue

            # Generate image
            image = current_model(
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
        except Exception as e:
            print(f"Error generating image: {e}")
            await ws.send_json({"status": "error", "message": str(e)})

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)
