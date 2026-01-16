from heartlib import HeartMuLaGenPipeline
import argparse
import torch
from transformers import BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
import uuid
import json
import asyncio
from queue import Queue
from threading import Thread

app = FastAPI(title="HeartMuLa Music Generation API")

# Global pipeline instance
pipe: Optional[HeartMuLaGenPipeline] = None


class GenerateRequest(BaseModel):
    lyrics: str
    tags: str
    max_audio_length_ms: int = 240_000
    topk: int = 50
    temperature: float = 1.0
    cfg_scale: float = 1.5
    save_path: str = None  # Optional: save directly to this path


class GenerateResponse(BaseModel):
    file_id: str
    message: str


# Store generated files temporarily
OUTPUT_DIR = tempfile.mkdtemp(prefix="heartmula_")


def run_generation(request: GenerateRequest, file_id: str, progress_queue: Queue):
    """Run generation in a thread, pushing progress to queue."""
    if request.save_path:
        save_path = request.save_path
    else:
        save_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp3")
    lyrics_path = os.path.join(OUTPUT_DIR, f"{file_id}_lyrics.txt")
    tags_path = os.path.join(OUTPUT_DIR, f"{file_id}_tags.txt")

    with open(lyrics_path, "w") as f:
        f.write(request.lyrics)
    with open(tags_path, "w") as f:
        f.write(request.tags)

    def progress_callback(current_frame, total_frames, generated_frames, finished):
        progress_queue.put({
            "type": "progress",
            "current_frame": current_frame,
            "total_frames": total_frames,
            "generated_frames": generated_frames,
            "progress_pct": round(current_frame / total_frames * 100, 1),
            "elapsed_ms": generated_frames * 80,
            "finished": finished,
        })

    try:
        with torch.no_grad():
            pipe(
                {
                    "lyrics": lyrics_path,
                    "tags": tags_path,
                },
                max_audio_length_ms=request.max_audio_length_ms,
                save_path=save_path,
                topk=request.topk,
                temperature=request.temperature,
                cfg_scale=request.cfg_scale,
                progress_callback=progress_callback,
            )
        progress_queue.put({
            "type": "complete",
            "file_id": file_id,
            "save_path": save_path,
            "message": "Generation complete",
        })
    except Exception as e:
        progress_queue.put({
            "type": "error",
            "error": str(e),
        })
    finally:
        # Clean up temp input files
        if os.path.exists(lyrics_path):
            os.unlink(lyrics_path)
        if os.path.exists(tags_path):
            os.unlink(tags_path)
        progress_queue.put(None)  # Signal end of stream


@app.post("/generate", response_model=GenerateResponse)
async def generate_music(request: GenerateRequest):
    """Non-streaming endpoint - returns when generation is complete."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    file_id = str(uuid.uuid4())
    save_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp3")

    lyrics_path = os.path.join(OUTPUT_DIR, f"{file_id}_lyrics.txt")
    tags_path = os.path.join(OUTPUT_DIR, f"{file_id}_tags.txt")

    with open(lyrics_path, "w") as f:
        f.write(request.lyrics)
    with open(tags_path, "w") as f:
        f.write(request.tags)

    try:
        with torch.no_grad():
            pipe(
                {
                    "lyrics": lyrics_path,
                    "tags": tags_path,
                },
                max_audio_length_ms=request.max_audio_length_ms,
                save_path=save_path,
                topk=request.topk,
                temperature=request.temperature,
                cfg_scale=request.cfg_scale,
            )
    finally:
        os.unlink(lyrics_path)
        os.unlink(tags_path)

    return GenerateResponse(file_id=file_id, message="Generation complete")


@app.post("/generate/stream")
async def generate_music_stream(request: GenerateRequest):
    """Streaming endpoint - returns Server-Sent Events with progress updates."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    file_id = str(uuid.uuid4())
    progress_queue = Queue()

    # Start generation in background thread
    thread = Thread(target=run_generation, args=(request, file_id, progress_queue))
    thread.start()

    async def event_generator():
        while True:
            # Check queue with small timeout to allow async cancellation
            await asyncio.sleep(0.05)
            while not progress_queue.empty():
                data = progress_queue.get()
                if data is None:
                    return
                yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/download/{file_id}")
async def download_file(file_id: str):
    file_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp3")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/mpeg", filename=f"{file_id}.mp3")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "cuda_available": torch.cuda.is_available(),
        "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--version", type=str, default="3B")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--bnb", type=str, choices=["4bit", "8bit"], default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile-cache-dir", type=str, default=None)
    return parser.parse_args()


def load_model(args):
    global pipe

    bnb_config = None
    if args.bnb == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif args.bnb == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.model_path,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        version=args.version,
        bnb_config=bnb_config,
        compile=args.compile,
        compile_mode=args.compile_mode,
        compile_cache_dir=args.compile_cache_dir,
    )

    print("[Server] Model loaded, ready to serve requests")


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    load_model(args)
    uvicorn.run(app, host=args.host, port=args.port)
