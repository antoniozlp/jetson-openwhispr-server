"""OpenAI-compatible speech-to-text server for OpenWhispr's Self-Hosted mode.

OpenWhispr POSTs multipart/form-data to {Server URL}/audio/transcriptions
(fields: file, model, optional language and prompt) and reads back
{"text": "..."}. Its settings panel checks the server with GET {Server URL}/models
(also trying /v1/models) and wants the typed model name to be in that list.

One Whisper model is loaded on the GPU at startup with faster-whisper
(CTranslate2). The `model` field of a request is not used to pick a model.
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import ctranslate2
import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from faster_whisper import WhisperModel

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
# The name OpenWhispr's Model field must match (it is compared with /models).
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "Whisper-Large-v3-Turbo")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "")
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "5"))
VAD_FILTER = os.environ.get("VAD_FILTER", "true").lower() in ("1", "true", "yes")
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("openwhispr-server")

state = {}
# One transcription at a time: the Orin Nano's 8GB is shared between CPU and GPU.
gpu_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    # float16 on the GPU; on the CPU, int8 is the fast option.
    compute_type = COMPUTE_TYPE or ("float16" if device == "cuda" else "int8")
    if device == "cpu":
        log.warning("No CUDA device found, running on the CPU. Is the container using --runtime nvidia?")

    start = time.perf_counter()
    model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type, download_root=MODELS_DIR)
    log.info("Loaded '%s' on %s (%s) in %.1fs", WHISPER_MODEL, device, compute_type, time.perf_counter() - start)

    # The first transcription on the GPU is slow; do it now instead of on the user's first dictation.
    segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1, language="en")
    list(segments)
    log.info("Ready, serving as model '%s'", SERVED_MODEL_NAME)

    state.update(model=model, device=device, compute_type=compute_type)
    yield
    state.clear()


app = FastAPI(title="jetson-openwhispr-server", lifespan=lifespan)
# OpenWhispr appends paths to the Server URL as typed, so serve with and without /v1.
api = APIRouter()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": WHISPER_MODEL,
        "served_as": SERVED_MODEL_NAME,
        "device": state.get("device"),
        "compute_type": state.get("compute_type"),
    }


@api.get("/models")
def models():
    return {
        "object": "list",
        "data": [{"id": SERVED_MODEL_NAME, "object": "model", "created": 0, "owned_by": "local"}],
    }


@api.post("/audio/transcriptions")
def transcribe(
    file: UploadFile = File(...),
    model: str = Form(""),
    language: str = Form(""),
    prompt: str = Form(""),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    language = language.strip().lower()
    if language in ("", "auto"):
        language = None

    start = time.perf_counter()
    try:
        with gpu_lock:
            # PyAV decodes whatever OpenWhispr sends (webm/opus by default; also ogg, mp4, mp3, wav).
            segments, info = state["model"].transcribe(
                file.file,
                beam_size=BEAM_SIZE,
                language=language,
                initial_prompt=prompt.strip() or None,
                temperature=temperature,
                vad_filter=VAD_FILTER,
            )
            # transcribe() is lazy: the work happens while iterating.
            segments = list(segments)
    except ValueError as e:  # e.g. an unknown language code
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # PyAV errors derive from OSError/ValueError depending on the cause.
        if type(e).__module__.startswith("av"):
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")
        raise

    text = "".join(s.text for s in segments).strip()
    elapsed = time.perf_counter() - start
    log.info(
        "%.1fs audio (%s, '%s') in %.2fs (%.1fx real time), %d chars",
        info.duration, file.filename, info.language, elapsed, info.duration / max(elapsed, 1e-6), len(text),
    )

    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": info.language,
            "duration": info.duration,
            "text": text,
            "segments": [
                {"id": i, "start": s.start, "end": s.end, "text": s.text} for i, s in enumerate(segments)
            ],
        }
    return {"text": text}


app.include_router(api)
app.include_router(api, prefix="/v1")
