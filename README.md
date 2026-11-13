# jetson-openwhispr-server

A speech-to-text server for the [OpenWhispr](https://openwhispr.com) app's
**Self-Hosted** mode, running Whisper on the GPU of an NVIDIA Jetson Orin Nano
(8GB, JetPack 6.2 / L4T R36.x).

```
OpenWhispr (Mac/PC)  ──POST /audio/transcriptions (webm/opus)──▶  Jetson :8178
                     ◀──────────── {"text": "..."} ─────────────  FastAPI ─▶ faster-whisper (CUDA)
```

## OpenWhispr settings

**Settings → Speech-to-Text → Self-Hosted** (set it on each tab you use:
Dictation, Note Recording, Audio Upload):

| Field | Value |
|-------|-------|
| Server URL | `http://<jetson-ip>:8178` |
| Model | `Whisper-Large-v3-Turbo` |

`<jetson-ip>` is the Jetson's LAN address (`hostname -I` on the Jetson), or
its Tailscale address (`tailscale ip -4`) to reach it from outside the LAN.

The Model field must match the name the server lists at `/models`
(`SERVED_MODEL_NAME`), or the panel reports the model as not found.

Setting a language in OpenWhispr instead of auto-detect roughly halves the
response time (see [Performance](#performance)).

## Running it

```bash
docker compose up -d --build     # build and start; restarts on boot (restart: unless-stopped)
docker compose logs -f           # one line per transcription, with timing
docker compose down              # stop
curl http://localhost:8178/health
```

The first start downloads the model (~1.6GB for `large-v3-turbo`) into
`models/`, about a minute. Later starts take ~10s.

Test without the app:

```bash
curl http://localhost:8178/audio/transcriptions -F file=@some_audio.wav -F language=en
```

## Configuration

Copy `.env.example` to `.env`, edit, then `docker compose up -d`.

| Variable | Default | Notes |
|----------|---------|-------|
| `PORT` | `8178` | Host port. |
| `WHISPER_MODEL` | `large-v3-turbo` | Any faster-whisper size or Hugging Face repo (`small`, `medium`, `distil-large-v3`, ...). `.en` models are English-only. |
| `SERVED_MODEL_NAME` | `Whisper-Large-v3-Turbo` | What to type in OpenWhispr's Model field. |
| `COMPUTE_TYPE` | `float16` | `int8_float16` uses less memory. |
| `BEAM_SIZE` | `5` | No measurable speed difference from `1` with the turbo model. |
| `VAD_FILTER` | `true` | Drops silence before transcribing; avoids hallucinated text on quiet recordings. |

## API

Paths are served both bare and under `/v1`, since OpenWhispr appends them to
the Server URL as typed.

| Endpoint | Purpose |
|----------|---------|
| `POST /audio/transcriptions` | multipart: `file` (required), `model` (ignored), `language`, `prompt` (OpenWhispr's custom dictionary), `response_format` (`json`, `text`, `verbose_json`), `temperature`. Returns `{"text": "..."}`. |
| `GET /models` | Lists `SERVED_MODEL_NAME`; OpenWhispr's connection check reads it. |
| `GET /health` | Model, device and compute type. `device` must be `cuda`. |

One model is loaded at startup and requests are transcribed one at a time.

## Performance

Measured on this board (25W power mode) with `large-v3-turbo`, `float16`, on
an 11s clip sent as webm/opus:

| Request | Time |
|---------|------|
| language set (`en`) | ~2.1s |
| language auto-detect | ~3.8s |

Memory: the container adds about 3GB to the board's used memory (8GB shared
between CPU and GPU).

## Why this design

| Option | Verdict |
|--------|---------|
| **Small FastAPI server on `dustynv/faster-whisper`** (this repo) | Chosen. The image is already on the board and proven on its GPU; only three pip packages are added. Full control over the paths and model list OpenWhispr expects. |
| [speaches](https://github.com/speaches-ai/speaches) (faster-whisper-server) | Same engine with an OpenAI API, but its CUDA images are x86-only; on a Jetson it would have to be installed into the same base image anyway, with many more dependencies. |
| whisper.cpp `whisper-server` | Needs a CUDA build from source, and has no `/models` list. |
| Original `openai-whisper` (PyTorch) | No prebuilt image for JetPack 6.2; slower and heavier than CTranslate2. |

## Plan

1. **Done: speech-to-text.** This server, on port 8178, started by Docker on boot.
2. **Optional: Language Models (cleanup / chat).** OpenWhispr's Language
   Models → Self-Hosted takes any OpenAI-compatible LLM server. Ollama
   is one option (e.g. with `gemma3:4b`); by default it listens on
   `127.0.0.1:11434` only. To use it from the app, make it listen on the LAN
   (`OLLAMA_HOST=0.0.0.0` in `sudo systemctl edit ollama`) and set the app's
   server URL to `http://<jetson-ip>:11434`. Whisper (~3GB) plus a 4B model
   (~4GB) is tight on 8GB; a smaller LLM or `COMPUTE_TYPE=int8_float16` gives
   more room.
3. **Optional: more speed.** `sudo nvpmodel -m 2` (MAXN SUPER) instead of the
   current 25W mode.

## Notes

- There is no authentication (OpenWhispr's Self-Hosted panel sends no key).
  Anyone on the LAN or tailnet can send audio to port 8178; don't expose it to
  the internet.
- OpenWhispr allows plain `http://` only for private addresses (192.168.x.x,
  10.x, Tailscale's 100.64.0.0/10, `*.local`). The Jetson's address comes from
  DHCP; reserve it in the router, or use the Tailscale address.
- The base image tag is pinned in the `Dockerfile`. Images tagged `r36.4.0`
  run on this board's L4T R36.5.
