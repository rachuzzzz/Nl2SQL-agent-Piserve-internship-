"""
Audio routes — /api/transcribe (ASR), /api/tts (TTS).
"""

import asyncio
import logging
import os
import re
import subprocess
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.security.auth import require_auth
from src.api.deps import REPO_ROOT
from src.api.schemas import TTSRequest

router = APIRouter()

log = logging.getLogger("inspectai.audio")

# ── TTS — lazy-loaded on first /api/tts call ──────────────────────────────────
_tts_model = None

_TTS_DIR    = os.path.join(REPO_ROOT, "models")
_TTS_MODEL  = os.path.join(_TTS_DIR, "kokoro-v1.0.onnx")
_TTS_VOICES = os.path.join(_TTS_DIR, "voices-v1.0.bin")

def _get_tts():
    global _tts_model
    if _tts_model is None:
        from kokoro_onnx import Kokoro
        log.info("Loading Kokoro TTS …")
        _tts_model = Kokoro(_TTS_MODEL, _TTS_VOICES)
        log.info("Kokoro TTS ready.")
    return _tts_model

_MD_STRIP = re.compile(
    r'\*\*(.+?)\*\*'       # **bold**
    r'|\*(.+?)\*'          # *italic*
    r'|`[^`]*`'            # `code`
    r'|#{1,6}\s+'          # ## headings
    r'|\|[^\n]*'           # | table cells
    r'|^[-|=]+$'           # table separator rows
    r'|\[([^\]]*)\]\([^)]*\)',  # [link](url) → keep label
    re.MULTILINE,
)

def _clean_for_tts(text: str) -> str:
    text = _MD_STRIP.sub(lambda m: m.group(1) or m.group(2) or m.group(3) or '', text)
    # Expand common patterns that confuse espeak word-count alignment
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    text = re.sub(r'\$(\d[\d,]*)', r'\1 dollars', text)
    text = re.sub(r'(\d+)/(\d+)', r'\1 out of \2', text)
    text = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1 \2 \3', text)  # ISO dates
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:800]

# ── ASR — lazy-loaded on first /api/transcribe call ───────────────────────────
_asr_pipeline = None

def _get_asr():
    global _asr_pipeline
    if _asr_pipeline is None:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        log.info("Loading openai/whisper-tiny on %s …", "GPU" if device == 0 else "CPU")
        _asr_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=device,
            generate_kwargs={"language": "en", "task": "transcribe"},
        )
        log.info("ASR model ready.")
    return _asr_pipeline


@router.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...), _: None = Depends(require_auth)):
    """
    Accepts any browser audio blob (WebM/Opus, WAV, MP4 …),
    converts to 16 kHz mono WAV via ffmpeg, then transcribes
    with dimavz/whisper-tiny.  Returns {"text": "..."}.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    suffix = ".webm"
    ct = audio.content_type or ""
    if "wav" in ct:   suffix = ".wav"
    elif "mp4" in ct: suffix = ".mp4"
    elif "ogg" in ct: suffix = ".ogg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fin:
        fin.write(raw)
        in_path = fin.name

    out_path = in_path.replace(suffix, "_16k.wav")
    try:
        # Decode & resample to 16 kHz mono PCM WAV
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=422,
                                detail="Audio conversion failed: " + result.stderr.decode()[-300:])

        import soundfile as sf
        data, sr = sf.read(out_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # stereo → mono (safety)

        pipe = _get_asr()
        out  = pipe({"array": data, "sampling_rate": sr})
        text = out.get("text", "").strip()
        return {"text": text}

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


@router.post("/api/tts")
async def tts(req: TTSRequest, _: None = Depends(require_auth)):
    """
    Converts answer text to speech using Kokoro-ONNX (CPU).
    Strips markdown, caps at 800 chars, returns audio/wav.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    clean = _clean_for_tts(req.text)
    if not clean:
        raise HTTPException(status_code=400, detail="Nothing to speak after stripping markdown")

    try:
        import io
        import soundfile as sf
        from fastapi.responses import Response

        import warnings
        kokoro = _get_tts()
        def _synthesize():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*words count mismatch.*")
                return kokoro.create(clean, voice="af_bella", speed=1.05, lang="en-us")
        samples, sample_rate = await asyncio.to_thread(_synthesize)
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")

    except Exception as exc:
        log.exception("TTS error")
        raise HTTPException(status_code=500, detail=str(exc))
