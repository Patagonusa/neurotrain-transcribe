import os
import tempfile
import subprocess
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import whisper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NeuroTrain Transcribe API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model on startup
MODEL_SIZE = os.getenv("tiny", "base")
logger.info(f"Loading Whisper model: {tiny}")
model = whisper.load_model(tiny)

class TranscriptionResponse(BaseModel):
    transcript: str
    language: str
    tldr: str
    duration: Optional[float] = None
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    model: str
    version: str = "1.0.0"

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        model=tiny
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model=tiny
    )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None
):
    """
    Transcribe audio file (supports OGG, MP3, WAV, etc.)
    """
    logger.info(f"Received file: {file.filename}, type: {file.content_type}")
    
    # Validate file size (max 25MB for Render free tier)
    contents = await file.read()
    file_size = len(contents) / (1024 * 1024)  # MB
    if file_size > 25:
        raise HTTPException(status_code=413, detail="File too large. Max 25MB.")
    
    wav_path = None
    tmp_input_path = None
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp_file:
            tmp_file.write(contents)
            tmp_input_path = tmp_file.name
        
        # Convert to WAV using ffmpeg
        wav_path = tmp_input_path.replace('.ogg', '.wav')
        
        # Run ffmpeg conversion
        result = subprocess.run([
            'ffmpeg', '-i', tmp_input_path,
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono
            '-f', 'wav',
            wav_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise HTTPException(status_code=500, detail="Audio conversion failed")
        
        # Transcribe with Whisper
        logger.info("Starting transcription...")
        result = model.transcribe(
            wav_path,
            language=language,
            fp16=False  # Disable FP16 for CPU on Render
        )
        
        text = result["text"].strip()
        detected_language = result.get("language", "unknown")
        
        # Simple summarization (first 150 chars)
        tldr = text[:150] + "..." if len(text) > 150 else text
        
        # Calculate duration if possible
        duration = result.get("duration", None)
        
        logger.info(f"Transcription complete. Language: {detected_language}")
        
        return TranscriptionResponse(
            transcript=text,
            language=detected_language,
            tldr=tldr,
            duration=duration
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary files
        for path in [tmp_input_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
