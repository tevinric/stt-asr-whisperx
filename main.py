from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import whisperx
import torch
import torchaudio
import tempfile
import os
import asyncio
import logging
from typing import Optional, Dict, Any
import uuid
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Speaker Diarization API",
    description="Industry-leading speaker diarization API for South African call recordings",
    version="1.0.0"
)

# Global model storage
models = {}

class DiarizationResponse(BaseModel):
    job_id: str
    transcript: str
    speakers: Dict[str, Any]
    segments: list
    processing_time: float
    audio_duration: float

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[DiarizationResponse] = None
    error: Optional[str] = None

# In-memory job storage (use Redis in production)
job_storage = {}

def load_models():
    """Load WhisperX models on startup"""
    global models
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"Loading models on device: {device}")
        
        # Load Whisper model - using medium for better accuracy
        models['whisper'] = whisperx.load_model("medium", device, compute_type=compute_type)
        logger.info("WhisperX model loaded successfully")
        
        # Load alignment model
        models['align_model'], models['align_metadata'] = whisperx.load_align_model(
            language_code="en", device=device
        )
        logger.info("Alignment model loaded successfully")
        
        # Load diarization model
        models['diarize_model'] = whisperx.DiarizationPipeline(
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),  # Optional: for better models
            device=device
        )
        logger.info("Diarization model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()

def convert_audio_format(audio_path: str) -> str:
    """Convert audio to WAV format if needed"""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Save as WAV
        output_path = audio_path.replace(Path(audio_path).suffix, "_processed.wav")
        torchaudio.save(output_path, waveform, sample_rate)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {str(e)}")

def process_diarization(audio_path: str, job_id: str) -> Dict[str, Any]:
    """Process speaker diarization using WhisperX"""
    try:
        start_time = datetime.now()
        
        # Update job status
        job_storage[job_id]["status"] = "processing"
        job_storage[job_id]["progress"] = 0.1
        
        # Convert audio format
        processed_audio = convert_audio_format(audio_path)
        
        # Get audio duration
        waveform, sample_rate = torchaudio.load(processed_audio)
        audio_duration = waveform.shape[1] / sample_rate
        
        job_storage[job_id]["progress"] = 0.2
        
        # 1. Transcribe with Whisper
        logger.info("Starting transcription...")
        audio = whisperx.load_audio(processed_audio)
        result = models['whisper'].transcribe(audio, batch_size=16)
        
        job_storage[job_id]["progress"] = 0.4
        
        # 2. Align whisper output
        logger.info("Aligning transcript...")
        result = whisperx.align(
            result["segments"], 
            models['align_model'], 
            models['align_metadata'], 
            audio, 
            device=models['whisper'].device
        )
        
        job_storage[job_id]["progress"] = 0.6
        
        # 3. Assign speaker labels
        logger.info("Performing speaker diarization...")
        diarize_segments = models['diarize_model'](processed_audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        job_storage[job_id]["progress"] = 0.8
        
        # 4. Format output
        formatted_result = format_diarization_output(result, audio_duration)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        formatted_result["processing_time"] = processing_time
        formatted_result["job_id"] = job_id
        
        # Clean up temp files
        os.unlink(processed_audio)
        if processed_audio != audio_path:
            os.unlink(audio_path)
        
        job_storage[job_id]["progress"] = 1.0
        job_storage[job_id]["status"] = "completed"
        job_storage[job_id]["result"] = formatted_result
        
        logger.info(f"Diarization completed in {processing_time:.2f}s")
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error in diarization: {e}")
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = str(e)
        
        # Clean up temp files
        try:
            if os.path.exists(processed_audio):
                os.unlink(processed_audio)
            if processed_audio != audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")

def format_diarization_output(result: Dict, audio_duration: float) -> Dict[str, Any]:
    """Format the diarization output into a clean structure"""
    
    segments = []
    speakers = {}
    full_transcript = ""
    
    current_speaker = None
    current_text = ""
    current_start = None
    current_end = None
    
    for segment in result["segments"]:
        if "speaker" in segment:
            speaker = segment["speaker"]
            text = segment["text"].strip()
            start = segment["start"]
            end = segment["end"]
            
            # Initialize speaker if not seen before
            if speaker not in speakers:
                speakers[speaker] = {
                    "total_duration": 0,
                    "segment_count": 0,
                    "words": []
                }
            
            # Group consecutive segments from same speaker
            if current_speaker == speaker:
                current_text += " " + text
                current_end = end
            else:
                # Save previous speaker segment
                if current_speaker is not None:
                    segments.append({
                        "speaker": current_speaker,
                        "text": current_text.strip(),
                        "start": current_start,
                        "end": current_end,
                        "duration": current_end - current_start
                    })
                    
                    speakers[current_speaker]["total_duration"] += current_end - current_start
                    speakers[current_speaker]["segment_count"] += 1
                    speakers[current_speaker]["words"].extend(current_text.split())
                
                # Start new speaker segment
                current_speaker = speaker
                current_text = text
                current_start = start
                current_end = end
    
    # Don't forget the last segment
    if current_speaker is not None:
        segments.append({
            "speaker": current_speaker,
            "text": current_text.strip(),
            "start": current_start,
            "end": current_end,
            "duration": current_end - current_start
        })
        
        speakers[current_speaker]["total_duration"] += current_end - current_start
        speakers[current_speaker]["segment_count"] += 1
        speakers[current_speaker]["words"].extend(current_text.split())
    
    # Create formatted transcript
    for segment in segments:
        timestamp = f"[{segment['start']:.2f}s - {segment['end']:.2f}s]"
        full_transcript += f"{segment['speaker']} {timestamp}: {segment['text']}\n\n"
    
    # Calculate speaker statistics
    for speaker_id, speaker_data in speakers.items():
        speaker_data["percentage"] = (speaker_data["total_duration"] / audio_duration) * 100
        speaker_data["average_segment_duration"] = speaker_data["total_duration"] / speaker_data["segment_count"]
        speaker_data["word_count"] = len(speaker_data["words"])
    
    return {
        "transcript": full_transcript.strip(),
        "speakers": speakers,
        "segments": segments,
        "audio_duration": audio_duration,
        "total_speakers": len(speakers)
    }

@app.post("/diarize", response_model=Dict[str, str])
async def diarize_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
):
    """
    Upload audio file for speaker diarization
    Supports MP3 and WAV formats
    """
    
    # Validate file type
    if not audio_file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload MP3, WAV, M4A, or FLAC files."
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    job_storage[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "result": None,
        "error": None,
        "created_at": datetime.now()
    }
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process in background
        background_tasks.add_task(process_diarization, tmp_file_path, job_id)
        
        return {"job_id": job_id, "status": "queued"}
        
    except Exception as e:
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a diarization job"""
    
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job["result"],
        error=job["error"]
    )

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a completed job from storage"""
    
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del job_storage[job_id]
    return {"message": "Job deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "active_jobs": len([j for j in job_storage.values() if j["status"] in ["queued", "processing"]])
    }

@app.get("/")
async def root():
    return {
        "message": "Speaker Diarization API",
        "version": "1.0.0",
        "endpoints": {
            "diarize": "POST /diarize - Upload audio for diarization",
            "status": "GET /status/{job_id} - Check job status",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
