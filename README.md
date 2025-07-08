# Speaker Diarization API

Industry-leading speaker diarization API built with FastAPI, optimized for South African call recordings. Uses state-of-the-art open-source models for accurate speaker separation and transcription.

## Features

- **High Accuracy**: Uses WhisperX (Whisper + speaker diarization) for robust performance
- **Multi-format Support**: MP3, WAV, M4A, FLAC audio formats
- **Noise Robust**: Handles background noise and call center audio quality
- **Scalable**: Designed for high-volume processing
- **GPU Accelerated**: CUDA support for fast processing
- **Real-time Status**: Asynchronous processing with job status tracking
- **Speaker Statistics**: Detailed analytics per speaker
- **Formatted Output**: Clean transcript with timestamps and speaker labels

## Alternative Models

While this implementation uses **WhisperX**, here are other excellent open-source alternatives:

### 1. **WhisperX** (Recommended - Used in this API)
- **Pros**: Extremely robust to noise, excellent transcription, good diarization
- **Best for**: Call center audio, noisy environments, South African accents
- **Performance**: Medium model ~2-3x real-time on GPU

### 2. **Pyannote.audio**
- **Pros**: State-of-the-art diarization accuracy, very configurable
- **Best for**: Clean audio, multiple speakers, academic use
- **Note**: Requires HuggingFace token for best models

### 3. **SpeechBrain**
- **Pros**: End-to-end pipeline, good documentation
- **Best for**: Research applications, custom training

### 4. **NVIDIA NeMo**
- **Pros**: Enterprise-grade, highly optimized
- **Best for**: Large-scale production deployments

## Quick Start

### 1. Local Development

```bash
# Clone and setup
git clone <repository>
cd speaker-diarization-api

# Copy environment file
cp .env.example .env

# Build and run with Docker Compose
chmod +x build.sh
./build.sh full
```

### 2. Test the API

```bash
# Test with sample audio
python test_client.py --audio sample_call.wav --output results.json

# Or use curl
curl -X POST "http://localhost:8000/diarize" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "audio_file=@sample_call.wav"
```

### 3. Check Results

```bash
# Get job status
curl "http://localhost:8000/status/{job_id}"
```

## API Endpoints

### POST `/diarize`
Upload audio file for speaker diarization.

**Parameters:**
- `audio_file`: Audio file (MP3, WAV, M4A, FLAC)

**Response:**
```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

### GET `/status/{job_id}`
Get job status and results.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "transcript": "Speaker_01 [0.00s - 5.23s]: Hello, thank you for calling...",
    "speakers": {
      "Speaker_01": {
        "total_duration": 45.2,
        "percentage": 65.3,
        "segment_count": 12,
        "word_count": 156
      }
    },
    "segments": [...],
    "processing_time": 23.4,
    "audio_duration": 69.2,
    "total_speakers": 2
  }
}
```

### GET `/health`
Health check endpoint.

## Deployment

### Docker Compose (Recommended for Development)

```bash
# Quick start
./build.sh deploy-local

# Manual deployment
docker-compose up -d
```

### Kubernetes (Production)

```bash
# Deploy to cluster
./build.sh deploy-k8s

# Manual deployment
kubectl apply -f k8s-deployment.yaml
```

### Configuration

Edit `.env` file for customization:

```env
# Model configuration
WHISPER_MODEL=medium  # tiny, base, small, medium, large
DEVICE=cuda          # cuda, cpu
BATCH_SIZE=16

# API settings
API_WORKERS=1
MAX_CONCURRENT_JOBS=5

# Optional: HuggingFace token for better models
HUGGINGFACE_TOKEN=your_token_here
```

## Performance Optimization

### Hardware Requirements

**Minimum:**
- 8GB RAM
- 4 CPU cores
- GPU with 6GB VRAM (optional but recommended)

**Recommended:**
- 16GB RAM
- 8+ CPU cores
- GPU with 8GB+ VRAM (Tesla V100, RTX 3080, etc.)

### Performance Tips

1. **Use GPU**: 5-10x faster processing
2. **Batch Processing**: Process multiple files simultaneously
3. **Model Selection**: 
   - `tiny`: Fastest, lower accuracy
   - `medium`: Good balance (recommended)
   - `large`: Best accuracy, slower
4. **Audio Preprocessing**: Convert to WAV 16kHz mono for best performance

### Scaling for High Volume

1. **Horizontal Scaling**: Deploy multiple API instances
2. **Load Balancing**: Use nginx or cloud load balancer
3. **Queue Management**: Implement Redis/RabbitMQ for job queuing
4. **Caching**: Cache models in shared storage
5. **Monitoring**: Use Prometheus + Grafana

## Troubleshooting

### Common Issues

**GPU Not Detected:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Install CUDA toolkit
# Follow: https://developer.nvidia.com/cuda-toolkit
```

**Out of Memory:**
- Reduce batch size in `.env`
- Use smaller Whisper model
- Process shorter audio segments

**Poor Accuracy:**
- Ensure audio is clear (>16kHz, minimal noise)
- Try larger Whisper model
- Check if speakers overlap significantly

**Slow Processing:**
- Enable GPU acceleration
- Reduce audio file size
- Use faster storage (SSD)

### Logs and Debugging

```bash
# View API logs
docker-compose logs -f speaker-diarization-api

# Debug mode
docker run -it --rm speaker-diarization-api:latest python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"
```

## Development

### Code Structure

```
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Local deployment
├── k8s-deployment.yaml # Kubernetes manifests
├── build.sh           # Build and deployment script
├── test_client.py     # Test client
└── .env.example       # Environment template
```

### Adding Custom Models

To use different diarization models, modify `main.py`:

```python
# Example: Using pyannote instead of WhisperX
from pyannote.audio import Pipeline

def load_models():
    models['diarization'] = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="your_token"
    )
```

### API Extensions

Add new endpoints in `main.py`:

```python
@app.post("/batch-diarize")
async def batch_diarize(files: List[UploadFile]):
    # Process multiple files
    pass
```

## Production Checklist

- [ ] Configure proper authentication
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation
- [ ] Set up backup and recovery
- [ ] Configure SSL/TLS certificates
- [ ] Set resource limits and quotas
- [ ] Configure auto-scaling policies
- [ ] Set up health checks and alerts
- [ ] Test disaster recovery procedures

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error details
3. Open an issue with audio sample and error details
