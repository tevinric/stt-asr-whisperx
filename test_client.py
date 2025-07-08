#!/usr/bin/env python3
"""
Test client for Speaker Diarization API
"""

import requests
import time
import json
import argparse
from pathlib import Path

class DiarizationClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def upload_audio(self, audio_file_path: str) -> str:
        """Upload audio file and get job ID"""
        
        if not Path(audio_file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        url = f"{self.base_url}/diarize"
        
        with open(audio_file_path, 'rb') as f:
            files = {'audio_file': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            return result['job_id']
        else:
            raise Exception(f"Upload failed: {response.status_code} - {response.text}")
    
    def get_job_status(self, job_id: str) -> dict:
        """Get job status and results"""
        
        url = f"{self.base_url}/status/{job_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Status check failed: {response.status_code} - {response.text}")
    
    def wait_for_completion(self, job_id: str, timeout: int = 300) -> dict:
        """Wait for job completion with timeout"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            
            print(f"Status: {status['status']}", end="")
            if status.get('progress') is not None:
                print(f" - Progress: {status['progress']*100:.1f}%")
            else:
                print()
            
            if status['status'] == 'completed':
                return status['result']
            elif status['status'] == 'failed':
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
            
            time.sleep(5)
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    def health_check(self) -> dict:
        """Check API health"""
        
        url = f"{self.base_url}/health"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Health check failed: {response.status_code} - {response.text}")

def main():
    parser = argparse.ArgumentParser(description='Test Speaker Diarization API')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds')
    
    args = parser.parse_args()
    
    client = DiarizationClient(args.url)
    
    try:
        # Health check
        print("Checking API health...")
        health = client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Models Loaded: {health['models_loaded']}")
        print(f"Device: {health['device']}")
        print()
        
        # Upload audio
        print(f"Uploading audio file: {args.audio}")
        job_id = client.upload_audio(args.audio)
        print(f"Job ID: {job_id}")
        print()
        
        # Wait for completion
        print("Waiting for processing to complete...")
        result = client.wait_for_completion(job_id, args.timeout)
        print()
        
        # Display results
        print("=== DIARIZATION RESULTS ===")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        print(f"Audio Duration: {result['audio_duration']:.2f} seconds")
        print(f"Total Speakers: {result['total_speakers']}")
        print()
        
        print("Speaker Statistics:")
        for speaker_id, speaker_data in result['speakers'].items():
            print(f"  {speaker_id}:")
            print(f"    - Total Duration: {speaker_data['total_duration']:.2f}s ({speaker_data['percentage']:.1f}%)")
            print(f"    - Segments: {speaker_data['segment_count']}")
            print(f"    - Words: {speaker_data['word_count']}")
            print()
        
        print("=== TRANSCRIPT ===")
        print(result['transcript'])
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
