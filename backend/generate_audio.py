import os
import subprocess
import requests
from pathlib import Path
from typing import List, Dict, Optional

# Python libraries to be installed: requests, google-cloud-texttospeech, pydub(optional)
# Also install ffmpeg for pydub. This is required for smaller audio files to be merged.

"""
Unified Text-to-Speech Interface with Multi-Provider Support

This module provides a unified interface for text-to-speech using various providers
including Azure OpenAI TTS, Google Cloud Text-to-Speech, and local espeak-ng.

SETUP:
Users are expected to set appropriate environment variables for their chosen TTS provider
before calling the generate_audio function.

Environment Variables:

TTS_PROVIDER (default: "local")
    - "azure": Azure OpenAI TTS
    - "gcp": Google Cloud Text-to-Speech
    - "local": Local TTS implementation (default, uses espeak-ng)

TTS_CLOUD_MAX_CHARS (default: 3000)
    - Applies only to cloud providers: "azure" and "gcp"
    - Maximum number of characters per TTS API call
    - If the input text exceeds this limit, it will be split into chunks and synthesized sequentially,
      then concatenated into the final audio file
    - Set to a non-positive value to disable chunking
    - Requires `pydub` (and ffmpeg installed on the system) to merge chunked audio outputs

For Azure TTS:
    AZURE_TTS_KEY: Your Azure OpenAI API key
    AZURE_TTS_ENDPOINT: Azure OpenAI endpoint URL
    AZURE_TTS_VOICE (default: "alloy"): Voice to use (alloy, echo, fable, onyx, nova, shimmer)
    AZURE_TTS_DEPLOYMENT (default: "tts"): Deployment name
    AZURE_TTS_API_VERSION (default: "2025-03-01-preview"): API version

For Google Cloud TTS:
    GOOGLE_API_KEY: Your Google API key (recommended)
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON file (alternative)
    GCP_TTS_VOICE (default: "en-US-Neural2-F"): Voice to use
    GCP_TTS_LANGUAGE (default: "en-US"): Language code

For Local TTS (espeak-ng):
    ESPEAK_VOICE (default: "en"): Voice to use
    ESPEAK_SPEED (default: "150"): Speech rate (words per minute)
    Note: Participants can modify the local provider implementation to use any local TTS solution
    
    Installation:
        # Ubuntu/Debian
        sudo apt-get install espeak-ng
        
        # macOS
        brew install espeak
        
        # CentOS/RHEL
        sudo yum install espeak-ng

Usage:
    from tts import generate_audio
    
    # Basic usage with default provider (local)
    generate_audio("Hello, world!", "output.wav")
    
    # With specific provider
    generate_audio("Hello, world!", "output.mp3", provider="azure")
    
    # With custom voice
    generate_audio("Hello, world!", "output.wav", voice="alloy")
    

"""

def generate_audio(text, output_file, provider=None, voice=None):
    """
    Generate audio from text using the specified TTS provider.
    
    Args:
        text (str): Text to convert to speech
        output_file (str): Output file path
        provider (str, optional): TTS provider to use. Defaults to TTS_PROVIDER env var or "festival"
        voice (str, optional): Voice to use. Defaults to provider-specific default
    
    Returns:
        str: Path to the generated audio file
    
    Raises:
        RuntimeError: If TTS provider is not available or synthesis fails
        ValueError: If text is empty or invalid
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    provider = provider or os.getenv("TTS_PROVIDER", "local").lower()
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Cloud input size limit handling via environment variable
    # TTS_CLOUD_MAX_CHARS: Maximum characters per request for cloud providers (azure/gcp)
    # Defaults to 3000 if not set. Local provider is never chunked.
    max_chars_env = os.getenv("TTS_CLOUD_MAX_CHARS", "3000")
    max_chars = None
    try:
        max_chars = int(max_chars_env)
        if max_chars <= 0:
            max_chars = None
    except (TypeError, ValueError):
        max_chars = 3000

    if provider in ("azure", "gcp") and max_chars and len(text) > max_chars:
        return _generate_cloud_tts_chunked(text, output_file, provider, voice, max_chars)

    if provider == "azure":
        return _generate_azure_tts(text, output_file, voice)
    elif provider == "gcp":
        return _generate_gcp_tts(text, output_file, voice)
    elif provider == "local":
        return _generate_local_tts(text, output_file, voice)
    else:
        raise ValueError(f"Unsupported TTS_PROVIDER: {provider}")

def _chunk_text_by_chars(text, max_chars):
    """Split text into chunks not exceeding max_chars, preferring whitespace boundaries.

    If a single token exceeds max_chars, it will be split hard.
    """
    import re

    if len(text) <= max_chars:
        return [text]

    tokens = re.findall(r"\S+\s*", text)
    chunks = []
    current = ""

    for token in tokens:
        if len(current) + len(token) <= max_chars:
            current += token
        else:
            if current:
                chunks.append(current.strip())
                current = ""
            # If token itself is longer than max_chars, split it
            if len(token) > max_chars:
                start = 0
                while start < len(token):
                    part = token[start:start + max_chars]
                    part = part.strip()
                    if part:
                        chunks.append(part)
                    start += max_chars
            else:
                current = token

    if current.strip():
        chunks.append(current.strip())

    # Final safety: ensure no empty strings
    return [c for c in chunks if c]

def _generate_cloud_tts_chunked(text, output_file, provider, voice, max_chars):
    """Chunk long text for cloud providers and concatenate resulting audio files.

    This function only applies to cloud providers (azure, gcp). Local provider is excluded.
    """
    from pathlib import Path
    from pydub import AudioSegment

    chunks = _chunk_text_by_chars(text, max_chars)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_files = []
    try:
        for index, chunk in enumerate(chunks):
            temp_file = str(output_path.parent / f".tts_chunk_{index}.mp3")
            if provider == "azure":
                _generate_azure_tts(chunk, temp_file, voice)
            elif provider == "gcp":
                _generate_gcp_tts(chunk, temp_file, voice)
            else:
                raise ValueError("Chunked synthesis is only supported for cloud providers 'azure' and 'gcp'.")
            temp_files.append(temp_file)

        # Concatenate audio segments
        combined_audio = None
        for temp_file in temp_files:
            segment = AudioSegment.from_file(temp_file, format="mp3")
            if combined_audio is None:
                combined_audio = segment
            else:
                combined_audio += segment

        # Determine export format from output extension; default to mp3
        suffix = output_path.suffix.lower().lstrip(".") or "mp3"
        combined_audio.export(str(output_path), format=suffix)

        print(f"Chunked {provider.upper()} TTS audio saved to: {output_file} ({len(chunks)} chunks)")
        return str(output_path)
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception:
                pass

def _generate_azure_tts(text, output_file, voice=None):
    """Generate audio using Azure OpenAI TTS."""
    api_key = os.getenv("AZURE_TTS_KEY")
    endpoint = os.getenv("AZURE_TTS_ENDPOINT")
    deployment = os.getenv("AZURE_TTS_DEPLOYMENT", "OpenAICreate-20250819125105")
    voice = voice or os.getenv("AZURE_TTS_VOICE", "alloy")
    api_version = os.getenv("AZURE_TTS_API_VERSION", "2025-03-01-preview")
    
    if not api_key or not endpoint:
        raise ValueError("AZURE_TTS_KEY and AZURE_TTS_ENDPOINT must be set for Azure OpenAI TTS")
    
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": deployment,
        "input": text,
        "voice": voice,
    }
    
    try:
        response = requests.post(
            f"{endpoint}/openai/deployments/{deployment}/audio/speech?api-version={api_version}",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        print(f"Azure OpenAI TTS audio saved to: {output_file}")
        return output_file
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Azure OpenAI TTS failed: {e}")

def _generate_gcp_tts(text, output_file, voice=None):
    """Generate audio using Google Cloud Text-to-Speech."""
    api_key = os.getenv("GOOGLE_API_KEY")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    gcp_voice = voice or os.getenv("GCP_TTS_VOICE", "en-US-Neural2-F")
    language = os.getenv("GCP_TTS_LANGUAGE", "en-US")
    
    if not api_key and not credentials_path:
        raise ValueError("Either GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS must be set for Google Cloud TTS")
    
    try:
        # Use API key if available, otherwise use service account credentials
        if api_key:
            # For API key usage with Google Cloud TTS, use REST with key query parameter
            import requests

            url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
            headers = {"Content-Type": "application/json"}

            payload = {
                "input": {"text": text},
                "voice": {
                    "languageCode": language,
                    "name": gcp_voice
                },
                "audioConfig": {"audioEncoding": "MP3"}
            }

            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            # Decode the base64 audio content
            import base64
            audio_content = base64.b64decode(response.json()["audioContent"])
            
            with open(output_file, "wb") as f:
                f.write(audio_content)
                
        else:
            # Use service account credentials
            if credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            try:
                from google.cloud import texttospeech as g_tts
            except Exception as ie:
                raise RuntimeError("google-cloud-texttospeech is not installed. Install it with: pip install google-cloud-texttospeech")
            client = g_tts.TextToSpeechClient()
            
            input_text = g_tts.SynthesisInput(text=text)
            
            voice_params = g_tts.VoiceSelectionParams(
                language_code=language,
                name=gcp_voice
            )
            
            audio_config = g_tts.AudioConfig(
                audio_encoding=g_tts.AudioEncoding.MP3
            )
            
            response = client.synthesize_speech(
                input=input_text,
                voice=voice_params,
                audio_config=audio_config
            )
            
            with open(output_file, "wb") as f:
                f.write(response.audio_content)
        
        print(f"Google Cloud TTS audio saved to: {output_file}")
        return output_file
        
    except Exception as e:
        raise RuntimeError(f"Google Cloud TTS failed: {e}")


def combine_audio_files(file_paths: List[str], output_file: str, gap_ms: int = 250) -> str:
    """Concatenate multiple audio files with a short silence gap between them.

    Requires pydub and ffmpeg on the system path.
    """
    try:
        from pydub import AudioSegment
    except Exception:
        raise RuntimeError("pydub is required to combine audio. Install it with: pip install pydub (and ensure ffmpeg is installed on your system)")

    silence = AudioSegment.silent(duration=gap_ms)
    combined = None
    for p in file_paths:
        seg = AudioSegment.from_file(p)
        if combined is None:
            combined = seg
        else:
            combined = combined + silence + seg
    if combined is None:
        raise RuntimeError("No audio segments to combine")

    out_suffix = Path(output_file).suffix.lower().lstrip(".") or "mp3"
    combined.export(output_file, format=out_suffix)
    return output_file


def generate_audio_podcast(segments: List[Dict[str, str]], output_file: str, provider: Optional[str] = None, voice_map: Optional[Dict[str, str]] = None, gap_ms: int = 250) -> str:
    """Synthesize a multi-speaker podcast by generating each segment and concatenating them.

    segments: list of {"speaker": str, "text": str}
    voice_map: mapping from speaker name to provider-specific voice name
    """
    if not segments:
        raise ValueError("segments cannot be empty")
    provider = provider or os.getenv("TTS_PROVIDER", "local")
    voice_map = voice_map or {}

    temp_files: List[str] = []
    try:
        base = Path(output_file)
        base.parent.mkdir(parents=True, exist_ok=True)
        for i, seg in enumerate(segments):
            spk = (seg.get("speaker") or "speaker").strip()
            txt = (seg.get("text") or "").strip()
            if not txt:
                continue
            voice = voice_map.get(spk)
            tmp_path = str(base.parent / f".pod_{i:03d}.mp3")
            generate_audio(txt, tmp_path, provider=provider, voice=voice)
            temp_files.append(tmp_path)
        if not temp_files:
            raise RuntimeError("All segments were empty; nothing to synthesize")
        return combine_audio_files(temp_files, str(base), gap_ms=gap_ms)
    finally:
        for p in temp_files:
            try:
                os.remove(p)
            except Exception:
                pass

def _generate_local_tts(text, output_file, voice=None):
    """Generate audio using local TTS implementation (espeak-ng command line).
    
    Note: Participants can modify this function to use any local TTS solution
    such as pyttsx3, say, or other local TTS tools.
    """
    # TODO: Participants can modify this implementation to use any local TTS solution
    # Examples: pyttsx3, say (macOS), gtts, or any other local TTS tool
    
    espeak_voice = voice or os.getenv("ESPEAK_VOICE", "en")
    espeak_speed = os.getenv("ESPEAK_SPEED", "150")
    
    # Create temporary WAV file for espeak-ng
    temp_wav_file = output_file.replace('.mp3', '.wav')
    
    try:
        # Use espeak-ng command line tool
        cmd = [
            'espeak-ng',
            '-v', espeak_voice,
            '-s', str(espeak_speed),
            '-w', temp_wav_file,
            text
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise RuntimeError(f"espeak-ng failed: {result.stderr}")
        
        # Check if temporary WAV file was created
        if not os.path.exists(temp_wav_file):
            raise RuntimeError(f"espeak-ng did not create output file {temp_wav_file}")
        
        # Convert WAV to MP3 if output file is MP3
        if output_file.endswith('.mp3'):
            try:
                from pydub import AudioSegment
                
                # Load WAV file
                audio = AudioSegment.from_wav(temp_wav_file)
                
                # Export as MP3
                audio.export(output_file, format="mp3")
                
                # Remove temporary WAV file
                os.remove(temp_wav_file)
                
                print(f"Local TTS audio saved to: {output_file}")
                return output_file
                
            except ImportError:
                raise RuntimeError("pydub library not installed. Please install it with: pip install pydub")
            except Exception as e:
                raise RuntimeError(f"Failed to convert WAV to MP3: {e}")
        else:
            # If output is WAV, just rename the file
            os.rename(temp_wav_file, output_file)
            print(f"Local TTS audio saved to: {output_file}")
            return output_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("espeak-ng synthesis timed out")
    except FileNotFoundError:
        raise RuntimeError("espeak-ng is not installed. Please install it first:\nUbuntu/Debian: sudo apt-get install espeak-ng\nmacOS: brew install espeak\nCentOS/RHEL: sudo yum install espeak-ng")
    except Exception as e:
        raise RuntimeError(f"Local TTS synthesis error: {str(e)}")

def test_tts_providers():
    """Test all available TTS providers."""
    test_text = "Hello, this is a test of text to speech functionality. "
    test_file = "test_output"
    
    providers = ["local", "azure", "gcp"]
    
    for provider in providers:
        try:
            print(f"\nTesting {provider.upper()} TTS...")
            output_file = generate_audio(test_text, f"{test_file}_{provider}", provider=provider)
            print(f"✅ {provider.upper()} TTS test successful: {output_file}")
        except Exception as e:
            print(f"❌ {provider.upper()} TTS test failed: {e}")

def list_available_providers():
    """List available TTS providers and their status."""
    providers = {
        "local": "Local TTS implementation (uses espeak-ng, can be modified)",
        "azure": "Azure OpenAI TTS (requires AZURE_TTS_KEY and AZURE_TTS_ENDPOINT)",
        "gcp": "Google Cloud Text-to-Speech (requires GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS)"
    }
    
    print("Available TTS Providers:")
    for provider, description in providers.items():
        status = "✅ Available" if _test_provider(provider) else "❌ Not available"
        print(f"  {provider}: {description} - {status}")

def _test_provider(provider):
    """Test if a specific provider is available."""
    try:
        if provider == "local":
            result = subprocess.run(['espeak-ng', '--version'], capture_output=True, timeout=5)
            return result.returncode == 0
        elif provider == "azure":
            return bool(os.getenv("AZURE_TTS_KEY") and os.getenv("AZURE_TTS_ENDPOINT"))
        elif provider == "gcp":
            return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        return False
    except:
        return False

if __name__ == "__main__":
    # Get the provider from environment variable
    provider = os.getenv("TTS_PROVIDER", "local").lower()
    
    print(f"Testing TTS provider: {provider.upper()}")
    print("="*50)
    
    # Test the specified provider
    test_text = "Hello, this is a test of text to speech functionality."
    test_file = f"test_output_{provider}.mp3"
    
    try:
        output_file = generate_audio(test_text, test_file, provider=provider)
        print(f"✅ {provider.upper()} TTS test successful: {output_file}")
    except Exception as e:
        print(f"❌ {provider.upper()} TTS test failed: {e}")
        print("\nAvailable providers and their status:")
        list_available_providers() 