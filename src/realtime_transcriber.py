"""
Real-time transcription module with dual audio capture and speaker diarization.
Captures both system audio (via SystemAudioDump) and microphone audio,
transcribes in real-time using mlx-whisper (GPU accelerated on Apple Silicon),
and identifies speakers.
"""

import logging
import threading
import queue
import time
import subprocess
import os
import sys
import wave
import io
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None
    SOUNDDEVICE_AVAILABLE = False

logger = logging.getLogger(__name__)


# Common Whisper hallucination phrases on silence/noise
HALLUCINATION_PHRASES = {
    "you", "thank you", "thanks", "thanks for watching",
    "bye", "goodbye", "see you", "okay", "ok",
    "hmm", "um", "uh", "ah", "oh", "mm-hmm", "mm",
    ".", "..", "...", "!", "?",
    "thanks for watching.", "thank you for watching.",
    "subscribe", "like and subscribe",
    "i love you", "i like you", "bye, little girl",
    "team team", "hello", "hi", "yes", "no",
}

# Phrases that are hallucinations when repeated
REPETITION_TRIGGER_WORDS = {
    "okay", "ok", "thank you", "thanks", "hello", "hi",
    "yes", "no", "bye", "mm-hmm", "uh-huh", "right",
    "i love you", "i like you", "team", "you",
}


def is_repetitive_hallucination(text: str) -> bool:
    """
    Detect repetitive hallucination patterns like 'okay, okay, okay...'
    or 'thank you. thank you.' etc.

    Returns True if the text appears to be a repetitive hallucination.
    """
    text_lower = text.lower().strip()

    # Remove common punctuation for analysis
    cleaned = text_lower.replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ")
    words = [w.strip() for w in cleaned.split() if w.strip()]

    if not words:
        return True

    # Check for single word/phrase repeated multiple times
    if len(words) >= 3:
        # Count unique words
        unique_words = set(words)

        # If very few unique words compared to total, likely repetition
        # e.g., "okay okay okay okay" = 1 unique word, 4 total
        if len(unique_words) <= 2 and len(words) >= 4:
            # Check if the dominant word is a trigger word
            for word in unique_words:
                if word in REPETITION_TRIGGER_WORDS:
                    word_count = words.count(word)
                    if word_count >= 3:  # Same word repeated 3+ times
                        logger.debug(f"Detected repetitive hallucination: '{text}' ('{word}' x{word_count})")
                        return True

    # Check for phrase repetition patterns like "thank you. thank you."
    # Split by common phrase boundaries
    phrases = [p.strip() for p in text_lower.replace(".", ",").split(",") if p.strip()]
    if len(phrases) >= 2:
        # Check if all phrases are the same or very similar
        unique_phrases = set(phrases)
        if len(unique_phrases) == 1 and phrases[0] in HALLUCINATION_PHRASES:
            logger.debug(f"Detected repeated phrase hallucination: '{text}'")
            return True

        # Check for high phrase repetition rate
        if len(unique_phrases) <= 2 and len(phrases) >= 3:
            most_common = max(unique_phrases, key=lambda p: phrases.count(p))
            if phrases.count(most_common) >= 3:
                logger.debug(f"Detected phrase repetition: '{text}'")
                return True

    return False


def is_hallucination(text: str) -> bool:
    """
    Check if text is a known hallucination or repetitive pattern.
    """
    text_lower = text.lower().strip()

    # Check direct hallucination phrases
    if text_lower in HALLUCINATION_PHRASES:
        return True

    # Check for repetitive patterns
    if is_repetitive_hallucination(text):
        return True

    return False


@dataclass
class TranscriptSegment:
    """A segment of transcribed text with metadata."""
    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    source: str = "unknown"  # "microphone", "system", or "mixed"
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker": self.speaker,
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }

    def format_log_entry(self) -> str:
        """Format as a log entry with timestamp and speaker."""
        time_str = f"[{self.start_time:.2f}s - {self.end_time:.2f}s]"
        speaker_str = f"[{self.speaker}]" if self.speaker else "[Unknown]"
        return f"{time_str} {speaker_str}: {self.text}"


class AudioBuffer:
    """Thread-safe audio buffer for accumulating audio chunks."""

    def __init__(self, sample_rate: int = 16000, max_duration: float = 30.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.buffer = np.array([], dtype=np.float32) if NUMPY_AVAILABLE else []
        self.lock = threading.Lock()
        self.start_time = time.time()

    def add_chunk(self, chunk: np.ndarray) -> None:
        """Add audio chunk to buffer."""
        with self.lock:
            if NUMPY_AVAILABLE:
                self.buffer = np.concatenate([self.buffer, chunk.flatten()])
                # Trim if exceeds max duration
                if len(self.buffer) > self.max_samples:
                    self.buffer = self.buffer[-self.max_samples:]
            else:
                self.buffer.extend(chunk.flatten().tolist())
                if len(self.buffer) > self.max_samples:
                    self.buffer = self.buffer[-self.max_samples:]

    def get_audio(self, duration: Optional[float] = None) -> np.ndarray:
        """Get audio from buffer, optionally limited to duration."""
        with self.lock:
            if duration:
                samples = int(self.sample_rate * duration)
                return np.array(self.buffer[-samples:], dtype=np.float32)
            return np.array(self.buffer, dtype=np.float32)

    def clear(self) -> None:
        """Clear the buffer."""
        with self.lock:
            self.buffer = np.array([], dtype=np.float32) if NUMPY_AVAILABLE else []
            self.start_time = time.time()

    def duration(self) -> float:
        """Get current buffer duration in seconds."""
        with self.lock:
            return len(self.buffer) / self.sample_rate


class SystemAudioCapture:
    """Captures system audio using platform-specific loopback devices."""

    def __init__(self, sample_rate: Optional[int] = None):
        self.sample_rate = sample_rate or 16000
        self.running = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.stream: Optional[Any] = None
        self.device_id: Optional[int] = None
        self.channel_count = 1
        self.extra_settings: Optional[Any] = None
        self.capture_mode = "unsupported"

        self.device_id = self._find_system_audio_device()

    def _find_system_audio_device(self) -> Optional[int]:
        """Find a platform-appropriate system audio capture device."""
        if sys.platform == "darwin":
            return self._find_blackhole_device()
        if sys.platform.startswith("win"):
            return self._find_windows_loopback_device()

        logger.info("System audio capture is only configured for macOS and Windows")
        return None

    def _find_blackhole_device(self) -> Optional[int]:
        """Find BlackHole virtual audio device."""
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice not available for system audio capture")
            return None

        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                device_name = device.get('name', '').lower()
                # Look for BlackHole or other virtual audio devices
                if 'blackhole' in device_name:
                    if device.get('max_input_channels', 0) > 0:
                        self.channel_count = max(1, min(2, int(device.get('max_input_channels', 1))))
                        self.sample_rate = int(device.get('default_samplerate') or self.sample_rate)
                        self.capture_mode = "blackhole"
                        logger.info(f"Found BlackHole device: {device['name']} (ID: {i})")
                        return i

            logger.warning("BlackHole device not found. Install BlackHole for system audio capture.")
            logger.warning("Download from: https://existential.audio/blackhole/")
            logger.warning("After installing, create a Multi-Output Device in Audio MIDI Setup")
            return None
        except Exception as e:
            logger.error(f"Error finding BlackHole device: {e}")
            return None

    def _find_windows_loopback_device(self) -> Optional[int]:
        """Find the default output device and capture it through WASAPI loopback."""
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice not available for system audio capture")
            return None

        if not hasattr(sd, "WasapiSettings"):
            logger.warning("WASAPI loopback is not available in this sounddevice build")
            return None

        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            wasapi_hostapis = {
                index for index, api in enumerate(hostapis)
                if "wasapi" in api.get("name", "").lower()
            }

            if not wasapi_hostapis:
                logger.warning("No WASAPI host API found for system audio capture")
                return None

            default_output = None
            try:
                default_devices = sd.default.device
                if isinstance(default_devices, (list, tuple)) and len(default_devices) > 1:
                    default_output = default_devices[1]
            except Exception:
                default_output = None

            candidate_ids = []
            if isinstance(default_output, int) and default_output >= 0:
                candidate_ids.append(default_output)

            for i, device in enumerate(devices):
                if i in candidate_ids:
                    continue
                if device.get("hostapi") in wasapi_hostapis and device.get("max_output_channels", 0) > 0:
                    candidate_ids.append(i)

            for device_id in candidate_ids:
                device = devices[device_id]
                if device.get("hostapi") not in wasapi_hostapis:
                    continue
                output_channels = int(device.get("max_output_channels", 0))
                if output_channels <= 0:
                    continue

                self.channel_count = max(1, min(2, output_channels))
                self.sample_rate = int(device.get("default_samplerate") or self.sample_rate or 48000)
                self.extra_settings = sd.WasapiSettings(loopback=True)
                self.capture_mode = "wasapi-loopback"
                logger.info(f"Found WASAPI loopback device: {device['name']} (ID: {device_id})")
                return device_id

            logger.warning("No WASAPI output device available for system audio capture")
            return None
        except Exception as e:
            logger.error(f"Error finding Windows loopback device: {e}")
            return None

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            logger.warning(f"System audio callback status: {status}")
        if NUMPY_AVAILABLE and self.running:
            chunk = indata.copy()
            if chunk.ndim > 1 and chunk.shape[1] > 1:
                chunk = np.mean(chunk, axis=1, keepdims=True)
            self.audio_queue.put(chunk.astype(np.float32, copy=False))

    def start(self) -> bool:
        """Start capturing system audio using the configured capture mode."""
        if self.device_id is None:
            if sys.platform == "darwin":
                logger.error("BlackHole device not available")
                logger.error("To capture system audio:")
                logger.error("  1. Install BlackHole")
                logger.error("  2. Open Audio MIDI Setup")
                logger.error("  3. Create Multi-Output Device with your speakers + BlackHole")
                logger.error("  4. Set Multi-Output as your system output")
            elif sys.platform.startswith("win"):
                logger.error("Windows loopback device not available")
                logger.error("Make sure your default playback device is active and WASAPI is enabled")
            return False

        if not SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice not available")
            return False

        if self.running:
            logger.warning("System audio capture already running")
            return True

        try:
            stream_kwargs = dict(
                samplerate=self.sample_rate,
                channels=self.channel_count,
                dtype='float32',
                device=self.device_id,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.2)  # 200ms blocks
            )
            if self.extra_settings is not None:
                stream_kwargs["extra_settings"] = self.extra_settings

            self.stream = sd.InputStream(**stream_kwargs)
            self.stream.start()
            self.running = True
            logger.info(
                f"System audio capture started via {self.capture_mode} "
                f"(device {self.device_id}, sample_rate={self.sample_rate}, channels={self.channel_count})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start system audio capture: {e}")
            return False

    def stop(self) -> None:
        """Stop capturing system audio."""
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Error stopping system audio stream: {e}")
            self.stream = None
        logger.info("System audio capture stopped")

    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get next audio chunk from queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class MicrophoneCapture:
    """Captures microphone audio using sounddevice."""

    def __init__(self, sample_rate: int = 16000, device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device = device
        self.running = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.stream: Optional[Any] = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        if NUMPY_AVAILABLE:
            self.audio_queue.put(indata.copy())

    def start(self) -> bool:
        """Start capturing microphone audio."""
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice not available")
            return False

        if self.running:
            logger.warning("Microphone capture already running")
            return True

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.device,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.2)  # 200ms blocks
            )
            self.stream.start()
            self.running = True
            logger.info("Microphone capture started")
            return True
        except Exception as e:
            logger.error(f"Failed to start microphone capture: {e}")
            return False

    def stop(self) -> None:
        """Stop capturing microphone audio."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        logger.info("Microphone capture stopped")

    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get next audio chunk from queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None



class RealtimeTranscriber:
    """
    Real-time transcription engine with dual audio capture.
    Captures system audio and microphone, transcribes using faster-whisper,
    and provides speaker diarization.
    """

    def __init__(
        self,
        model_size: str = "small",
        language: str = "en",
        mic_device: Optional[int] = None,
        enable_system_audio: bool = True,
        enable_microphone: bool = True,
        transcription_callback: Optional[Callable[[TranscriptSegment], None]] = None,
        chunk_duration: float = 5.0,  # Transcribe every N seconds
        overlap_duration: float = 1.0,  # Overlap between chunks
    ):
        self.model_size = model_size
        self.language = language
        self.enable_system_audio = enable_system_audio
        self.enable_microphone = enable_microphone
        self.transcription_callback = transcription_callback
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration

        # Audio capture
        self.system_capture: Optional[SystemAudioCapture] = None
        self.mic_capture: Optional[MicrophoneCapture] = None

        # Audio buffers
        self.system_buffer = AudioBuffer(sample_rate=16000)
        self.mic_buffer = AudioBuffer(sample_rate=16000)

        # Transcription model
        self.model = None
        self.model_loaded = False

        # State
        self.running = False
        self.transcription_thread: Optional[threading.Thread] = None
        self.audio_thread: Optional[threading.Thread] = None
        self.segments: List[TranscriptSegment] = []
        self.segments_lock = threading.Lock()

        # Timing
        self.start_time: float = 0
        self.last_transcription_time: float = 0

    def load_model(self) -> bool:
        """Initialize mlx-whisper for GPU-accelerated transcription on Apple Silicon."""
        try:
            import mlx_whisper

            # Map model sizes to mlx-community HuggingFace repos
            # Note: Models need "-mlx" suffix except for "tiny"
            model_map = {
                "tiny": "mlx-community/whisper-tiny",
                "base": "mlx-community/whisper-base-mlx",
                "small": "mlx-community/whisper-small-mlx",
                "medium": "mlx-community/whisper-medium-mlx",
                "large": "mlx-community/whisper-large-v3-mlx",
                "large-v2": "mlx-community/whisper-large-v2-mlx",
                "large-v3": "mlx-community/whisper-large-v3-mlx",
            }

            self.mlx_model_path = model_map.get(self.model_size, f"mlx-community/whisper-{self.model_size}-mlx")

            logger.info(f"Loading mlx-whisper model: {self.mlx_model_path} (GPU accelerated on Apple Silicon)")

            # mlx-whisper doesn't require pre-loading, it loads on first transcribe call
            # But we store the reference to the module for transcription
            self.mlx_whisper = mlx_whisper
            self.model = True  # Flag to indicate model is ready
            self.model_loaded = True
            logger.info("mlx-whisper initialized successfully (GPU accelerated)")
            return True
        except ImportError:
            logger.warning("mlx-whisper not available, falling back to faster-whisper (CPU)")
            return self._load_faster_whisper_fallback()
        except Exception as e:
            logger.error(f"Failed to initialize mlx-whisper: {e}")
            return self._load_faster_whisper_fallback()

    def _load_faster_whisper_fallback(self) -> bool:
        """Fallback to faster-whisper if mlx-whisper is not available."""
        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading faster-whisper model (CPU fallback): {self.model_size}")

            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8"
            )
            self.mlx_whisper = None  # Indicate we're using faster-whisper
            self.model_loaded = True
            logger.info("faster-whisper model loaded successfully (CPU)")
            return True
        except ImportError:
            logger.error("Neither mlx-whisper nor faster-whisper installed.")
            return False
        except Exception as e:
            logger.error(f"Failed to load faster-whisper: {e}")
            return False

    def start(self) -> bool:
        """Start real-time transcription."""
        if self.running:
            logger.warning("Transcription already running")
            return True

        # Load model if not loaded
        if not self.model_loaded:
            if not self.load_model():
                return False

        # Start audio capture
        if self.enable_system_audio:
            self.system_capture = SystemAudioCapture()
            if not self.system_capture.start():
                logger.warning("System audio capture not available")
                self.system_capture = None
            else:
                self.system_buffer = AudioBuffer(sample_rate=self.system_capture.sample_rate)

        if self.enable_microphone:
            self.mic_capture = MicrophoneCapture()
            if not self.mic_capture.start():
                logger.warning("Microphone capture not available")
                self.mic_capture = None

        if not self.system_capture and not self.mic_capture:
            logger.error("No audio capture available")
            return False

        self.running = True
        self.start_time = time.time()
        self.last_transcription_time = 0

        # Start audio collection thread
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()

        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self.transcription_thread.start()

        logger.info("Real-time transcription started")
        return True

    def stop(self) -> List[TranscriptSegment]:
        """Stop transcription and return all segments."""
        self.running = False

        # Stop audio capture
        if self.system_capture:
            self.system_capture.stop()
        if self.mic_capture:
            self.mic_capture.stop()

        # Wait for threads
        if self.audio_thread:
            self.audio_thread.join(timeout=2)
        if self.transcription_thread:
            self.transcription_thread.join(timeout=5)

        logger.info("Real-time transcription stopped")

        with self.segments_lock:
            return list(self.segments)

    def _audio_loop(self) -> None:
        """Collect audio from capture sources."""
        while self.running:
            # Collect system audio
            if self.system_capture:
                chunk = self.system_capture.get_audio_chunk(timeout=0.05)
                if chunk is not None:
                    self.system_buffer.add_chunk(chunk)

            # Collect microphone audio
            if self.mic_capture:
                chunk = self.mic_capture.get_audio_chunk(timeout=0.05)
                if chunk is not None:
                    self.mic_buffer.add_chunk(chunk)

    def _transcription_loop(self) -> None:
        """Periodically transcribe accumulated audio."""
        while self.running:
            current_time = time.time() - self.start_time

            # Wait for enough audio to accumulate
            if current_time - self.last_transcription_time < self.chunk_duration:
                time.sleep(0.1)
                continue

            # Transcribe microphone audio (primary source for "You")
            if self.mic_buffer.duration() >= self.chunk_duration:
                self._transcribe_buffer(
                    self.mic_buffer,
                    source="microphone",
                    speaker="You"
                )

            # Transcribe system audio (for "Others")
            if self.system_buffer.duration() >= self.chunk_duration:
                self._transcribe_buffer(
                    self.system_buffer,
                    source="system",
                    speaker="Other"
                )

            self.last_transcription_time = current_time

    def _transcribe_buffer(
        self,
        buffer: AudioBuffer,
        source: str,
        speaker: str
    ) -> None:
        """Transcribe audio from a buffer using mlx-whisper (GPU) or faster-whisper (CPU fallback)."""
        if not self.model:
            return

        # Get audio and resample to 16kHz if needed
        audio = buffer.get_audio(duration=self.chunk_duration + self.overlap_duration)

        if len(audio) < 1600:  # Less than 0.1s of audio
            return

        # Resample if needed
        if NUMPY_AVAILABLE and buffer.sample_rate != 16000:
            audio = self._resample(audio, buffer.sample_rate, 16000)

        try:
            current_time = time.time() - self.start_time

            # Use mlx-whisper if available (GPU accelerated)
            if hasattr(self, 'mlx_whisper') and self.mlx_whisper is not None:
                result = self._transcribe_with_mlx(audio, source)
                if result:
                    self._process_mlx_result(result, current_time, source, speaker)
            else:
                # Fallback to faster-whisper (CPU)
                self._transcribe_with_faster_whisper(audio, current_time, source, speaker)

        except Exception as e:
            logger.error(f"Transcription error: {e}")

        # Clear processed audio (keep overlap)
        buffer.clear()

    def _transcribe_with_mlx(self, audio, source: str):
        """Transcribe using mlx-whisper (GPU accelerated on Apple Silicon)."""
        try:
            result = self.mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.mlx_model_path,
                language=self.language,
                verbose=False,
                condition_on_previous_text=False,  # Better for real-time
            )
            return result
        except Exception as e:
            logger.error(f"mlx-whisper transcription error: {e}")
            return None

    def _process_mlx_result(self, result, current_time: float, source: str, speaker: str):
        """Process mlx-whisper transcription result."""
        if not result or "segments" not in result:
            return

        for segment in result["segments"]:
            text = segment.get("text", "").strip()

            # Skip empty or very short segments (less than 4 chars)
            if not text or len(text) < 4:
                continue

            # Skip known hallucination phrases and repetitive patterns
            if is_hallucination(text):
                logger.debug(f"Skipping hallucination: '{text}'")
                continue

            transcript_segment = TranscriptSegment(
                text=text,
                start_time=current_time - self.chunk_duration + segment.get("start", 0),
                end_time=current_time - self.chunk_duration + segment.get("end", 0),
                speaker=speaker,
                source=source,
                confidence=segment.get("avg_logprob", 0.0)
            )

            with self.segments_lock:
                self.segments.append(transcript_segment)

            # Call callback if provided
            if self.transcription_callback:
                try:
                    self.transcription_callback(transcript_segment)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            logger.info(f"[{source}] {speaker}: {text}")

    def _transcribe_with_faster_whisper(self, audio, current_time: float, source: str, speaker: str):
        """Fallback transcription using faster-whisper (CPU)."""
        # Transcribe with source-specific VAD settings
        if source == "microphone":
            segments, _ = self.model.transcribe(
                audio,
                language=self.language,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=400,
                    threshold=0.3
                )
            )
        else:
            segments, _ = self.model.transcribe(
                audio,
                language=self.language,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                    threshold=0.5  # Higher threshold for system audio to reduce noise
                )
            )

        for segment in segments:
            text = segment.text.strip()

            # Skip empty or very short segments (less than 4 chars)
            if not text or len(text) < 4:
                continue

            # Skip known hallucination phrases and repetitive patterns
            if is_hallucination(text):
                logger.debug(f"Skipping hallucination: '{text}'")
                continue

            transcript_segment = TranscriptSegment(
                text=text,
                start_time=current_time - self.chunk_duration + segment.start,
                end_time=current_time - self.chunk_duration + segment.end,
                speaker=speaker,
                source=source,
                confidence=segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.0
            )

            with self.segments_lock:
                self.segments.append(transcript_segment)

            # Call callback if provided
            if self.transcription_callback:
                try:
                    self.transcription_callback(transcript_segment)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            logger.info(f"[{source}] {speaker}: {segment.text.strip()}")

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio

        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)

        if NUMPY_AVAILABLE:
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        return audio

    def get_segments(self) -> List[TranscriptSegment]:
        """Get all transcribed segments."""
        with self.segments_lock:
            return list(self.segments)

    def get_full_transcript(self) -> str:
        """Get full transcript as formatted text."""
        with self.segments_lock:
            lines = []
            for seg in sorted(self.segments, key=lambda s: s.start_time):
                lines.append(seg.format_log_entry())
            return "\n".join(lines)

    def save_transcript(self, filepath: str) -> None:
        """Save transcript to file."""
        with open(filepath, 'w') as f:
            f.write(self.get_full_transcript())
        logger.info(f"Transcript saved to {filepath}")


class LiveTranscriptLogger:
    """
    Logs transcript segments incrementally to a file during recording.
    Provides real-time persistence of transcription.
    """

    def __init__(self, output_dir: str = "transcripts", session_name: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if session_name:
            self.session_name = session_name
        else:
            self.session_name = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.log_file = self.output_dir / f"{self.session_name}_live.txt"
        self.json_file = self.output_dir / f"{self.session_name}_live.json"
        self.segments: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

        # Initialize files
        self._init_files()

    def _init_files(self) -> None:
        """Initialize log files with headers."""
        with open(self.log_file, 'w') as f:
            f.write(f"# Live Transcript - {self.session_name}\n")
            f.write(f"# Started: {datetime.now().isoformat()}\n")
            f.write("# " + "=" * 50 + "\n\n")

    def log_segment(self, segment: TranscriptSegment) -> None:
        """Log a transcript segment to files."""
        with self.lock:
            # Append to text log
            with open(self.log_file, 'a') as f:
                f.write(segment.format_log_entry() + "\n")

            # Add to JSON segments
            self.segments.append(segment.to_dict())

            # Update JSON file
            with open(self.json_file, 'w') as f:
                import json
                json.dump({
                    "session_name": self.session_name,
                    "started": datetime.now().isoformat(),
                    "segments": self.segments
                }, f, indent=2)

    def finalize(self) -> str:
        """Finalize the log and return the file path."""
        with self.lock:
            with open(self.log_file, 'a') as f:
                f.write("\n# " + "=" * 50 + "\n")
                f.write(f"# Ended: {datetime.now().isoformat()}\n")
                f.write(f"# Total segments: {len(self.segments)}\n")

        logger.info(f"Live transcript finalized: {self.log_file}")
        return str(self.log_file)



def create_realtime_transcriber(
    model_size: str = "small",
    language: str = "en",
    enable_system_audio: bool = True,
    enable_microphone: bool = True,
    callback: Optional[Callable[[TranscriptSegment], None]] = None,
    session_name: Optional[str] = None,
    enable_live_logging: bool = True
) -> tuple:
    """
    Factory function to create a configured RealtimeTranscriber with optional live logging.

    Returns:
        tuple: (transcriber, live_logger) - live_logger may be None if disabled
    """
    live_logger = None

    if enable_live_logging:
        live_logger = LiveTranscriptLogger(session_name=session_name)

        # Wrap callback to also log to file
        original_callback = callback

        def logging_callback(segment: TranscriptSegment):
            live_logger.log_segment(segment)
            if original_callback:
                original_callback(segment)

        callback = logging_callback

    transcriber = RealtimeTranscriber(
        model_size=model_size,
        language=language,
        enable_system_audio=enable_system_audio,
        enable_microphone=enable_microphone,
        transcription_callback=callback
    )

    return transcriber, live_logger


# CLI for testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    def on_transcript(segment: TranscriptSegment):
        print(f"\n{segment.format_log_entry()}")

    print("Starting real-time transcription with live logging...")
    print("Press Ctrl+C to stop\n")

    transcriber, live_logger = create_realtime_transcriber(
        model_size="small",
        enable_system_audio=True,
        enable_microphone=True,
        callback=on_transcript,
        enable_live_logging=True
    )

    if not transcriber.start():
        print("Failed to start transcription")
        sys.exit(1)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping transcription...")
        segments = transcriber.stop()

        print(f"\n\nTotal segments: {len(segments)}")
        print("\n--- Full Transcript ---")
        print(transcriber.get_full_transcript())

        # Finalize live log
        if live_logger:
            log_path = live_logger.finalize()
            print(f"\nLive transcript saved to: {log_path}")

        # Also save final transcript
        output_path = f"transcripts/realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs("transcripts", exist_ok=True)
        transcriber.save_transcript(output_path)
        print(f"Final transcript saved to: {output_path}")
