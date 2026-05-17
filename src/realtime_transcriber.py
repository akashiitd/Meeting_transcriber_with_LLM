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


def _is_virtual_audio_device_name(device_name: str) -> bool:
    """Return True for virtual routing devices that shouldn't be used for direct listening."""
    name = device_name.lower()
    virtual_markers = (
        "cable ",
        "vb-audio",
        "stereo mix",
        "what u hear",
        "wave out mix",
        "blackhole",
        "monitor of",
    )
    return any(marker in name for marker in virtual_markers)


def _resample_audio_chunk(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono or multichannel float audio using linear interpolation."""
    if not NUMPY_AVAILABLE or orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    if len(audio) == 0:
        return audio

    target_length = max(1, int(len(audio) * target_sr / orig_sr))
    source_positions = np.arange(len(audio), dtype=np.float32)
    target_positions = np.linspace(0, len(audio) - 1, target_length, dtype=np.float32)
    channels = [
        np.interp(target_positions, source_positions, audio[:, channel_index])
        for channel_index in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


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


class SystemAudioMonitor:
    """Mirror captured virtual-cable audio to a physical output device for live listening."""

    def __init__(self, sample_rate: int, preferred_device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.preferred_device = preferred_device
        self.output_device_id: Optional[int] = None
        self.output_sample_rate = sample_rate
        self.output_channels = 2
        self.audio_queue: queue.Queue = queue.Queue(maxsize=32)
        self.stream: Optional[Any] = None
        self.running = False
        self._current_chunk: Optional[np.ndarray] = None
        self._current_offset = 0

        self.output_device_id = self._find_output_device()

    def _find_output_device(self) -> Optional[int]:
        """Prefer a real playback device instead of another virtual routing endpoint."""
        if not SOUNDDEVICE_AVAILABLE:
            return None

        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()

            preferred_output = self.preferred_device
            if preferred_output is None:
                env_override = os.environ.get("STENOAI_MONITOR_OUTPUT_DEVICE")
                if env_override and env_override.isdigit():
                    preferred_output = int(env_override)

            if isinstance(preferred_output, int):
                device = devices[preferred_output]
                if device.get("max_output_channels", 0) > 0:
                    return preferred_output

            default_output = None
            try:
                if sys.platform.startswith("win"):
                    wasapi_defaults = [
                        api.get("default_output_device")
                        for api in hostapis
                        if "wasapi" in api.get("name", "").lower()
                    ]
                    default_output = next(
                        (
                            device_id
                            for device_id in wasapi_defaults
                            if isinstance(device_id, int) and device_id >= 0
                        ),
                        None,
                    )

                default_devices = sd.default.device
                if default_output is None and isinstance(default_devices, (list, tuple)) and len(default_devices) > 1:
                    default_output = default_devices[1]
            except Exception:
                default_output = None

            def is_physical_output(device_index: int) -> bool:
                device = devices[device_index]
                if int(device.get("max_output_channels", 0)) <= 0:
                    return False
                return not _is_virtual_audio_device_name(device.get("name", ""))

            if isinstance(default_output, int) and default_output >= 0 and is_physical_output(default_output):
                return default_output

            candidates = []
            for index, device in enumerate(devices):
                if not is_physical_output(index):
                    continue

                hostapi = hostapis[device.get("hostapi", -1)] if device.get("hostapi", -1) >= 0 else {}
                host_name = hostapi.get("name", "").lower()
                host_rank = 0 if "wasapi" in host_name else 1 if "directsound" in host_name else 2
                candidates.append((host_rank, index))

            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]))
                return candidates[0][1]
        except Exception as e:
            logger.warning(f"Failed to determine monitor output device: {e}")

        return None

    def _fit_channels(self, chunk: np.ndarray, channel_count: int) -> np.ndarray:
        """Match chunk channels to the output device shape."""
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)

        if chunk.shape[1] == channel_count:
            return chunk

        if channel_count == 1:
            return np.mean(chunk, axis=1, keepdims=True)

        if chunk.shape[1] == 1:
            return np.repeat(chunk, channel_count, axis=1)

        if chunk.shape[1] > channel_count:
            return chunk[:, :channel_count]

        repeats = int(np.ceil(channel_count / chunk.shape[1]))
        expanded = np.tile(chunk, (1, repeats))
        return expanded[:, :channel_count]

    def _output_callback(self, outdata, frames, time_info, status):
        """Feed queued captured audio into the monitoring output stream."""
        if status:
            logger.warning(f"System audio monitor callback status: {status}")

        outdata.fill(0)
        frames_written = 0

        while frames_written < frames:
            if self._current_chunk is None or self._current_offset >= len(self._current_chunk):
                try:
                    self._current_chunk = self.audio_queue.get_nowait()
                    self._current_offset = 0
                except queue.Empty:
                    break

            remaining_chunk = len(self._current_chunk) - self._current_offset
            frames_to_copy = min(frames - frames_written, remaining_chunk)
            segment = self._current_chunk[self._current_offset:self._current_offset + frames_to_copy]
            outdata[frames_written:frames_written + frames_to_copy] = self._fit_channels(
                segment,
                outdata.shape[1],
            )
            self._current_offset += frames_to_copy
            frames_written += frames_to_copy

    def start(self) -> bool:
        """Start monitoring captured system audio to a real playback device."""
        if not SOUNDDEVICE_AVAILABLE or self.output_device_id is None:
            return False

        if self.running:
            return True

        try:
            device_info = sd.query_devices(self.output_device_id)
            self.output_channels = max(1, min(2, int(device_info.get("max_output_channels", 2))))
            self.output_sample_rate = int(device_info.get("default_samplerate") or self.sample_rate)

            self.stream = sd.OutputStream(
                samplerate=self.output_sample_rate,
                channels=self.output_channels,
                dtype="float32",
                device=self.output_device_id,
                callback=self._output_callback,
                blocksize=int(self.output_sample_rate * 0.2),
            )
            self.stream.start()
            self.running = True
            logger.info(
                f"System audio monitoring enabled on device {self.output_device_id} "
                f"(sample_rate={self.output_sample_rate}, channels={self.output_channels})"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to start system audio monitoring: {e}")
            self.stream = None
            return False

    def enqueue_chunk(self, chunk: np.ndarray) -> None:
        """Queue captured audio for local playback on the monitor device."""
        if not self.running or not NUMPY_AVAILABLE:
            return

        try:
            monitor_chunk = np.asarray(chunk, dtype=np.float32)
            if monitor_chunk.ndim == 1:
                monitor_chunk = monitor_chunk.reshape(-1, 1)

            if self.output_sample_rate != self.sample_rate:
                monitor_chunk = _resample_audio_chunk(
                    monitor_chunk,
                    self.sample_rate,
                    self.output_sample_rate,
                )

            while self.audio_queue.qsize() >= 8:
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            self.audio_queue.put_nowait(monitor_chunk)
        except queue.Full:
            pass
        except Exception as e:
            logger.debug(f"Failed to enqueue monitor audio chunk: {e}")

    def stop(self) -> None:
        """Stop the monitor playback stream."""
        self.running = False
        self._current_chunk = None
        self._current_offset = 0

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Error stopping system audio monitor: {e}")
            self.stream = None

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break


class SystemAudioCapture:
    """Captures system audio using platform-specific loopback devices."""

    def __init__(self, sample_rate: Optional[int] = None):
        self.sample_rate = sample_rate or 16000
        self.running = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.stream: Optional[Any] = None
        self.monitor: Optional[SystemAudioMonitor] = None
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
            hostapi_names = {
                index: api.get("name", "")
                for index, api in enumerate(hostapis)
            }
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

            loopback_settings = None
            try:
                loopback_settings = sd.WasapiSettings(loopback=True)
            except TypeError:
                logger.info(
                    "This sounddevice build does not support WasapiSettings(loopback=True); "
                    "falling back to real input devices like VB-Cable or Stereo Mix."
                )

            if loopback_settings is not None:
                for device_id in candidate_ids:
                    device = devices[device_id]
                    if device.get("hostapi") not in wasapi_hostapis:
                        continue
                    output_channels = int(device.get("max_output_channels", 0))
                    if output_channels <= 0:
                        continue

                    self.channel_count = max(1, min(2, output_channels))
                    self.sample_rate = int(device.get("default_samplerate") or self.sample_rate or 48000)
                    self.extra_settings = loopback_settings
                    self.capture_mode = "wasapi-loopback"
                    logger.info(f"Found WASAPI loopback device: {device['name']} (ID: {device_id})")
                    return device_id

            # Fallback: use actual input devices that carry system audio, e.g. VB-Cable or Stereo Mix.
            fallback_patterns = (
                "cable output",
                "stereo mix",
                "what u hear",
                "wave out mix",
                "monitor of",
            )
            fallback_candidates = []
            for i, device in enumerate(devices):
                input_channels = int(device.get("max_input_channels", 0))
                if input_channels <= 0:
                    continue

                name = device.get("name", "")
                name_lower = name.lower()
                if not any(pattern in name_lower for pattern in fallback_patterns):
                    continue

                hostapi_name = hostapi_names.get(device.get("hostapi"), "").lower()
                hostapi_rank = 0 if "wasapi" in hostapi_name else 1 if "mme" in hostapi_name else 2
                pattern_rank = 0 if "cable output" in name_lower else 1
                fallback_candidates.append((hostapi_rank, pattern_rank, i, device))

            if fallback_candidates:
                fallback_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
                _, _, device_id, device = fallback_candidates[0]
                self.channel_count = max(1, min(2, int(device.get("max_input_channels", 1))))
                self.sample_rate = int(device.get("default_samplerate") or self.sample_rate or 48000)
                self.extra_settings = None
                self.capture_mode = "virtual-cable"
                logger.info(
                    f"Found Windows system-audio input device: {device['name']} (ID: {device_id}, host API: "
                    f"{hostapi_names.get(device.get('hostapi'), 'unknown')})"
                )
                return device_id

            logger.warning("No WASAPI loopback or virtual system-audio input device available for capture")
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
            if self.monitor:
                self.monitor.enqueue_chunk(chunk)
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
                logger.error("Use a supported loopback setup or route playback through VB-Cable / Stereo Mix")
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

            if sys.platform.startswith("win") and self.capture_mode == "virtual-cable":
                self.monitor = SystemAudioMonitor(sample_rate=self.sample_rate)
                if not self.monitor.start():
                    self.monitor = None

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
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
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
        self.device = device if device is not None else self._find_default_input_device()
        self.sample_rate = self._resolve_input_sample_rate(sample_rate)
        self.running = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.stream: Optional[Any] = None

    def _find_default_input_device(self) -> Optional[int]:
        """Prefer WASAPI input devices on Windows to keep host APIs aligned."""
        if not SOUNDDEVICE_AVAILABLE:
            return None

        try:
            if sys.platform.startswith("win"):
                hostapis = sd.query_hostapis()
                wasapi_hostapis = {
                    index for index, api in enumerate(hostapis)
                    if "wasapi" in api.get("name", "").lower()
                }
                for hostapi_index in wasapi_hostapis:
                    default_input = hostapis[hostapi_index].get("default_input_device")
                    if isinstance(default_input, int) and default_input >= 0:
                        return default_input

            default_devices = sd.default.device
            if isinstance(default_devices, (list, tuple)) and len(default_devices) > 0:
                default_input = default_devices[0]
                if isinstance(default_input, int) and default_input >= 0:
                    return default_input
        except Exception as e:
            logger.warning(f"Could not determine default microphone device: {e}")

        return None

    def _resolve_input_sample_rate(self, requested_sample_rate: int) -> int:
        """Use the device's preferred sample rate when Windows rejects 16 kHz input."""
        if not SOUNDDEVICE_AVAILABLE or self.device is None:
            return requested_sample_rate

        try:
            device_info = sd.query_devices(self.device)
            default_sample_rate = int(device_info.get("default_samplerate") or requested_sample_rate)

            if sys.platform.startswith("win"):
                return default_sample_rate

            return requested_sample_rate
        except Exception as e:
            logger.warning(f"Could not determine microphone sample rate: {e}")
            return requested_sample_rate

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
            logger.info(
                f"Microphone capture started (device={self.device}, sample_rate={self.sample_rate})"
            )
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



def _get_sounddevice_name(device_id: Optional[int]) -> Optional[str]:
    """Return a readable sounddevice name for status reporting."""
    if not SOUNDDEVICE_AVAILABLE or device_id is None:
        return None

    try:
        device = sd.query_devices(device_id)
        return device.get("name")
    except Exception:
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
        self.mic_device = mic_device
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
        self.active_capture_mode = "none"

    def get_capture_mode(self) -> str:
        """Return the audio sources that are currently active."""
        system_active = self.system_capture is not None and self.system_capture.running
        mic_active = self.mic_capture is not None and self.mic_capture.running

        if system_active and mic_active:
            return "system+microphone"
        if system_active:
            return "system-only"
        if mic_active:
            return "microphone-only"
        return "none"

    def get_capture_status(self) -> Dict[str, Any]:
        """Return details about the currently active capture streams."""
        system_status = {
            "active": False,
            "device_id": None,
            "device_name": None,
            "capture_mode": None,
            "sample_rate": None,
            "channels": None,
        }
        mic_status = {
            "active": False,
            "device_id": None,
            "device_name": None,
            "sample_rate": None,
        }

        if self.system_capture:
            system_status.update({
                "active": self.system_capture.running,
                "device_id": self.system_capture.device_id,
                "device_name": _get_sounddevice_name(self.system_capture.device_id),
                "capture_mode": self.system_capture.capture_mode,
                "sample_rate": self.system_capture.sample_rate,
                "channels": self.system_capture.channel_count,
            })

        if self.mic_capture:
            mic_status.update({
                "active": self.mic_capture.running,
                "device_id": self.mic_capture.device,
                "device_name": _get_sounddevice_name(self.mic_capture.device),
                "sample_rate": self.mic_capture.sample_rate,
            })

        return {
            "mode": self.get_capture_mode(),
            "system_audio": system_status,
            "microphone": mic_status,
        }

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
            self.mic_capture = MicrophoneCapture(device=self.mic_device)
            if not self.mic_capture.start():
                logger.warning("Microphone capture not available")
                self.mic_capture = None
            else:
                self.mic_buffer = AudioBuffer(sample_rate=self.mic_capture.sample_rate)

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

        self.active_capture_mode = self.get_capture_mode()
        logger.info(f"Real-time transcription started ({self.active_capture_mode})")
        return True

    def stop(self) -> List[TranscriptSegment]:
        """Stop transcription and return all segments."""
        shutdown_time = max(0.0, time.time() - self.start_time) if self.start_time else 0.0
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

        # Drain any chunks queued after the last audio-loop iteration, then flush both
        # buffers once so the final few seconds of a meeting are not dropped on shutdown.
        self._drain_pending_capture_audio()
        self._flush_pending_buffers(reference_time=shutdown_time)

        self.active_capture_mode = "none"
        logger.info("Real-time transcription stopped")

        with self.segments_lock:
            return list(self.segments)

    def _drain_pending_capture_audio(self) -> None:
        """Move any queued capture chunks into the transcription buffers."""
        capture_pairs = (
            (self.system_capture, self.system_buffer),
            (self.mic_capture, self.mic_buffer),
        )

        for capture, buffer in capture_pairs:
            if not capture or not buffer or not hasattr(capture, "audio_queue"):
                continue

            while True:
                try:
                    chunk = capture.audio_queue.get_nowait()
                except queue.Empty:
                    break

                if chunk is not None:
                    buffer.add_chunk(chunk)

    def _flush_pending_buffers(self, reference_time: Optional[float] = None) -> None:
        """Transcribe any remaining buffered audio during shutdown."""
        pending_buffers = (
            (self.system_buffer, "system", "Other"),
            (self.mic_buffer, "microphone", "You"),
        )

        for buffer, source, speaker in pending_buffers:
            if not buffer:
                continue

            buffered_duration = buffer.duration()
            if buffered_duration < 0.5:
                continue

            logger.info(
                f"Flushing pending {source} audio during shutdown "
                f"({buffered_duration:.2f}s buffered)"
            )
            self._transcribe_buffer(
                buffer,
                source=source,
                speaker=speaker,
                reference_time=reference_time,
            )

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
            try:
                current_time = time.time() - self.start_time

                # Wait for enough audio to accumulate
                if current_time - self.last_transcription_time < self.chunk_duration:
                    time.sleep(0.1)
                    continue

                # Process system audio first on Windows so speaker audio is not starved by a
                # silent microphone buffer on slower CPU inference.
                if self.system_buffer and self.system_buffer.duration() >= self.chunk_duration:
                    self._transcribe_buffer(
                        self.system_buffer,
                        source="system",
                        speaker="Other",
                        reference_time=current_time,
                    )

                # Transcribe microphone audio for "You"
                if self.mic_buffer and self.mic_buffer.duration() >= self.chunk_duration:
                    self._transcribe_buffer(
                        self.mic_buffer,
                        source="microphone",
                        speaker="You",
                        reference_time=current_time,
                    )

                self.last_transcription_time = current_time
            except Exception as e:
                logger.error(f"Transcription loop error: {e}")
                time.sleep(0.2)

    def _transcribe_buffer(
        self,
        buffer: AudioBuffer,
        source: str,
        speaker: str,
        reference_time: Optional[float] = None,
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

        # Skip near-silent buffers before invoking Whisper. This is especially important on
        # Windows where the microphone buffer may fill with silence and otherwise block the
        # system-audio transcription pass on slower CPU inference.
        if NUMPY_AVAILABLE:
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            mean_abs = float(np.mean(np.abs(audio))) if len(audio) else 0.0
            if peak < 0.015 and mean_abs < 0.002:
                logger.info(
                    f"Skipping near-silent {source} buffer "
                    f"(peak={peak:.4f}, mean_abs={mean_abs:.4f})"
                )
                buffer.clear()
                return

        try:
            current_time = reference_time if reference_time is not None else time.time() - self.start_time

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


def detect_capture_capabilities() -> Dict[str, Any]:
    """
    Inspect configured audio devices without loading Whisper or starting transcription.

    This reports whether the app can attempt microphone and system-audio capture.
    It cannot guarantee speech will be present; virtual-cable setups still need the
    meeting app or system output routed into the cable.
    """
    capabilities: Dict[str, Any] = {
        "sounddevice": SOUNDDEVICE_AVAILABLE,
        "system_audio": {
            "available": False,
            "device_id": None,
            "device_name": None,
            "capture_mode": None,
            "sample_rate": None,
            "channels": None,
            "note": None,
        },
        "microphone": {
            "available": False,
            "device_id": None,
            "device_name": None,
            "sample_rate": None,
            "note": None,
        },
    }

    if not SOUNDDEVICE_AVAILABLE:
        capabilities["system_audio"]["note"] = "sounddevice is not installed"
        capabilities["microphone"]["note"] = "sounddevice is not installed"
        return capabilities

    try:
        system_capture = SystemAudioCapture()
        system_available = system_capture.device_id is not None
        system_note = None
        if sys.platform.startswith("win") and system_capture.capture_mode == "virtual-cable":
            system_note = "route meeting audio to VB-Cable/CABLE Input for speaker capture"
        elif not system_available:
            system_note = "no system-audio capture device found"

        capabilities["system_audio"].update({
            "available": system_available,
            "device_id": system_capture.device_id,
            "device_name": _get_sounddevice_name(system_capture.device_id),
            "capture_mode": system_capture.capture_mode,
            "sample_rate": system_capture.sample_rate,
            "channels": system_capture.channel_count,
            "note": system_note,
        })
    except Exception as e:
        capabilities["system_audio"]["note"] = str(e)

    try:
        mic_capture = MicrophoneCapture()
        mic_available = mic_capture.device is not None
        capabilities["microphone"].update({
            "available": mic_available,
            "device_id": mic_capture.device,
            "device_name": _get_sounddevice_name(mic_capture.device),
            "sample_rate": mic_capture.sample_rate,
            "note": None if mic_available else "no microphone input device found",
        })
    except Exception as e:
        capabilities["microphone"]["note"] = str(e)

    return capabilities



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
