#!/usr/bin/env python3
"""
Simple Audio Recorder & Transcriber for Electron App

Backend script that handles:
1. Recording system/microphone audio
2. Transcribing with Whisper  
3. Saving everything locally

Usage (called by Electron):
    python simple_recorder.py start "Meeting Name"
    python simple_recorder.py stop  
    python simple_recorder.py process recording.wav --name "Session"
    python simple_recorder.py status
"""

import click
import asyncio
import logging
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import modules with graceful fallback for missing dependencies
try:
    from src.audio_recorder import AudioRecorder
except ImportError:
    AudioRecorder = None

try:
    from src.transcriber import WhisperTranscriber
except ImportError:
    WhisperTranscriber = None

try:
    from src.realtime_transcriber import RealtimeTranscriber, create_realtime_transcriber, TranscriptSegment
except ImportError:
    RealtimeTranscriber = None
    create_realtime_transcriber = None
    TranscriptSegment = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Windows terminals and spawned subprocess pipes may default to a legacy code page.
# Force UTF-8 so status output with symbols does not crash the CLI or Electron bridge.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_app_data_dir() -> Path:
    """Return a per-user application data directory for the current platform."""
    override = os.environ.get("STENOAI_APP_DATA_DIR")
    if override:
        return Path(override)

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "stenoai"
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "stenoai"
        return Path.home() / "AppData" / "Roaming" / "stenoai"
    return Path.home() / ".config" / "stenoai"

class SimpleRecorder:
    """Simple audio recorder and transcriber."""
    
    def __init__(self):
        # Only initialize if dependencies are available
        self.audio_recorder = AudioRecorder() if AudioRecorder else None
        
        # Only initialize the transcriber when needed to save memory
        self.transcriber = None
        
        # Directories - use user data folder for packaged app distribution
        current_path = Path(__file__).parent
        if os.environ.get("STENOAI_APP_DATA_DIR") or "StenoAI.app" in str(current_path) or "Applications" in str(current_path):
            app_support = get_app_data_dir()
            self.recordings_dir = app_support / "recordings"
            self.transcripts_dir = app_support / "transcripts" 
            self.output_dir = app_support / "output"
        else:
            # Development: Use project relative paths
            self.recordings_dir = Path("recordings")
            self.transcripts_dir = Path("transcripts") 
            self.output_dir = Path("output")
        
        # Create directories (including parent directories)
        for dir_path in [self.recordings_dir, self.transcripts_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # State file
        self.state_file = Path("recorder_state.json")
        self.stop_request_file = Path("recorder_stop.request")
        
        # Global AudioRecorder instance to maintain state across CLI calls
        self.persistent_recorder = None
        
    def get_state(self) -> dict:
        """Get current recorder state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"recording": False, "current_file": None, "session_name": None}
    
    def save_state(self, state: dict):
        """Save recorder state."""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def clear_stop_request(self):
        """Clear any pending stop request file."""
        if self.stop_request_file.exists():
            try:
                self.stop_request_file.unlink()
            except Exception:
                pass

    def request_stop(self):
        """Request a running recorder loop to stop."""
        self.stop_request_file.write_text(datetime.now().isoformat())

    def should_stop(self) -> bool:
        """Check whether a stop was requested."""
        return self.stop_request_file.exists()
    
    def start_recording(self, session_name: str = "Recording") -> str:
        """Start recording audio."""
        state = self.get_state()
        if state.get("recording"):
            raise Exception(f"Already recording: {state.get('current_file', 'unknown file')}")
        self.clear_stop_request()
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in session_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{timestamp}_{safe_name}.wav"
        
        recording_path = self.recordings_dir / filename
        
        print(f"🎤 Starting recording: {session_name}")
        print(f"📁 File: {recording_path}")
        
        # Start recording
        self.audio_recorder.start_recording()
        
        # Update state
        new_state = {
            "recording": True,
            "current_file": str(recording_path), 
            "session_name": session_name,
            "start_time": datetime.now().isoformat()
        }
        self.save_state(new_state)
        
        return str(recording_path)
    
    def stop_recording(self) -> Optional[str]:
        """Stop current recording."""
        state = self.get_state()
        if not state.get("recording"):
            print("⚠️ No active recording to stop")
            return None
        
        print("🔴 Stopping recording")
        
        # Stop recording
        self.audio_recorder.stop_recording()
        
        # Wait a moment for recording to fully stop
        import time
        time.sleep(0.5)
        
        # Get the planned file path from state
        recording_path = state.get("current_file")
        if not recording_path:
            print("⚠️ No recording file path found in state")
            # Try to create a default path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recording_path = str(self.recordings_dir / f"{timestamp}_recording.wav")
        
        # Save the recording to the planned file
        from pathlib import Path
        if self.audio_recorder.save_recording(Path(recording_path)):
            print(f"✅ Recording saved: {recording_path}")
        else:
            print("❌ Failed to save recording")
            recording_path = None
        
        # Update state (always clear recording state)
        new_state = {
            "recording": False,
            "current_file": None,
            "session_name": None,
            "stop_time": datetime.now().isoformat()
        }
        if recording_path:
            new_state["last_recording"] = recording_path
        
        self.save_state(new_state)
        return recording_path
    
    async def transcribe_audio(self, audio_file: str, session_name: str = "Recording") -> dict:
        """Transcribe audio file."""
        audio_path = Path(audio_file)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        print(f"📝 Transcribing: {audio_path.name}")
        
        # Initialize transcriber only when needed
        if self.transcriber is None:
            self.transcriber = WhisperTranscriber()
        
        # Transcribe (pass Path object, not string)
        transcript_result = self.transcriber.transcribe_audio(audio_path)
        
        # Debug: Check what transcript_result actually is
        print(f"DEBUG: transcript_result type: {type(transcript_result)}")
        print(f"DEBUG: transcript_result: {transcript_result}")
        
        # Handle different return types
        if hasattr(transcript_result, 'text'):
            transcript_text = transcript_result.text
        elif isinstance(transcript_result, str):
            transcript_text = transcript_result
        else:
            transcript_text = str(transcript_result)
        
        # Save transcript
        transcript_path = self.transcripts_dir / f"{audio_path.stem}_transcript.txt"
        transcript_content = f"""Session: {session_name}
File: {audio_path.name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}

{transcript_text}
"""
        
        with open(transcript_path, 'w') as f:
            f.write(transcript_content)
        
        print(f"📄 Transcript saved: {transcript_path}")
        
        return {
            "audio_file": str(audio_path),
            "transcript_file": str(transcript_path), 
            "transcript_text": transcript_text,
            "session_name": session_name
        }
    
    async def process_recording(self, audio_file: str, session_name: str = "Recording") -> dict:
        """Complete processing: transcribe and save transcript data."""
        print(f"🔄 Processing recording: {audio_file}")
        
        # If no audio file provided, use the last recording
        if not audio_file:
            state = self.get_state()
            audio_file = state.get("last_recording")
            if not audio_file:
                raise Exception("No audio file specified and no recent recording found")
        
        # Ensure we have a proper path
        audio_file = str(audio_file)  # Convert to string if it's a Path object
        audio_path = Path(audio_file)
        
        # Calculate actual recording duration from file
        duration_minutes = 10  # Default fallback
        try:
            import wave
            with wave.open(str(audio_path), 'rb') as wav_file:
                frame_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()
                duration_seconds = num_frames / frame_rate
                if duration_seconds < 60:
                    duration_display = f"{int(duration_seconds)}s"
                    duration_minutes = 0  # Store as 0 for sub-minute recordings
                else:
                    duration_minutes = int(duration_seconds / 60)
                    duration_display = f"{duration_minutes}m"
                print(f"📏 Audio duration: {duration_seconds:.1f} seconds ({duration_display})")
        except Exception as e:
            print(f"⚠️ Could not determine audio duration: {e}")
            # Try to get duration from state file timestamps
            try:
                state = self.get_state()
                start_time = state.get("start_time")
                stop_time = state.get("stop_time")
                if start_time and stop_time:
                    from dateutil.parser import parse
                    start_dt = parse(start_time)
                    stop_dt = parse(stop_time)
                    duration_seconds = (stop_dt - start_dt).total_seconds()
                    duration_minutes = max(1, int(duration_seconds / 60))
                    print(f"📏 Duration from timestamps: {duration_seconds:.1f} seconds ({duration_minutes} minutes)")
            except Exception:
                pass
        
        # Step 1: Transcribe
        transcript_data = await self.transcribe_audio(audio_file, session_name)
        
        # Step 2: Save transcript record
        meeting_path = self.output_dir / f"{audio_path.stem}_meeting.json"
        transcript_text = transcript_data["transcript_text"]
        
        complete_data = {
            "session_info": {
                "name": session_name,
                "audio_file": str(audio_path),
                "transcript_file": transcript_data["transcript_file"],
                "meeting_file": str(meeting_path),
                "summary_file": str(meeting_path),
                "processed_at": datetime.now().isoformat(),
                "duration_seconds": int(duration_seconds) if 'duration_seconds' in locals() else None,
                "duration_minutes": duration_minutes,
                "mode": "transcription"
            },
            "transcript_preview": " ".join(transcript_text.split())[:240],
            "transcript": transcript_text
        }
        
        with open(meeting_path, 'w') as f:
            json.dump(complete_data, f, indent=2)
        
        print(f"✅ Transcript record saved: {meeting_path}")
        
        # Clean up WAV file after successful processing
        try:
            audio_path.unlink()
            print(f"🗑️ Cleaned up audio file: {audio_path}")
        except Exception as e:
            print(f"⚠️ Could not delete audio file: {e}")
        
        # Clear any recording state after successful processing
        state_file = Path("recorder_state.json")
        if state_file.exists():
            try:
                state_file.unlink()
                print(f"🧹 Cleared recording state")
            except Exception as e:
                print(f"⚠️ Could not clear state: {e}")
        
        print("📋 Processing complete - transcript available in list")
        
        return complete_data


# CLI Commands for Electron
@click.group()
def cli():
    """Simple Audio Recorder & Transcriber Backend"""
    pass


@cli.command()
@click.argument('session_name', default='Recording')
def start(session_name):
    """Start recording audio (stop with Ctrl+C to auto-process)"""
    import signal
    import time
    
    recorder = SimpleRecorder()
    recording_path = None
    recording_started = False
    processing_started = False
    
    def signal_handler(signum, frame):
        """Handle SIGTERM/SIGINT gracefully by stopping recording and processing"""
        nonlocal processing_started
        
        # Different handling for different signals
        signal_name = "SIGINT" if signum == 2 else f"SIGTERM" if signum == 15 else f"Signal {signum}"
        print(f"\n🛑 Received {signal_name} - stopping recording and processing...")
        
        # Prevent double processing if multiple signals received
        if processing_started:
            print("⚠️ Processing already started - please wait for completion...")
            if signum == 15:  # SIGTERM - ignore it during processing
                print("🔄 Ignoring SIGTERM during transcription")
                return
            exit(0)
            
        if recording_started and recorder:
            processing_started = True
            try:
                final_path = recorder.stop_recording()
                if final_path:
                    print(f"✅ Recording saved: {final_path}")
                    
                    # Check file size
                    from pathlib import Path
                    file_size = Path(final_path).stat().st_size
                    print(f"📏 File size: {file_size / 1024:.1f} KB")
                    
                    if file_size >= 1000:  # At least 1KB of audio data
                        print("🔄 Starting transcription pipeline...")
                        
                        # Process recording with proper async handling
                        try:
                            import asyncio
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            print("📝 Transcribing...")
                            result = loop.run_until_complete(recorder.process_recording(final_path, session_name))
                            
                            print("✅ Complete processing finished!")
                            print(f"📄 Transcript: {result['session_info']['transcript_file']}")
                            print(f"📋 Record: {result['session_info']['meeting_file']}")
                            print(f"📊 Meeting: {result['session_info']['name']}")
                            
                        except Exception as e:
                            print(f"❌ Processing pipeline failed: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print("⚠️ Recording too short - skipping processing")
                else:
                    print("❌ No recording data to save")
            except Exception as e:
                print(f"❌ Error during signal handling: {e}")
                import traceback
                traceback.print_exc()
        
        print("🏁 Recording session ended")
        exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        recorder.clear_stop_request()
        recording_path = recorder.start_recording(session_name)
        recording_started = True
        print(f"🎤 Recording '{session_name}' - Press Ctrl+C to stop and process")
        print(f"📁 File: {recording_path}")
        print("📢 Speak now...")
        
        # Wait indefinitely until interrupted
        while True:
            if recorder.should_stop():
                signal_handler(signal.SIGTERM, None)
            time.sleep(1)
            
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)


@cli.command()
def stop():
    """Stop current recording and trigger processing"""
    import subprocess
    import signal
    import os
    import time

    recorder = SimpleRecorder()
    recorder.request_stop()
    
    # First check if there's a recording process running
    try:
        if sys.platform.startswith("win"):
            print("🛑 Stop requested - waiting for background recorder to finish processing")
            print("✅ Stop request recorded for any active background recorder")
            return

        # Find running start processes
        result = subprocess.run(
            ['pgrep', '-f', 'simple_recorder.py start'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"🔍 Found {len(pids)} recording process(es)")
            
            for pid in pids:
                if pid.strip():
                    try:
                        pid_int = int(pid.strip())
                        print(f"🛑 Sending SIGINT to recording process (PID: {pid_int})")
                        os.kill(pid_int, signal.SIGINT)
                        
                        print(f"✅ Stop signal sent to process {pid_int}")
                        print("🔄 Recording will stop and transcription will begin automatically")
                        print(f"💡 Processing may take a few minutes - check output files when complete")
                            
                    except (ValueError, ProcessLookupError) as e:
                        print(f"⚠️ Could not signal process {pid}: {e}")
            
            print("✅ Stop signal sent - recording will be transcribed automatically")
            
        else:
            # Cross-platform fallback for background `record` mode uses the stop request file.
            print("🔍 No start process found, checking recording state...")
            state = recorder.get_state()
            
            if state.get("recording"):
                print("⚠️ Recording state shows active but no process found")
                print("🔧 Clearing stuck state...")
                recorder.save_state({
                    "recording": False,
                    "current_file": None,
                    "session_name": None
                })
                print("✅ State cleared")
            else:
                print("ℹ️ No active recording found")
                print("✅ Stop request recorded for any active background recorder")
                
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)


@cli.command()
@click.argument('audio_file', default='')
@click.option('--name', '-n', default='Recording', help='Session name for the recording')
def process(audio_file, name):
    """Process audio file: transcribe and save transcript data"""
    
    async def run_process():
        recorder = SimpleRecorder()
        
        try:
            result = await recorder.process_recording(audio_file, name)
            
            print("SUCCESS: Processing complete!")
            print(f"Transcript: {result['session_info']['transcript_file']}")
            print(f"Record: {result['session_info']['meeting_file']}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            exit(1)
    
    asyncio.run(run_process())


@cli.command()
def status():
    """Show recorder status"""
    recorder = SimpleRecorder()
    state = recorder.get_state()
    
    print("🎙️ Steno Recorder Status")
    print("=" * 25)
    
    if state.get("recording"):
        print("STATUS: RECORDING")
        print(f"Session: {state.get('session_name')}")
        print(f"File: {state.get('current_file')}")
        print(f"Started: {state.get('start_time')}")
    else:
        print("STATUS: READY")
    
    # Show recent recordings
    recordings = list(recorder.recordings_dir.glob("*.wav"))
    if recordings:
        recent = sorted(recordings, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
        print(f"\nRecent recordings ({len(recordings)} total):")
        for recording in recent:
            size_mb = recording.stat().st_size / (1024 * 1024)
            print(f"  • {recording.name} ({size_mb:.1f}MB)")


@cli.command()
@click.argument('duration', type=int, default=10)
@click.argument('session_name', default='Recording')
def record(duration, session_name):
    """Record audio for specified duration with real-time transcription (system audio + mic)"""
    import signal

    print(f"🎤 Recording {duration} seconds of audio for '{session_name}'...")

    # Check if RealtimeTranscriber is available
    if RealtimeTranscriber is None:
        print("❌ RealtimeTranscriber not available - falling back to basic recording")
        print("   Install faster-whisper: pip install faster-whisper")
        # Fall back to basic recording
        _record_basic(duration, session_name)
        return

    recorder = SimpleRecorder()
    transcriber = None
    live_logger = None
    recording_started = False
    start_time = None
    capture_mode = None

    # Generate file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = recorder.transcripts_dir / f"{timestamp}_{session_name}_transcript.txt"
    meeting_path = recorder.output_dir / f"{timestamp}_{session_name}_meeting.json"

    def on_transcript_segment(segment):
        """Callback for real-time transcript updates"""
        print(f"  [{segment.speaker}]: {segment.text}")

    def process_and_save(transcript_text: str, segments: list, duration_seconds: float):
        """Process transcript and save results"""
        print("🔄 Processing transcript...")

        # Save transcript file
        with open(transcript_path, 'w') as f:
            f.write(f"# Meeting Transcript: {session_name}\n")
            f.write(f"# Date: {datetime.now().isoformat()}\n")
            f.write(f"# Duration: {duration_seconds:.1f} seconds\n")
            f.write("# " + "=" * 50 + "\n\n")

            for seg in sorted(segments, key=lambda s: s.start_time):
                speaker = seg.speaker or "Unknown"
                f.write(f"[{seg.start_time:.2f}s] [{speaker}]: {seg.text}\n")

        print(f"📄 Transcript saved: {transcript_path}")

        # Get plain text transcript for storage
        plain_transcript = "\n".join([
            f"[{seg.speaker or 'Unknown'}]: {seg.text}"
            for seg in sorted(segments, key=lambda s: s.start_time)
        ])

        if not plain_transcript.strip():
            plain_transcript = "No speech detected in audio"

        result = {
            "session_info": {
                "name": session_name,
                "audio_file": None,  # No audio file saved in real-time mode
                "transcript_file": str(transcript_path),
                "meeting_file": str(meeting_path),
                "summary_file": str(meeting_path),
                "processed_at": datetime.now().isoformat(),
                "duration_seconds": int(duration_seconds),
                "duration_minutes": max(1, int(duration_seconds / 60)),
                "mode": "transcription"
            },
            "transcript_preview": " ".join(plain_transcript.split())[:240],
            "transcript": plain_transcript
        }

        with open(meeting_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✅ Transcript record saved: {meeting_path}")

        # Clean up state
        if recorder.state_file.exists():
            recorder.state_file.unlink()
            print("🧹 Cleared recording state")

        print("📋 Processing complete - transcript available in list")
        return result

    def signal_handler(signum, frame):
        """Handle SIGTERM gracefully by stopping and processing"""
        nonlocal transcriber, start_time

        print(f"\n🛑 Received termination signal ({signum})")
        print("⏹️ Stopping recording and starting processing pipeline...")

        if transcriber and recording_started:
            try:
                # Calculate duration
                duration_seconds = time.time() - start_time if start_time else 0
                print(f"📏 Recording duration: {duration_seconds:.1f} seconds")

                # Stop transcriber and get segments
                segments = transcriber.stop()
                print(f"📝 Captured {len(segments)} transcript segments")

                # Get full transcript
                transcript_text = transcriber.get_full_transcript()

                if segments:
                    print("🔄 Starting transcription pipeline...")
                    result = process_and_save(transcript_text, segments, duration_seconds)

                    print("✅ Complete processing finished!")
                    print(f"📄 Transcript: {result['session_info']['transcript_file']}")
                    print(f"📋 Record: {result['session_info']['meeting_file']}")
                    print(f"📊 Meeting: {result['session_info']['name']}")
                else:
                    print("⚠️ No speech detected - saving empty transcript")
                    process_and_save("No speech detected", [], duration_seconds)

            except Exception as e:
                print(f"❌ Error during signal handling: {e}")
                import traceback
                traceback.print_exc()

        print("🏁 Recording session ended - process complete")
        print(f"\n🎉 Recording and processing completed for: {session_name}")
        exit(0)

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("🎤 Starting recording: " + session_name)
        recorder.clear_stop_request()

        # Save state for status command
        state = {
            "recording": True,
            "session_name": session_name,
            "start_time": datetime.now().isoformat(),
            "mode": "realtime"
        }
        recorder.save_state(state)

        # Create realtime transcriber with dual audio capture
        print("🔊 Initializing dual audio capture (system + microphone)...")
        transcriber, live_logger = create_realtime_transcriber(
            model_size="small",  # Use small model for better accuracy
            language="en",
            enable_system_audio=True,
            enable_microphone=True,
            callback=on_transcript_segment,
            session_name=session_name,
            enable_live_logging=True
        )

        # Start real-time transcription
        if not transcriber.start():
            print("❌ Failed to start real-time transcription")
            print("   System audio capture may require platform-specific audio routing")
            # Try microphone-only fallback
            print("🎤 Trying microphone-only mode...")
            transcriber, live_logger = create_realtime_transcriber(
                model_size="small",
                language="en",
                enable_system_audio=False,
                enable_microphone=True,
                callback=on_transcript_segment,
                session_name=session_name,
                enable_live_logging=True
            )
            if not transcriber.start():
                print("❌ Failed to start transcription - check audio devices")
                exit(1)
            capture_mode = "microphone-only"
        else:
            capture_mode = "system+microphone"

        recording_started = True
        start_time = time.time()

        print(f"CAPTURE_MODE: {capture_mode}")
        print(f"📁 Recording to: {transcript_path}")
        print("📢 Speak into your microphone now!")
        if capture_mode == "system+microphone":
            print("🔊 System audio will also be captured (meetings, videos, etc.)")
        else:
            print("🎤 Running in microphone-only mode")
        print("=" * 50)

        # For very long durations, wait indefinitely
        if duration > 86400:
            print("🔄 Recording indefinitely (until stopped)...")
            try:
                while True:
                    if recorder.should_stop():
                        signal_handler(signal.SIGTERM, None)
                    time.sleep(5)
            except KeyboardInterrupt:
                signal_handler(signal.SIGINT, None)
        else:
            # Count down for normal durations
            for i in range(duration, 0, -1):
                if recorder.should_stop():
                    signal_handler(signal.SIGTERM, None)
                print(f"   {i}...")
                time.sleep(1)

        # Normal completion (if not interrupted)
        signal_handler(signal.SIGTERM, None)

    except Exception as e:
        print(f"❌ Recording failed: {e}")
        import traceback
        traceback.print_exc()
        if transcriber:
            transcriber.stop()
        exit(1)


def _record_basic(duration, session_name):
    """Fallback basic recording without real-time transcription"""
    import signal

    recorder = SimpleRecorder()
    recording_started = False

    def signal_handler(signum, frame):
        print(f"\n🛑 Received termination signal ({signum})")
        if recording_started:
            try:
                final_path = recorder.stop_recording()
                if final_path:
                    print(f"✅ Recording saved: {final_path}")
                    file_size = Path(final_path).stat().st_size
                    if file_size >= 1000:
                        print("🔄 Starting transcription...")
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(recorder.process_recording(final_path, session_name))
                        print("✅ Complete processing finished!")
                        print(f"📄 Transcript: {result['session_info']['transcript_file']}")
                        print(f"📋 Record: {result['session_info']['meeting_file']}")
            except Exception as e:
                print(f"❌ Error: {e}")
        print("🏁 Recording session ended")
        exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        recorder.clear_stop_request()
        recording_path = recorder.start_recording(session_name)
        recording_started = True
        print("CAPTURE_MODE: microphone-only")
        print(f"📁 Recording to: {recording_path}")
        print("📢 Speak into your microphone now!")

        if duration > 86400:
            while True:
                if recorder.should_stop():
                    signal_handler(signal.SIGTERM, None)
                time.sleep(5)
        else:
            for i in range(duration, 0, -1):
                if recorder.should_stop():
                    signal_handler(signal.SIGTERM, None)
                print(f"   {i}...")
                time.sleep(1)

        signal_handler(signal.SIGTERM, None)

    except Exception as e:
        print(f"❌ Recording failed: {e}")
        exit(1)


@cli.command()
def test():
    """Quick system test - check components can initialize"""
    print("🧪 Quick system test...")
    
    try:
        # Test audio recording capability
        print("🎤 Testing audio recording...")
        recorder = SimpleRecorder()
        if not recorder.audio_recorder:
            print("❌ Audio recording not available")
            print("ERROR: Audio dependencies missing")
            return
        print("✅ Audio recording ready")
        
        # Test transcriber availability
        print("🗣️ Testing Whisper transcriber...")
        if not WhisperTranscriber:
            print("❌ Whisper transcriber not available")
            print("ERROR: Whisper not installed")
            return
            
        try:
            transcriber = WhisperTranscriber()
            print("✅ Whisper transcriber ready")
        except Exception as e:
            print(f"❌ Whisper initialization failed: {e}")
            print(f"ERROR: {e}")
            return
        
        print("🎉 System check passed!")
        print("SUCCESS: Recording and transcription components are working correctly")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        print(f"ERROR: {e}")
        return


@cli.command()
def list_meetings():
    """List all processed meetings - optimized for fast loading"""
    # Don't initialize SimpleRecorder - just get the output directory
    current_path = Path(__file__).parent
    if os.environ.get("STENOAI_APP_DATA_DIR") or "StenoAI.app" in str(current_path) or "Applications" in str(current_path):
        app_support = get_app_data_dir()
        output_dir = app_support / "output"
    else:
        # Development: Use project relative paths
        output_dir = Path("output")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load new transcript records and legacy summary records for compatibility.
    meeting_files = list(output_dir.glob("*_meeting.json")) + list(output_dir.glob("*_summary.json"))
    meetings = []
    
    # Sort by actual meeting date, with fallback to modification time
    def get_meeting_date(meeting_file):
        try:
            with open(meeting_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('session_info', {}).get('processed_at', '')
        except:
            # Fallback to file modification time if JSON read fails
            return meeting_file.stat().st_mtime
    
    meeting_files.sort(key=get_meeting_date, reverse=True)
    
    for meeting_file in meeting_files:
        try:
            with open(meeting_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Only include essential fields for faster loading
                essential_meeting = {
                    "session_info": data.get("session_info", {}),
                    "transcript_preview": data.get("transcript_preview", ""),
                    "transcript": data.get("transcript", "")
                }
                meetings.append(essential_meeting)
        except Exception as e:
            # Log warning but continue processing other files
            logger.warning(f"Failed to load {meeting_file}: {e}")
            continue
    
    # Output as compact JSON for Electron (no indentation for speed)
    print(json.dumps(meetings, separators=(',', ':')))

@cli.command()
def clear_state():
    """Clear recording state (useful for resetting stuck recordings)"""
    recorder = SimpleRecorder()
    
    if recorder.state_file.exists():
        recorder.state_file.unlink()
        print("SUCCESS: Recording state cleared")
    else:
        print("SUCCESS: No state file found - already clear")


@cli.command()
def setup_check():
    """Check system setup and dependencies"""
    import subprocess
    import sys
    import os
    
    print("🔧 StenoAI Setup Check")
    print("=" * 25)
    
    checks = []
    
    # Check Python version
    try:
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            checks.append(("✅ Python", f"{version.major}.{version.minor}.{version.micro}"))
        else:
            checks.append(("❌ Python", f"{version.major}.{version.minor}.{version.micro} (need 3.8+)"))
    except Exception as e:
        checks.append(("❌ Python", f"Error: {e}"))
    
    # Check required directories - use same logic as SimpleRecorder.__init__
    current_path = Path(__file__).parent
    if os.environ.get("STENOAI_APP_DATA_DIR") or "StenoAI.app" in str(current_path) or "Applications" in str(current_path):
        app_support = get_app_data_dir()
        base_dirs = {
            "recordings": app_support / "recordings",
            "transcripts": app_support / "transcripts", 
            "output": app_support / "output"
        }
    else:
        # Development: Use project relative paths
        base_dirs = {
            "recordings": Path("recordings"),
            "transcripts": Path("transcripts"), 
            "output": Path("output")
        }
    
    for dir_name, dir_path in base_dirs.items():
        if dir_path.exists():
            checks.append((f"✅ {dir_name}/", f"exists at {dir_path}"))
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            checks.append((f"✅ {dir_name}/", f"created at {dir_path}"))
    
    # Check ffmpeg
    try:
        ffmpeg_found = False
        possible_ffmpeg_paths = ['ffmpeg']
        if sys.platform == "darwin":
            possible_ffmpeg_paths.extend([
                '/opt/homebrew/bin/ffmpeg',
                '/usr/local/bin/ffmpeg',
                '/usr/bin/ffmpeg',
            ])
        elif sys.platform.startswith("win"):
            possible_ffmpeg_paths.extend([
                r'C:\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
            ])
        else:
            possible_ffmpeg_paths.extend([
                '/usr/local/bin/ffmpeg',
                '/usr/bin/ffmpeg',
            ])
        
        for path in possible_ffmpeg_paths:
            try:
                result = subprocess.run([path, '-version'], 
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    checks.append(("✅ ffmpeg", f"found at {path}"))
                    ffmpeg_found = True
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        if not ffmpeg_found:
            if sys.platform.startswith("win"):
                checks.append(("❌ ffmpeg", "not found - install via winget/choco/scoop or add to PATH"))
            else:
                checks.append(("❌ ffmpeg", "not found - install ffmpeg and add it to PATH"))
    except Exception as e:
        checks.append(("❌ ffmpeg", f"Error: {e}"))
    
    # Check Python dependencies
    try:
        import sounddevice
        checks.append(("✅ sounddevice", "audio recording"))
    except ImportError:
        checks.append(("❌ sounddevice", "pip install sounddevice"))
    
    try:
        import whisper
        checks.append(("✅ whisper", "speech transcription"))
    except ImportError:
        checks.append(("❌ whisper", "pip install openai-whisper"))
    
    # Print results
    all_good = True
    for status, detail in checks:
        print(f"{status:<20} {detail}")
        if status.startswith("❌"):
            all_good = False
    
    print("\n" + "=" * 25)
    if all_good:
        print("🎉 System check passed! Ready to record and transcribe.")
    else:
        print("⚠️ Setup incomplete. Please install missing dependencies.")
    
    return {"success": all_good, "checks": checks}


@cli.command()
def get_notifications():
    """Get the current notification preference"""
    from src.config import get_config

    config = get_config()
    enabled = config.get_notifications_enabled()

    result = {
        "notifications_enabled": enabled
    }

    print(json.dumps(result, indent=2))


@cli.command()
@click.argument('enabled', type=bool)
def set_notifications(enabled):
    """Set notification preference (True/False)"""
    from src.config import get_config

    config = get_config()
    success = config.set_notifications_enabled(enabled)

    if success:
        print(f"SUCCESS: Notifications {'enabled' if enabled else 'disabled'}")
        print(json.dumps({"success": True, "notifications_enabled": enabled}))
    else:
        print(f"ERROR: Failed to save notification preference")
        print(json.dumps({"success": False, "error": "Failed to save config"}))


if __name__ == '__main__':
    cli()
