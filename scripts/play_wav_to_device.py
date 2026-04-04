#!/usr/bin/env python3
"""Play a WAV file to a specific sounddevice output device."""

import argparse
import wave

import numpy as np
import sounddevice as sd


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a WAV file to a specific output device.")
    parser.add_argument("wav_path", help="Path to the WAV file to play")
    parser.add_argument("--device", type=int, required=True, help="sounddevice output device index")
    args = parser.parse_args()

    with wave.open(args.wav_path, "rb") as wav_file:
        data = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        audio = audio.reshape(-1, wav_file.getnchannels())
        sd.play(audio, samplerate=wav_file.getframerate(), device=args.device)
        sd.wait()


if __name__ == "__main__":
    main()
