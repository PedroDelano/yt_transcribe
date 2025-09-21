import argparse
import os
import uuid

from core import download_audio_as_wav_bytes, logger, transcribe_wav


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a WAV file using OpenAI Whisper"
    )
    parser.add_argument(
        "url",
        help="Path to the youtube URL to transcribe (or - to read from stdin bytes)",
    )
    parser.add_argument(
        "--output",
        default=f"{uuid.uuid4()}.txt",
        help="Path to save the transcription text file. Default: %(default)s",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size (tiny, base, small, medium, large-v3). Default: %(default)s",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code to force (e.g. 'en', 'pt'). Default: auto-detect.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature. Default: %(default)s",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Force FP32 even on GPU (disable half precision)",
    )

    args = parser.parse_args()

    assert args.url, "You must provide a YouTube URL or - for stdin"
    assert args.model in [
        "tiny",
        "base",
        "small",
        "medium",
        "large-v3",
    ], "Invalid model name"
    assert 0.0 <= args.temperature <= 1.0, "Temperature must be between 0.0 and 1.0"
    assert os.path.splitext(args.output)[1] == ".txt", (
        "Output file must have .txt extension"
    )
    assert not os.path.exists(args.output), f"Output file {args.output} already exists"

    audio_bytes = download_audio_as_wav_bytes(args.url)

    text = transcribe_wav(
        audio_bytes,
        model_name=args.model,
        language=args.language,
        temperature=args.temperature,
        fp16=(not args.no_fp16),
    )

    # add source to text
    text = f"Source: {args.url}\n\n{text}"

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    logger.info(f"Transcription saved to {args.output}")
