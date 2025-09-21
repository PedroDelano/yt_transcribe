import io
import logging
import os
import subprocess
import sys
import tempfile
from typing import Optional, Union

import torch
import whisper
import yt_dlp

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def transcribe_wav(
    wav: Union[str, bytes, io.BufferedIOBase],
    model_name: str = "base",  # "tiny" | "base" | "small" | "medium" | "large-v3"
    language: Optional[str] = None,  # e.g. "en", "pt"; None = auto-detect
    temperature: float = 0.0,  # higher -> more creative, lower -> more deterministic
    fp16: Optional[bool] = None,  # None = auto (True if CUDA available)
) -> str:
    """
    Transcribe a WAV with Whisper and return the text.

    Args:
        wav: Path to .wav file OR raw bytes/IO object containing a WAV.
        model_name: Whisper checkpoint size.
        language: ISO-639-1 code or None to auto-detect.
        temperature: Decoding temperature.
        fp16: Use half precision when on GPU; defaults to True if CUDA is available.

    Returns:
        str: Transcribed text.
    """

    logger.info(f"Loading Whisper model '{model_name}'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp16 is None:
        fp16 = device == "cuda"

    model = whisper.load_model(model_name, device=device)

    logger.info(
        f"Transcribing audio with language={language} temperature={temperature} fp16={fp16}"
    )

    # Normalize input to a file path Whisper can read
    tmp_path = None
    if isinstance(wav, (bytes, bytearray)):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(wav)
        tmp.flush()
        tmp.close()
        tmp_path = tmp.name
        wav_path = tmp_path
    elif hasattr(wav, "read"):  # file-like object
        data = wav.read()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(data)
        tmp.flush()
        tmp.close()
        tmp_path = tmp.name
        wav_path = tmp_path
    else:
        wav_path = os.fspath(wav)  # assume str/PathLike

    try:
        # Fast path: use model.transcribe which handles loading & segmenting
        result = model.transcribe(
            wav_path,
            language=language,  # None => auto-detect
            temperature=temperature,
            fp16=fp16,
            verbose=False,
            condition_on_previous_text=True,
            word_timestamps=False,  # set True if you want word-level timestamps
        )
        return result.get("text", "").strip()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def download_audio_as_wav_bytes(url: str) -> bytes:
    """
    Download audio from a YouTube URL and return it as WAV bytes.

    Parameters
    ----------
    url : str
        The YouTube video URL.

    Returns
    -------
    bytes
        WAV audio data.
    """

    logger.info(f"Downloading audio from {url}")

    # Create temp files for intermediate download and wav output
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.%(ext)s")
        output_path = os.path.join(tmpdir, "output.wav")

        # 1. Download bestaudio with yt-dlp
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": input_path,
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_ext = ydl.prepare_filename(info)

        logger.info(f"Downloaded audio to {downloaded_ext}")
        logger.info("Converting to WAV format")

        # 2. Convert to WAV using ffmpeg
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                downloaded_ext,
                "-vn",  # no video
                "-acodec",
                "pcm_s16le",  # raw PCM 16-bit
                "-ar",
                "44100",  # sample rate
                "-ac",
                "2",  # stereo
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 3. Read WAV file into bytes
        with open(output_path, "rb") as f:
            wav_bytes = f.read()

        logger.info("Conversion to WAV completed")

    return wav_bytes
