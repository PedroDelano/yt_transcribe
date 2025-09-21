# yt_transcribe

A simple command-line tool that downloads audio from a YouTube video and transcribes it to text using [OpenAI Whisper](https://github.com/openai/whisper) .

## Features

* 📥 **Download audio** from YouTube using `yt-dlp`.
* 🎙️ **Convert to WAV** with `ffmpeg`.
* 📝 **Transcribe audio** to text using Whisper (tiny, base, small, medium, large-v3 models).
* 🌐 **Language auto-detection** or specify language code.
* ⚡ **CUDA/FP16 support** automatically enabled if GPU is available.

## Installation

You need Python ≥3.9.

```bash
git clone git@github.com:PedroDelano/yt_transcribe.git
cd yt_transcribe
pip install -e .
```

This will install the dependencies listed in `pyproject.toml`:

* `ffmpeg>=1.4`
* `yt-dlp>=2025.9.5`
* `openai-whisper>=20250625`
* `torch==2.5.0`
* `transformers>=4.56.2`

Make sure `ffmpeg` is also installed on your system.

## Usage

Run from the command line:

```bash
python -m yt_transcribe.main \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --model base \
  --language en \
  --temperature 0.0
```

Options:

| Flag            | Description                                                                          |
| --------------- | ------------------------------------------------------------------------------------ |
| `url`           | YouTube URL to transcribe (required).                                                |
| `--model`       | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`). Default: `base`. |
| `--language`    | Force a language code (e.g. `en`, `pt`). Default: auto-detect.                       |
| `--temperature` | Decoding temperature. Default: `0.0`.                                                |
| `--no-fp16`     | Disable half-precision even on GPU.                                                  |

The transcribed text will be saved under the `output/` folder with the same base name as the URL file.

Example:

```bash
python -m yt_transcribe.main "https://youtu.be/xyz" --model small --language pt
```

This will:

1. Download the audio as best available quality.
2. Convert it to 44.1 kHz stereo WAV using `ffmpeg`.
3. Transcribe it with Whisper (small model) in Portuguese.
4. Save a `.txt` file in `output/`.

## Programmatic Usage

You can also import the functions in Python:

```python
from yt_transcribe.main import download_audio_as_wav_bytes, transcribe_wav

audio_bytes = download_audio_as_wav_bytes("https://youtu.be/xyz")
text = transcribe_wav(audio_bytes, model_name="base", language="en")
print(text)
```

## Contributing

Pull requests are welcome. Please run `ruff`/`black` before committing.

## License

This project is licensed under the MIT License — you’re free to use, modify, and distribute it, even commercially, with no restrictions.
