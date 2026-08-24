# Watermarker Remover

Removes the Google Gemini AI sparkle watermark from the bottom-right corner of
videos using FFmpeg filters (`delogo` / `blur`) with automatic resolution
scaling.

## Requirements

- Python 3.8+
- FFmpeg and FFprobe installed and available in `PATH`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

## Usage

```bash
python remove_gemini_watermark.py input.mp4
python remove_gemini_watermark.py input.mp4 -o clean_output.mp4
python remove_gemini_watermark.py input.mp4 --preview
python remove_gemini_watermark.py input.mp4 --x 1130 --y 570 --w 56 --h 56
python remove_gemini_watermark.py ./videos_folder/ --batch
```

### Options

| Option | Description |
| --- | --- |
| `input` | Path to input video file or folder (used with `--batch`) |
| `-o, --output` | Output video file or output directory |
| `--batch` | Process all videos in the input directory |
| `--preview` | Generate a preview image with the watermark box marked |
| `--preview-time` | Timestamp (seconds) for preview frame (default: `2.0`) |
| `--method` | Removal method: `delogo` or `blur` (default: `delogo`) |
| `--crf` | H.264 CRF quality 0-51, lower is better (default: `18`) |
| `--preset` | x264 preset: `ultrafast`, `fast`, `medium`, `slow` (default: `medium`) |
| `--padding` | Extra padding in pixels around the watermark box (default: `4`) |
| `--x, --y, --width, --height` | Override watermark box coordinates/size |

## How it works

The script calculates a bounding box for the Gemini sparkle watermark, scaling
it proportionally to the input video height (baseline 1280x720). It then either:

- **delogo** — reconstructs the region using surrounding pixels, or
- **blur** — crops, blurs, and overlays the region back.

Use `--preview` to generate a single frame with the box drawn to verify the
region before processing the full video.

Batch mode saves output to `cleaned_videos/` inside the input directory, while
single-file mode writes `<stem>_cleaned.<ext>` next to the input.
