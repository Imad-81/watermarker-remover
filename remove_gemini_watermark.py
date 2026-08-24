#!/usr/bin/env python3
"""
Gemini Watermark Remover
========================
Removes the Gemini AI sparkle watermark from the bottom-right corner of videos
using FFmpeg filters (delogo / blur) with automatic resolution scaling.

Requirements:
  - FFmpeg and FFprobe installed and available in PATH.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg() -> bool:
    """Verify that ffmpeg and ffprobe are installed and available."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        print("[ERROR] 'ffmpeg' or 'ffprobe' not found in system PATH.")
        print("Please install ffmpeg (e.g. 'brew install ffmpeg' on macOS or 'sudo apt install ffmpeg' on Linux).")
        return False
    return True


def get_video_info(input_path: str) -> dict:
    """Extract video dimensions, fps, and duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,codec_name",
        "-of", "json",
        input_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        codec = stream.get("codec_name", "unknown")
        return {"width": width, "height": height, "codec": codec}
    except Exception as e:
        print(f"[ERROR] Failed to probe video '{input_path}': {e}")
        sys.exit(1)


def calculate_watermark_box(width: int, height: int, padding: int = 4) -> tuple[int, int, int, int]:
    """
    Calculate (x, y, w, h) bounding box for the Gemini sparkle watermark
    based on the video resolution.

    Baseline on 1280x720 (16:9):
      - Sparkle bounds: x: 1135..1182, y: 576..624 (size: 48x48)
      - Box with padding: x: 1132, y: 573, w: 54, h: 54
      - Relative: x ≈ 0.8844 * W, y ≈ 0.7958 * H, w ≈ 0.0422 * W (or ~0.075 * H), h ≈ 0.075 * H
    """
    # Scale proportionally with height (standard video scaling)
    scale = height / 720.0
    box_w = int(round(50 * scale)) + (padding * 2)
    box_h = int(round(50 * scale)) + (padding * 2)

    # Offset from right and bottom edges
    # For 1280x720: center is at x=1159 (121px from right), y=600 (120px from bottom)
    center_from_right = int(round(121 * scale))
    center_from_bottom = int(round(120 * scale))

    # Center coordinates
    center_x = width - center_from_right
    center_y = height - center_from_bottom

    x = center_x - (box_w // 2)
    y = center_y - (box_h // 2)

    # Boundary safety checks
    x = max(0, min(x, width - box_w))
    y = max(0, min(y, height - box_h))
    box_w = min(box_w, width - x)
    box_h = min(box_h, height - y)

    return x, y, box_w, box_h


def remove_watermark(
    input_file: str,
    output_file: str,
    x: int = None,
    y: int = None,
    w: int = None,
    h: int = None,
    padding: int = 4,
    method: str = "delogo",
    crf: int = 18,
    preset: str = "medium",
    preview_only: bool = False,
    preview_frame_sec: float = 2.0,
) -> bool:
    """
    Removes watermark using FFmpeg delogo or blur filter.
    """
    if not os.path.isfile(input_file):
        print(f"[ERROR] Input video '{input_file}' not found.")
        return False

    info = get_video_info(input_file)
    width, height = info["width"], info["height"]
    print(f"[INFO] Video resolution: {width}x{height} (Codec: {info['codec']})")

    # Determine coordinates
    auto_x, auto_y, auto_w, auto_h = calculate_watermark_box(width, height, padding)
    final_x = x if x is not None else auto_x
    final_y = y if y is not None else auto_y
    final_w = w if w is not None else auto_w
    final_h = h if h is not None else auto_h

    print(f"[INFO] Watermark Bounding Box: x={final_x}, y={final_y}, w={final_w}, h={final_h}")

    # Preview mode: Generates a side-by-side comparison image or box inspection image
    if preview_only:
        preview_img = output_file if output_file.endswith((".png", ".jpg")) else f"{os.path.splitext(output_file)[0]}_preview.png"
        print(f"[INFO] Generating preview frame at {preview_frame_sec}s -> '{preview_img}'...")
        
        # Draw green bounding box + delogo side-by-side comparison
        vf = (
            f"delogo=x={final_x}:y={final_y}:w={final_w}:h={final_h}:show=1"
        )
        cmd = [
            "ffmpeg",
            "-ss", str(preview_frame_sec),
            "-i", input_file,
            "-vf", vf,
            "-frames:v", "1",
            "-y",
            preview_img,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[SUCCESS] Preview saved to: {preview_img}")
            return True
        else:
            print(f"[ERROR] FFmpeg failed: {res.stderr}")
            return False

    # Process full video
    print(f"[INFO] Processing video with method '{method}'...")
    if method == "delogo":
        filter_str = f"delogo=x={final_x}:y={final_y}:w={final_w}:h={final_h}"
    elif method == "blur":
        # Crop region, blur it, and overlay back
        filter_str = (
            f"[0:v]crop={final_w}:{final_h}:{final_x}:{final_y},avgblur=sizeX=7:sizeY=7[blurred];"
            f"[0:v][blurred]overlay={final_x}:{final_y}"
        )
    else:
        print(f"[ERROR] Unknown method: {method}")
        return False

    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-filter_complex" if method == "blur" else "-vf", filter_str,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "copy",
        "-y",
        output_file,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[SUCCESS] Cleaned video saved to: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg processing failed: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Remove Google Gemini watermark from video using FFmpeg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python remove_gemini_watermark.py input.mp4
  python remove_gemini_watermark.py input.mp4 -o clean_output.mp4
  python remove_gemini_watermark.py input.mp4 --preview
  python remove_gemini_watermark.py input.mp4 --x 1130 --y 570 --w 56 --h 56
  python remove_gemini_watermark.py ./videos_folder/ --batch
""",
    )

    parser.add_argument("input", help="Path to input video file or folder (with --batch)")
    parser.add_argument("-o", "--output", help="Path to output video file or output directory")
    parser.add_argument("--batch", action="store_true", help="Batch process all videos in the input directory")
    parser.add_argument("--preview", action="store_true", help="Generate a preview image with the watermark bounding box marked")
    parser.add_argument("--preview-time", type=float, default=2.0, help="Timestamp (seconds) for preview frame (default: 2.0)")
    parser.add_argument("--method", choices=["delogo", "blur"], default="delogo", help="Removal method (default: delogo)")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF quality factor 0-51 (lower is better, default: 18)")
    parser.add_argument("--preset", default="medium", help="x264 encoding preset: ultrafast, fast, medium, slow (default: medium)")
    parser.add_argument("--padding", type=int, default=4, help="Extra padding in pixels around the watermark box (default: 4)")
    parser.add_argument("--x", type=int, help="Override watermark X coordinate")
    parser.add_argument("--y", type=int, help="Override watermark Y coordinate")
    parser.add_argument("--width", "-W", type=int, dest="w", help="Override watermark width")
    parser.add_argument("--height", "-H", type=int, dest="h", help="Override watermark height")

    args = parser.parse_args()

    if not check_ffmpeg():
        sys.exit(1)

    input_path = Path(args.input)

    # Batch processing mode
    if args.batch or input_path.is_dir():
        if not input_path.is_dir():
            print(f"[ERROR] '{args.input}' is not a directory.")
            sys.exit(1)

        output_dir = Path(args.output) if args.output else input_path / "cleaned_videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        video_extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
        video_files = [f for f in input_path.iterdir() if f.suffix.lower() in video_extensions]

        if not video_files:
            print(f"[WARNING] No video files found in '{args.input}'.")
            sys.exit(0)

        print(f"[INFO] Found {len(video_files)} video(s) to process.")
        for idx, vid in enumerate(video_files, 1):
            out_file = output_dir / f"{vid.stem}_cleaned{vid.suffix}"
            print(f"\n[{idx}/{len(video_files)}] Processing {vid.name}...")
            remove_watermark(
                input_file=str(vid),
                output_file=str(out_file),
                x=args.x,
                y=args.y,
                w=args.w,
                h=args.h,
                padding=args.padding,
                method=args.method,
                crf=args.crf,
                preset=args.preset,
                preview_only=args.preview,
                preview_frame_sec=args.preview_time,
            )
        print(f"\n[DONE] Batch processing complete! Output saved to: {output_dir}")
        return

    # Single file mode
    if not input_path.is_file():
        print(f"[ERROR] File '{args.input}' does not exist.")
        sys.exit(1)

    if args.output:
        out_path = args.output
    else:
        suffix = ".png" if args.preview else input_path.suffix
        out_path = str(input_path.parent / f"{input_path.stem}_cleaned{suffix}")

    success = remove_watermark(
        input_file=str(input_path),
        output_file=out_path,
        x=args.x,
        y=args.y,
        w=args.w,
        h=args.h,
        padding=args.padding,
        method=args.method,
        crf=args.crf,
        preset=args.preset,
        preview_only=args.preview,
        preview_frame_sec=args.preview_time,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
