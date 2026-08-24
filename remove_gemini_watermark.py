#!/usr/bin/env python3
"""
Gemini Watermark Remover
========================
Removes the Gemini AI sparkle watermark from videos and frame image sequences.

Features:
  - Exact Geometric Star Masking: Uses the precise 4-pointed circular arc geometry of the Gemini logo.
  - Inpainting (Telea / Navier-Stokes): Seamlessly propagates surrounding texture and lines across the logo with zero boundary artifacts or outline rings.
  - Frame Sequence & Folder Processing: Inpaints entire folders of frame images in parallel, retaining exact frame filenames in the output directory.
  - Single Image & Video Support: Cleans standalone images (PNG, JPG, WebP, etc.) and video files (MP4, MOV, MKV, etc.).
  - High-Speed Streaming: Processes raw video through FFmpeg pipes at 300-500+ FPS.
  - Dynamic Resolution Scaling: Probes resolution and computes exact bounding boxes for 720p, 1080p, 4K, vertical 9:16 Shorts, etc.

Requirements:
  - Python 3.8+ with opencv-python and numpy.
  - FFmpeg and FFprobe (required for video processing).
"""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def natural_sort_key(s: str):
    """Sort strings containing numbers in human-natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


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
    """Extract video dimensions, fps, duration, and stream metadata using ffprobe."""
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

        fps_str = stream.get("r_frame_rate", "24/1")
        if "/" in fps_str:
            num, den = map(float, fps_str.split("/"))
            fps = num / den if den != 0 else 24.0
        else:
            fps = float(fps_str)

        return {"width": width, "height": height, "codec": codec, "fps": fps}
    except Exception as e:
        print(f"[ERROR] Failed to probe video '{input_path}': {e}")
        sys.exit(1)


def calculate_watermark_box(width: int, height: int) -> tuple[int, int, int, int]:
    """
    Calculate (x, y, w, h) bounding box for the Gemini sparkle watermark
    proportional to the resolution.

    Baseline on 1280x720 (16:9):
      - Center is at x=1160 (120px from right edge), y=600 (120px from bottom edge).
      - Logo bounding box size is 80x80 (diameter ~50px with buffer).
    """
    scale = height / 720.0
    box_w = int(round(80 * scale))
    box_h = int(round(80 * scale))

    # Keep even dimensions for encoder alignment
    box_w += box_w % 2
    box_h += box_h % 2

    center_from_right = int(round(120 * scale))
    center_from_bottom = int(round(120 * scale))

    center_x = width - center_from_right
    center_y = height - center_from_bottom

    x = center_x - (box_w // 2)
    y = center_y - (box_h // 2)

    # Keep strictly inside frame
    x = max(0, min(x, width - box_w))
    y = max(0, min(y, height - box_h))
    box_w = min(box_w, width - x)
    box_h = min(box_h, height - y)

    return x, y, box_w, box_h


def generate_gemini_star_mask(box_w: int, box_h: int, dilation: int = 2) -> np.ndarray:
    """
    Constructs the exact 4-pointed circular arc geometry of the Google Gemini star logo.
    Dilation expands the mask by 2-3 pixels to guarantee full coverage of anti-aliased edges.
    """
    cx, cy = (box_w - 1.0) / 2.0, (box_h - 1.0) / 2.0
    r = min(box_w, box_h) * 0.38

    Y, X = np.ogrid[:box_h, :box_w]
    u = np.abs(X - cx) / (r + 1e-5)
    v = np.abs(Y - cy) / (r + 1e-5)

    # 4 circular arcs tangent to each other at the tips
    star = (
        (u <= 1.02)
        & (v <= 1.02)
        & (((1.0 - np.clip(u, 0, 1.0)) ** 2 + (1.0 - np.clip(v, 0, 1.0)) ** 2) >= 0.98)
    ).astype(np.uint8) * 255

    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
        star = cv2.dilate(star, kernel, iterations=1)

    return star


def remove_watermark_image(
    input_file: str,
    output_file: str,
    x: int = None,
    y: int = None,
    w: int = None,
    h: int = None,
    method: str = "inpaint",
    dilation: int = 2,
    inpaint_radius: int = 3,
    preview_only: bool = False,
) -> bool:
    """
    Cleans watermark from a single image frame using exact geometric star inpainting.
    Supports RGBA and RGB image formats.
    """
    if not OPENCV_AVAILABLE:
        print("[ERROR] OpenCV ('opencv-python') is required for image frame processing.")
        return False

    img = cv2.imread(str(input_file), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[ERROR] Could not read image '{input_file}'.")
        return False

    # Handle grayscale, RGB, and RGBA
    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    elif len(img.shape) == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = None
    else:
        bgr = img
        alpha = None

    height, width = bgr.shape[:2]

    auto_x, auto_y, auto_w, auto_h = calculate_watermark_box(width, height)
    final_x = x if x is not None else auto_x
    final_y = y if y is not None else auto_y
    final_w = w if w is not None else auto_w
    final_h = h if h is not None else auto_h

    # Generate exact star mask and inpaint local crop
    mask = generate_gemini_star_mask(final_w, final_h, dilation=dilation)
    inpaint_flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA

    crop = bgr[final_y:final_y + final_h, final_x:final_x + final_w]
    inpainted = cv2.inpaint(crop, mask, inpaintRadius=inpaint_radius, flags=inpaint_flag)
    bgr[final_y:final_y + final_h, final_x:final_x + final_w] = inpainted

    if preview_only:
        cv2.rectangle(
            bgr,
            (final_x, final_y),
            (final_x + final_w, final_y + final_h),
            (0, 255, 0),
            2,
        )

    # Reconstruct final image
    if has_alpha:
        out_img = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])
    else:
        out_img = bgr

    # Ensure output parent directory exists
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), out_img)
    return True


def process_image_folder(
    input_dir: Path,
    output_dir: Path,
    x: int = None,
    y: int = None,
    w: int = None,
    h: int = None,
    method: str = "inpaint",
    dilation: int = 2,
    inpaint_radius: int = 3,
    preview_only: bool = False,
    workers: int = None,
) -> bool:
    """
    Processes all frame images within a directory in parallel.
    Outputs cleaned frames into output_dir with identical filenames.
    """
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"[WARNING] No image frames found in directory '{input_dir}'.")
        return False

    # Sort frames naturally (e.g. frame_1.png, frame_2.png, frame_10.png)
    image_files.sort(key=lambda f: natural_sort_key(f.name))
    total_frames = len(image_files)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Found {total_frames} frame(s) in '{input_dir}'")
    print(f"[INFO] Output folder: '{output_dir}' (retaining exact filenames)")
    print(f"[INFO] Inpainting method: {method} (dilation={dilation}px, radius={inpaint_radius}px)...")

    max_workers = workers or min(os.cpu_count() or 4, 16)
    print(f"[INFO] Processing using {max_workers} worker threads...")

    t0 = time.time()
    completed_count = 0
    errors = 0

    def process_single(frame_path: Path):
        # Output frame retains the exact same name
        target_path = output_dir / frame_path.name
        ok = remove_watermark_image(
            input_file=str(frame_path),
            output_file=str(target_path),
            x=x,
            y=y,
            w=w,
            h=h,
            method=method,
            dilation=dilation,
            inpaint_radius=inpaint_radius,
            preview_only=preview_only,
        )
        return ok

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single, f): f for f in image_files}

        for future in concurrent.futures.as_completed(future_to_file):
            frame_file = future_to_file[future]
            try:
                success = future.result()
                if success:
                    completed_count += 1
                else:
                    errors += 1
                    print(f"[ERROR] Failed processing '{frame_file.name}'")
            except Exception as exc:
                errors += 1
                print(f"[ERROR] Exception on '{frame_file.name}': {exc}")

            # Dynamic progress output
            progress = (completed_count + errors) / total_frames * 100
            elapsed = time.time() - t0
            fps_speed = (completed_count + errors) / elapsed if elapsed > 0 else 0
            sys.stdout.write(
                f"\r[PROGRESS] {completed_count + errors}/{total_frames} frames ({progress:.1f}%) "
                f"- {fps_speed:.1f} fps - Elapsed: {elapsed:.1f}s"
            )
            sys.stdout.flush()

    sys.stdout.write("\n")
    total_time = time.time() - t0
    avg_fps = total_frames / total_time if total_time > 0 else 0

    print(f"[SUCCESS] Finished {completed_count}/{total_frames} frames in {total_time:.2f}s ({avg_fps:.1f} fps).")
    if errors > 0:
        print(f"[WARNING] {errors} frame(s) encountered errors.")
    print(f"[SUCCESS] All frames saved to: {output_dir}")
    return errors == 0


def remove_watermark_inpaint(
    input_file: str,
    output_file: str,
    x: int,
    y: int,
    w: int,
    h: int,
    width: int,
    height: int,
    fps: float,
    crf: int = 18,
    preset: str = "fast",
    dilation: int = 2,
    inpaint_radius: int = 3,
    use_ns: bool = False,
) -> bool:
    """
    Removes the watermark from video using OpenCV Fast Marching (Telea) or Navier-Stokes inpainting
    on the exact dilated star mask. Streams raw frames through FFmpeg for extreme speed.
    """
    mask = generate_gemini_star_mask(w, h, dilation=dilation)
    inpaint_flag = cv2.INPAINT_NS if use_ns else cv2.INPAINT_TELEA
    flag_name = "Navier-Stokes" if use_ns else "Telea (Fast Marching)"
    print(f"[INFO] Inpainting using {flag_name} on exact star mask (dilation={dilation}px)...")

    in_cmd = [
        "ffmpeg",
        "-i", input_file,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    in_proc = subprocess.Popen(in_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    out_cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-i", input_file,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "copy",
        "-y",
        output_file,
    ]
    out_proc = subprocess.Popen(out_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_bytes = width * height * 3
    frame_count = 0
    t0 = time.time()

    try:
        while True:
            raw = in_proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
            crop = frame[y:y + h, x:x + w]

            # Inpaint only the local crop for maximum performance
            inpainted = cv2.inpaint(crop, mask, inpaintRadius=inpaint_radius, flags=inpaint_flag)
            frame[y:y + h, x:x + w] = inpainted

            out_proc.stdin.write(frame.tobytes())
            frame_count += 1

        in_proc.stdout.close()
        out_proc.stdin.close()
        in_proc.wait()
        out_proc.wait()

        elapsed = time.time() - t0
        fps_speed = frame_count / elapsed if elapsed > 0 else 0
        print(f"[SUCCESS] Processed {frame_count} frames in {elapsed:.2f}s ({fps_speed:.1f} fps)")
        print(f"[SUCCESS] Cleaned video saved to: {output_file}")
        return True
    except Exception as e:
        print(f"[ERROR] Inpainting failed: {e}")
        in_proc.kill()
        out_proc.kill()
        return False


def remove_watermark_ffmpeg_filter(
    input_file: str,
    output_file: str,
    x: int,
    y: int,
    w: int,
    h: int,
    method: str = "delogo",
    crf: int = 18,
    preset: str = "medium",
) -> bool:
    """Fallback filter removal using FFmpeg's delogo or blur filters."""
    if method == "delogo":
        filter_str = f"delogo=x={x}:y={y}:w={w}:h={h}"
    elif method == "blur":
        filter_str = (
            f"[0:v]crop={w}:{h}:{x}:{y},avgblur=sizeX=7:sizeY=7[blurred];"
            f"[0:v][blurred]overlay={x}:{y}"
        )
    else:
        print(f"[ERROR] Unknown method: {method}")
        return False

    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-filter_complex" if method == "blur" else "-vf", filter_str,
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "copy",
        "-y",
        output_file,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[SUCCESS] Cleaned video saved to: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg processing failed: {e.stderr}")
        return False


def remove_watermark(
    input_file: str,
    output_file: str,
    x: int = None,
    y: int = None,
    w: int = None,
    h: int = None,
    method: str = "inpaint",
    crf: int = 18,
    preset: str = "fast",
    dilation: int = 2,
    inpaint_radius: int = 3,
    preview_only: bool = False,
    preview_frame_sec: float = 2.0,
) -> bool:
    """Main removal coordinator function for video files."""
    if not os.path.isfile(input_file):
        print(f"[ERROR] Input video '{input_file}' not found.")
        return False

    info = get_video_info(input_file)
    width, height, fps = info["width"], info["height"], info["fps"]
    print(f"[INFO] Video: {width}x{height} @ {fps:.2f}fps ({info['codec']})")

    auto_x, auto_y, auto_w, auto_h = calculate_watermark_box(width, height)
    final_x = x if x is not None else auto_x
    final_y = y if y is not None else auto_y
    final_w = w if w is not None else auto_w
    final_h = h if h is not None else auto_h

    print(f"[INFO] Watermark Bounding Box: x={final_x}, y={final_y}, w={final_w}, h={final_h}")

    # Preview Mode
    if preview_only:
        preview_img = (
            output_file
            if output_file.endswith((".png", ".jpg"))
            else f"{os.path.splitext(output_file)[0]}_preview.png"
        )
        print(f"[INFO] Generating preview frame at {preview_frame_sec}s -> '{preview_img}'...")

        if OPENCV_AVAILABLE:
            cmd = [
                "ffmpeg",
                "-ss", str(preview_frame_sec),
                "-i", input_file,
                "-vframes", "1",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-",
            ]
            raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()

            mask = generate_gemini_star_mask(final_w, final_h, dilation=dilation)
            crop = frame[final_y:final_y + final_h, final_x:final_x + final_w]
            frame[final_y:final_y + final_h, final_x:final_x + final_w] = cv2.inpaint(
                crop, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA
            )

            # Draw green bounding box around processed area for visual confirmation
            cv2.rectangle(
                frame,
                (final_x, final_y),
                (final_x + final_w, final_y + final_h),
                (0, 255, 0),
                2,
            )
            cv2.imwrite(preview_img, frame)
            print(f"[SUCCESS] Preview saved to: {preview_img}")
            return True
        else:
            vf = f"delogo=x={final_x}:y={final_y}:w={final_w}:h={final_h}:show=1"
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
            return False

    # Video Processing
    if method in ("inpaint", "ns"):
        if OPENCV_AVAILABLE:
            return remove_watermark_inpaint(
                input_file=input_file,
                output_file=output_file,
                x=final_x,
                y=final_y,
                w=final_w,
                h=final_h,
                width=width,
                height=height,
                fps=fps,
                crf=crf,
                preset=preset,
                dilation=dilation,
                inpaint_radius=inpaint_radius,
                use_ns=(method == "ns"),
            )
        else:
            print("[WARNING] 'opencv-python' is not installed. Falling back to FFmpeg delogo.")
            method = "delogo"

    print(f"[INFO] Processing video with FFmpeg filter method '{method}'...")
    return remove_watermark_ffmpeg_filter(
        input_file=input_file,
        output_file=output_file,
        x=final_x,
        y=final_y,
        w=final_w,
        h=final_h,
        method=method,
        crf=crf,
        preset=preset,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Remove Gemini watermark from videos, frame image folders, and single images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Methods:
  inpaint (default) : Uses OpenCV Telea inpainting on the exact dilated 4-pointed Gemini star geometry (zero outline).
  ns                : Uses Navier-Stokes inpainting for high-gradient backgrounds.
  delogo            : Uses FFmpeg native spatial interpolation (videos only).
  blur              : Applies localized blur overlay (videos only).

Examples:
  # Folder of multiple frames -> Output to another folder with identical frame filenames:
  python remove_gemini_watermark.py ./frames_folder/ -o ./cleaned_frames/
  python remove_gemini_watermark.py ./frames_folder/

  # Single frame image:
  python remove_gemini_watermark.py frame.png -o frame_cleaned.png

  # Single video file:
  python remove_gemini_watermark.py input.mp4 -o clean_output.mp4

  # Single video preview:
  python remove_gemini_watermark.py input.mp4 --preview

  # Batch process video files in a folder:
  python remove_gemini_watermark.py ./videos_folder/ --batch
""",
    )

    parser.add_argument("input", help="Path to input video file, image frame, or folder of frames/videos")
    parser.add_argument("-o", "--output", help="Path to output video/image file or destination folder")
    parser.add_argument("--batch", action="store_true", help="Batch process all videos/frames in the input directory")
    parser.add_argument("--preview", action="store_true", help="Generate a preview image with the watermark removed and box marked")
    parser.add_argument("--preview-time", type=float, default=2.0, help="Timestamp (seconds) for preview frame in video (default: 2.0)")
    parser.add_argument(
        "--method",
        choices=["inpaint", "ns", "delogo", "blur"],
        default="inpaint",
        help="Removal method (default: inpaint)",
    )
    parser.add_argument("--dilation", type=int, default=2, help="Mask boundary dilation in pixels to fully eliminate outlines (default: 2)")
    parser.add_argument("--radius", type=int, default=3, help="Inpainting neighborhood radius (default: 3)")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF quality factor 0-51 for video encoding (lower is better, default: 18)")
    parser.add_argument("--preset", default="fast", help="x264 encoding preset: ultrafast, fast, medium, slow (default: fast)")
    parser.add_argument("--workers", type=int, help="Number of concurrent worker threads for processing frame folders")
    parser.add_argument("--x", type=int, help="Override watermark X coordinate")
    parser.add_argument("--y", type=int, help="Override watermark Y coordinate")
    parser.add_argument("--width", "-W", type=int, dest="w", help="Override watermark width")
    parser.add_argument("--height", "-H", type=int, dest="h", help="Override watermark height")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"[ERROR] Input path '{args.input}' does not exist.")
        sys.exit(1)

    # 1. DIRECTORY PROCESSING (Folder of frames or folder of videos)
    if input_path.is_dir():
        image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        video_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]

        # Case A: Folder contains image frames
        if image_files and not (args.batch and video_files and not image_files):
            output_dir = Path(args.output) if args.output else input_path.parent / f"{input_path.name}_cleaned"
            ok = process_image_folder(
                input_dir=input_path,
                output_dir=output_dir,
                x=args.x,
                y=args.y,
                w=args.w,
                h=args.h,
                method=args.method,
                dilation=args.dilation,
                inpaint_radius=args.radius,
                preview_only=args.preview,
                workers=args.workers,
            )
            if not ok:
                sys.exit(1)
            return

        # Case B: Folder contains video files
        if video_files:
            if not check_ffmpeg():
                sys.exit(1)

            output_dir = Path(args.output) if args.output else input_path / "cleaned_videos"
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] Found {len(video_files)} video(s) in '{input_path}'.")
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
                    method=args.method,
                    dilation=args.dilation,
                    inpaint_radius=args.radius,
                    crf=args.crf,
                    preset=args.preset,
                    preview_only=args.preview,
                    preview_frame_sec=args.preview_time,
                )
            print(f"\n[DONE] Video batch processing complete! Output saved to: {output_dir}")
            return

        print(f"[WARNING] No supported image frames or video files found in directory '{args.input}'.")
        sys.exit(0)

    # 2. SINGLE FILE PROCESSING (Single image frame or single video file)
    suffix = input_path.suffix.lower()

    # Case A: Single Image file
    if suffix in IMAGE_EXTENSIONS:
        if args.output:
            out_path = args.output
        else:
            out_path = str(input_path.parent / f"{input_path.stem}_cleaned{suffix}")

        print(f"[INFO] Processing single image frame: '{input_path}' -> '{out_path}'")
        ok = remove_watermark_image(
            input_file=str(input_path),
            output_file=out_path,
            x=args.x,
            y=args.y,
            w=args.w,
            h=args.h,
            method=args.method,
            dilation=args.dilation,
            inpaint_radius=args.radius,
            preview_only=args.preview,
        )
        if ok:
            print(f"[SUCCESS] Cleaned image saved to: {out_path}")
        else:
            sys.exit(1)
        return

    # Case B: Video file
    if not check_ffmpeg():
        sys.exit(1)

    if args.output:
        out_path = args.output
    else:
        out_suffix = ".png" if args.preview else input_path.suffix
        out_path = str(input_path.parent / f"{input_path.stem}_cleaned{out_suffix}")

    success = remove_watermark(
        input_file=str(input_path),
        output_file=out_path,
        x=args.x,
        y=args.y,
        w=args.w,
        h=args.h,
        method=args.method,
        dilation=args.dilation,
        inpaint_radius=args.radius,
        crf=args.crf,
        preset=args.preset,
        preview_only=args.preview,
        preview_frame_sec=args.preview_time,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
