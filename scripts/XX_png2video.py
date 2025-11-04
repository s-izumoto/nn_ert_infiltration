# ===============================================================
# XX_png2video.py
# ---------------------------------------------------------------
# This script converts a series of PNG images in a folder into a video.
# It first attempts to use OpenCV for writing the video; if OpenCV is
# not available or fails, it automatically falls back to imageio.
#
# Usage example:
#   python XX_png2video.py -i frames -o out.mp4 -p "*.png" --fps 12 --size 1280x720 --debug
#
# Main features:
#   - Automatically detects numeric order of filenames (e.g., frame_1, frame_2, ...)
#   - Supports resizing to a specified resolution
#   - Can repeat each frame multiple times
#   - Works without OpenCV (fallback to imageio)
# ===============================================================

import argparse, os, glob, re, sys
from typing import List, Tuple

# --- Utility functions ---

def natural_key(s: str):
    """Sort filenames in natural (human-friendly) order, e.g., frame2 < frame10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def parse_size(s: str) -> Tuple[int, int]:
    """Parse a string like '1280x720' into (width, height)."""
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("Size must be specified as WxH (e.g., 1280x720)")

def collect_images(folder: str, pattern: str) -> List[str]:
    """Collect and sort image paths within the given folder using the specified pattern."""
    paths = sorted(glob.glob(os.path.join(folder, pattern)), key=natural_key)
    return [p for p in paths if os.path.isfile(p)]

def ensure_rgb(arr):
    """Convert a grayscale or RGBA image to RGB."""
    import numpy as np
    if arr.ndim == 2:  # Grayscale → RGB
        return np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:  # RGBA → RGB (drop alpha channel)
        return arr[:, :, :3]
    return arr

def resize_np(arr, W, H):
    """Resize an image using Pillow (used when OpenCV is unavailable)."""
    from PIL import Image
    import numpy as np
    img = Image.fromarray(arr)
    img = img.resize((W, H), Image.BILINEAR)
    return np.asarray(img)

# --- Video writing using OpenCV ---

def write_with_opencv(img_paths, out, fps, size, repeat, debug):
    """Attempt to write a video using OpenCV."""
    try:
        import cv2
        import numpy as np
    except Exception as e:
        if debug: print("[opencv] Import failed:", e)
        return 0, "import_error"

    if not img_paths:
        return 0, "no_images"

    # Read the first image
    first = cv2.imread(img_paths[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        return 0, f"read_fail:{img_paths[0]}"
    first = ensure_rgb(first)
    H0, W0 = first.shape[:2]
    if size:
        W, H = size
    else:
        W, H = W0, H0

    # Try multiple FourCC codecs
    for fourcc_name in ["mp4v", "avc1", "H264", "XVID"]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(out, fourcc, fps, (W, H))
        if writer.isOpened():
            selected = fourcc_name
            break
        writer.release()
        writer = None
    if writer is None:
        return 0, "videowriter_open_fail"

    def prep(img):
        """Prepare an image for writing (ensure RGB and resize)."""
        if img is None: return None
        img = ensure_rgb(img)
        if (img.shape[1], img.shape[0]) != (W, H):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img

    frames = 0
    img = prep(first)
    if img is None:
        writer.release()
        return 0, f"read_fail:{img_paths[0]}"
    for _ in range(max(1, repeat)):
        writer.write(img)
        frames += 1

    # Process remaining frames
    for p in img_paths[1:]:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            if debug: print(f"[opencv][warn] Failed to read: {p}")
            continue
        im = prep(im)
        if im is None:
            if debug: print(f"[opencv][warn] Preprocessing failed: {p}")
            continue
        for _ in range(max(1, repeat)):
            writer.write(im)
            frames += 1

    writer.release()
    if debug: print(f"[opencv] fourcc={selected}, frames={frames}, size={W}x{H}, fps={fps}")
    return frames, "ok"

# --- Video writing using imageio (fallback) ---

def write_with_imageio(img_paths, out, fps, size, repeat, debug):
    """Fallback method using imageio if OpenCV is unavailable."""
    try:
        import imageio.v3 as iio
        import imageio
        import numpy as np
    except Exception as e:
        if debug: print("[imageio] Import failed:", e)
        return 0, "import_error"

    if not img_paths:
        return 0, "no_images"

    # Read first image
    first = iio.imread(img_paths[0])
    first = ensure_rgb(first)

    if size:
        W, H = size
        first = resize_np(first, W, H)
    else:
        H, W = first.shape[:2]

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    writer = imageio.get_writer(out, fps=fps)  # Uses imageio-ffmpeg backend

    frames = 0
    for _ in range(max(1, repeat)):
        writer.append_data(first)
        frames += 1

    # Write remaining frames
    for p in img_paths[1:]:
        try:
            im = iio.imread(p)
        except Exception:
            if debug: print(f"[imageio][warn] Failed to read: {p}")
            continue
        im = ensure_rgb(im)
        if (im.shape[1], im.shape[0]) != (W, H):
            im = resize_np(im, W, H)
        for _ in range(max(1, repeat)):
            writer.append_data(im)
            frames += 1

    writer.close()
    if debug: print(f"[imageio] frames={frames}, size={W}x{H}, fps={fps}")
    return frames, "ok"

# --- Main entry point ---

def main():
    ap = argparse.ArgumentParser(description="Create a video from PNG images (OpenCV → imageio fallback).")
    ap.add_argument("-i", "--input", required=True, help="Input folder containing images")
    ap.add_argument("-o", "--output", default="output.mp4", help="Output video path (.mp4 recommended)")
    ap.add_argument("-p", "--pattern", default="*.png", help="Filename pattern (e.g., *.png, frame_*.png)")
    ap.add_argument("--fps", type=int, default=10, help="Frames per second")
    ap.add_argument("--size", type=parse_size, default=None, help="Resize video to WxH (e.g., 1280x720)")
    ap.add_argument("--start", type=int, default=None, help="Start index of images to include")
    ap.add_argument("--end", type=int, default=None, help="End index of images to include")
    ap.add_argument("--repeat", type=int, default=1, help="Number of times to repeat each frame")
    ap.add_argument("--debug", action="store_true", help="Print debug messages")
    args = ap.parse_args()

    imgs = collect_images(args.input, args.pattern)
    if args.start is not None or args.end is not None:
        imgs = imgs[args.start or 0 : args.end or len(imgs)]
    if not imgs:
        sys.exit(f"[error] No images found: {os.path.join(args.input, args.pattern)}")

    if args.debug:
        print(f"[debug] Number of images: {len(imgs)}  First 3: {imgs[:3]}")

    # Try OpenCV first
    frames, status = write_with_opencv(imgs, args.output, args.fps, args.size, args.repeat, args.debug)
    if frames > 0:
        print(f"[done] Successfully written with OpenCV: {args.output}  frames={frames}")
        return
    if args.debug:
        print(f"[debug] OpenCV failed: {status} → Falling back to imageio")

    # Fallback to imageio
    frames, status = write_with_imageio(imgs, args.output, args.fps, args.size, args.repeat, args.debug)
    if frames > 0:
        print(f"[done] Successfully written with imageio: {args.output}  frames={frames}")
        return

    sys.exit(f"[error] Failed to write video with either backend (status={status}).")

if __name__ == "__main__":
    main()
