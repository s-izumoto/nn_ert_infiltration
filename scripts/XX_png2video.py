# png2video_robust.py
# 使い方:
#   python png2video_robust.py -i frames -o out.mp4 -p "*.png" --fps 12 --size 1280x720 --debug

import argparse, os, glob, re, sys
from typing import List, Tuple

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def parse_size(s: str) -> Tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("サイズは WxH（例: 1280x720）で指定してください")

def collect_images(folder: str, pattern: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(folder, pattern)), key=natural_key)
    return [p for p in paths if os.path.isfile(p)]

def ensure_rgb(arr):
    import numpy as np
    if arr.ndim == 2:
        return np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:  # RGBA→RGB（アルファ捨て）
        return arr[:, :, :3]
    return arr

def resize_np(arr, W, H):
    # Pillowでリサイズ（OpenCV不要）
    from PIL import Image
    import numpy as np
    img = Image.fromarray(arr)
    img = img.resize((W, H), Image.BILINEAR)
    return np.asarray(img)

def write_with_opencv(img_paths, out, fps, size, repeat, debug):
    try:
        import cv2
        import numpy as np
    except Exception as e:
        if debug: print("[opencv] import失敗:", e)
        return 0, "import_error"

    if not img_paths:
        return 0, "no_images"

    # 1枚目
    first = cv2.imread(img_paths[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        return 0, f"read_fail:{img_paths[0]}"
    first = ensure_rgb(first)
    H0, W0 = first.shape[:2]
    if size:
        W, H = size
    else:
        W, H = W0, H0

    # fourcc は汎用 mp4v を優先
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
        writer.write(img); frames += 1

    for p in img_paths[1:]:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            if debug: print(f"[opencv][warn] 読み込み失敗: {p}")
            continue
        im = prep(im)
        if im is None:
            if debug: print(f"[opencv][warn] 前処理失敗: {p}")
            continue
        for _ in range(max(1, repeat)):
            writer.write(im); frames += 1

    writer.release()
    if debug: print(f"[opencv] fourcc={selected}, frames={frames}, size={W}x{H}, fps={fps}")
    return frames, "ok"

def write_with_imageio(img_paths, out, fps, size, repeat, debug):
    try:
        import imageio.v3 as iio
        import imageio
        import numpy as np
    except Exception as e:
        if debug: print("[imageio] import失敗:", e)
        return 0, "import_error"

    if not img_paths:
        return 0, "no_images"

    # 1枚目
    first = iio.imread(img_paths[0])
    first = ensure_rgb(first)

    if size:
        W, H = size
        first = resize_np(first, W, H)
    else:
        H, W = first.shape[:2]

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    writer = imageio.get_writer(out, fps=fps)  # imageio-ffmpeg 使用

    frames = 0
    for _ in range(max(1, repeat)):
        writer.append_data(first); frames += 1

    for p in img_paths[1:]:
        try:
            im = iio.imread(p)
        except Exception:
            if debug: print(f"[imageio][warn] 読み込み失敗: {p}")
            continue
        im = ensure_rgb(im)
        if (im.shape[1], im.shape[0]) != (W, H):
            im = resize_np(im, W, H)
        for _ in range(max(1, repeat)):
            writer.append_data(im); frames += 1

    writer.close()
    if debug: print(f"[imageio] frames={frames}, size={W}x{H}, fps={fps}")
    return frames, "ok"

def main():
    ap = argparse.ArgumentParser(description="フォルダー内PNGから動画を作成（OpenCV→imageioの自動フォールバック）。")
    ap.add_argument("-i","--input", required=True, help="画像フォルダー")
    ap.add_argument("-o","--output", default="output.mp4", help="出力動画パス（.mp4推奨）")
    ap.add_argument("-p","--pattern", default="*.png", help="検索パターン（例: *.png, frame_*.png）")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--size", type=parse_size, default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    imgs = collect_images(args.input, args.pattern)
    if args.start is not None or args.end is not None:
        imgs = imgs[args.start or 0 : args.end or len(imgs)]
    if not imgs:
        sys.exit(f"[error] 画像が見つかりません: {os.path.join(args.input, args.pattern)}")

    if args.debug:
        print(f"[debug] 画像枚数: {len(imgs)} 先頭3件: {imgs[:3]}")

    # まずOpenCV
    frames, status = write_with_opencv(imgs, args.output, args.fps, args.size, args.repeat, args.debug)
    if frames > 0:
        print(f"[done] OpenCVで書き出し完了: {args.output}  frames={frames}")
        return
    if args.debug: print(f"[debug] OpenCV失敗: {status} → imageioにフォールバック")

    # imageio フォールバック
    frames, status = write_with_imageio(imgs, args.output, args.fps, args.size, args.repeat, args.debug)
    if frames > 0:
        print(f"[done] imageioで書き出し完了: {args.output}  frames={frames}")
        return

    sys.exit(f"[error] どちらの方法でも書けませんでした（status={status}）。")

if __name__ == "__main__":
    main()
