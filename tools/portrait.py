"""FoundKeys Associates portrait style: navy-blue monochrome, round, 240x240 PNG.

Usage: python3 -I tools/portrait.py <source image> <output.png>
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Colour ramp: output luminance -> navy tint (measured from the approved Hayley example).
RAMP = [  # (level, R, G, B)
    (0, 0, 11, 32), (16, 1, 15, 38), (32, 10, 27, 52), (48, 26, 44, 70), (64, 41, 59, 86),
    (80, 57, 75, 101), (96, 74, 91, 117), (112, 91, 107, 133), (128, 107, 122, 147),
    (144, 124, 138, 162), (160, 141, 154, 177), (176, 158, 170, 191), (192, 174, 185, 205),
    (208, 190, 200, 218), (224, 210, 218, 233), (240, 230, 236, 246), (255, 246, 248, 252),
]
GAMMA = 1.35      # deepens the mid-tones like the example
CLIP = 0.5        # % of darkest/lightest pixels clipped when stretching contrast


FACE_TARGET = 135  # dark photos are lifted until the face area averages at least this
MAX_LIFT = 0.5     # strongest brightening allowed (gamma)


def face_lift(a):
    """Gamma (<=1) that brightens a dark photo's face area (upper middle) to FACE_TARGET."""
    h, w = a.shape
    face = a[int(h * 0.25):int(h * 0.625), int(w * 0.31):int(w * 0.69)] / 255.0
    if face.mean() * 255 >= FACE_TARGET:
        return 1.0
    lo, hi = MAX_LIFT, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if (face ** mid).mean() * 255 < FACE_TARGET:
            hi = mid
        else:
            lo = mid
    return hi


def navy(im):
    g = ImageOps.autocontrast(im.convert("L"), cutoff=CLIP)
    lift = face_lift(np.asarray(g).astype(float))
    x = np.arange(256) / 255.0
    tone = np.clip(((x ** lift) ** GAMMA) * 255.0, 0, 255)
    lv = [r[0] for r in RAMP]
    lut = [np.interp(tone, lv, [r[c] for r in RAMP]) for c in (1, 2, 3)]
    a = np.asarray(g)
    rgb = np.stack([lut[c][a] for c in range(3)], axis=-1).round().astype("uint8")
    return Image.fromarray(rgb, "RGB")


def make(src, out):
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    if h > w:
        t = max(0, t - int(0.05 * s))
    im = im.crop((l, t, l + s, t + s)).resize((480, 480), Image.LANCZOS)
    im = navy(im)
    m = Image.new("L", (480, 480), 0)
    ImageDraw.Draw(m).ellipse((0, 0, 479, 479), fill=255)
    o = Image.new("RGBA", (480, 480), (0, 0, 0, 0))
    o.paste(im, (0, 0), m)
    o.resize((240, 240), Image.LANCZOS).save(out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
