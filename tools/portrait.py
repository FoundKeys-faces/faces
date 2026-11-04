"""FoundKeys Associates portrait style "navy-studio-v1".

Each person is cut out of their photo, placed on the same soft navy studio backdrop,
toned in navy-blue monochrome and saved as a round 240x240 transparent PNG.
If the cut-out tools are missing or fail, it falls back to plain navy monochrome
of the whole photo and prints "FALLBACK".

Needs: pip install rembg onnxruntime pymatting scipy pillow numpy --break-system-packages
Usage: python3 -I tools/portrait.py <source image> <output.png>
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

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


def disk(r):
    y, x = np.ogrid[-r:r+1, -r:r+1]; return x*x + y*y <= r*r
def cutout(src):
    """Cut the person out: AI mask, drop people/objects behind, crisp matted edges, no halo."""
    from rembg import remove, new_session
    from pymatting import estimate_alpha_cf, estimate_foreground_ml
    im = ImageOps.exif_transpose(Image.open(src)).convert('RGB')
    w, h = im.size
    if min(w, h) < 600:  # work at a decent size for crisper edges
        k = 600 / min(w, h); im = im.resize((round(w*k), round(h*k)), Image.LANCZOS); w, h = im.size
    a = np.asarray(remove(im, session=new_session('isnet-general-use'), only_mask=True)).astype(float) / 255
    b = a > 0.5
    r = max(4, round(min(w, h) * 0.025))
    op = ndimage.binary_opening(b, disk(r))          # break thin bridges to things behind
    lab, n = ndimage.label(op)
    if n:
        sizes = ndimage.sum(op, lab, range(1, n+1)); best = int(np.argmax(sizes)) + 1
        keep = ndimage.binary_dilation(lab == best, disk(r + 2)) & b
        keep = ndimage.binary_fill_holes(keep)
        # top of the head: trim narrow lumps (other people's hair behind) with a stronger opening
        L = round(min(w, h) * 0.12); top = int(h * 0.35)
        big = ndimage.binary_opening(keep, np.ones((1, L), bool))   # drop rows narrower than L
        big = ndimage.binary_opening(big, disk(3))
        lab2, n2 = ndimage.label(big)
        if n2 > 1:
            sz = ndimage.sum(big, lab2, range(1, n2 + 1)); big = lab2 == (int(np.argmax(sz)) + 1)
        keep[:top] = big[:top]
    else:
        keep = b
    # trimap, then closed-form matting for crisp hair edges
    e = max(2, round(min(w, h) * 0.006))
    fg = ndimage.binary_erosion(keep, disk(e))
    bg = ~ndimage.binary_dilation(keep, disk(e))
    tri = np.full(keep.shape, 0.5); tri[fg] = 1; tri[bg] = 0
    img = np.asarray(im).astype(float) / 255
    al = estimate_alpha_cf(img, tri)
    al = np.clip((al - 0.12) / 0.76, 0, 1)            # firm up the edge, no soft ghosting
    F = estimate_foreground_ml(img, al)               # remove background colour bleeding (halo)
    rgba = np.dstack([np.clip(F, 0, 1), al])
    return Image.fromarray((rgba * 255).round().astype('uint8'), 'RGBA')


def backdrop(n=480):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - n / 2, yy - n * 0.4167) / (n * 0.708)
    return np.clip(0.42 - 0.30 * r ** 1.6, 0.08, 1)   # lighter behind the head, darker at the edge


def square(im):
    w, h = im.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    if h > w:
        t = max(0, t - int(0.05 * s))
    return im.crop((l, t, l + s, t + s)).resize((480, 480), Image.LANCZOS)


def ramp(v):
    lv = [q[0] for q in RAMP]
    return np.stack([np.interp(v, lv, [q[c] for q in RAMP]) for c in (1, 2, 3)], -1).round().astype("uint8")


def studio(src):
    im = square(cutout(src))
    a = np.asarray(im.getchannel("A")).astype(float) / 255
    g = np.asarray(im.convert("RGB").convert("L")).astype(float)
    m = a > 0.5
    lo, hi = np.percentile(g[m], [0.5, 99.5])
    g = np.clip((g - lo) / max(1, hi - lo) * 255, 0, 255)
    lift = face_lift(np.where(m, g, g[m].mean()))
    x = np.clip(((g / 255) ** lift) ** GAMMA, 0, 1)
    v = (x * a + backdrop() * (1 - a)) * 255
    return Image.fromarray(ramp(v), "RGB")


def plain(src):
    return navy(square(ImageOps.exif_transpose(Image.open(src)).convert("RGB")))


def make(src, out):
    try:
        im = studio(src)
    except Exception as e:  # cut-out tools missing or failed
        print("FALLBACK", repr(e)[:200])
        im = plain(src)
    m = Image.new("L", (480, 480), 0)
    ImageDraw.Draw(m).ellipse((0, 0, 479, 479), fill=255)
    o = Image.new("RGBA", (480, 480), (0, 0, 0, 0))
    o.paste(im, (0, 0), m)
    o.resize((240, 240), Image.LANCZOS).save(out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
