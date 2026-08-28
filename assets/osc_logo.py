"""Draw the Open Spine Consortium mark.

A Zenodo community avatar is seen at about 40 pixels beside a name and almost never larger.
That constraint drives everything: at that size a detailed silhouette turns to mush, so the
mark has to survive being reduced to a handful of shapes.

The idea it carries is the consortium's own thesis rather than generic anatomy. Vertebral
level is conventionally established by counting down from C2, and an abdominal CT does not
contain C2 -- so the work is about recognising WHICH level you are looking at from the level
itself. The mark is a lumbar column with one vertebra picked out in gold: the identified
level, known without the count.

Two earlier attempts are worth recording, because each failed for a reason that is easy to
repeat:

  TOO MUCH CURVE. Tilting the bodies hard for anatomical fidelity made them read as a pile
  of cushions. Fidelity that does not survive downscaling is not fidelity.

  THE PROCESS FUSED WITH THE BODY. Adding a posterior process in the same colour, touching
  the body, merged the two into one long bar at every size -- a stack of planks, not
  vertebrae. Same-coloured shapes that touch are one shape.

What actually makes a lateral spine legible small is the RHYTHM: bone, dark gap, bone, dark
gap, gently curving, on a wedge. So the disc spaces are as deliberate as the bodies, and the
whole assembly is centred by measuring its bounding box rather than by eye.

Deliberately not: a caduceus, a cross, a helix, or a whole skeleton. Two colours and one
accent, no gradients, no thin strokes -- everything that vanishes under downscaling.
"""
import argparse
import math
from pathlib import Path

PETROL = "#12303A"      # ground
BONE = "#F2ECE1"        # vertebral bodies
GOLD = "#E4B363"        # the identified level
RING_COLOR = "#27505E"  # one step off the ground, so it reads as depth not decoration

N = 5                   # lumbar bodies
LIT = 3                 # the picked-out level, second from the bottom


def geometry(S):
    """Body rectangles and the sacrum, in a local frame; centred afterwards."""
    bh = S * 0.082                     # body height
    disc = S * 0.038                   # the gap between bodies -- as designed as the bodies
    bw = S * 0.245
    sweep = S * 0.032                  # gentle lordosis
    bodies = []
    for i in range(N):
        t = i / (N - 1)
        y = i * (bh + disc)
        x = sweep * math.sin(math.pi * t)
        w = bw * (1 + 0.14 * t)
        tilt = -5.0 * math.cos(math.pi * t)
        bodies.append((x, y, w, bh, tilt, GOLD if i == LIT else BONE))
    # sacrum: a wedge under the column, separated by a disc space so it stays distinct
    sy = (N - 1) * (bh + disc) + bh / 2 + disc + bh * 0.95
    sw = bw * 1.00
    return bodies, (0.0, sy, sw, bh * 1.65), bh, disc


def build(size=1024, ring=True):
    S = size
    bodies, sac, bh, disc = geometry(S)

    xs, ys = [], []
    for x, y, w, h, _, _ in bodies:
        xs += [x - w / 2, x + w / 2]
        ys += [y - h / 2, y + h / 2]
    sx, sy, sw, sh = sac
    xs += [sx - sw / 2, sx + sw / 2]
    ys += [sy - sh / 2, sy + sh / 2]
    # centre the drawn mark in the badge instead of trusting the arithmetic above
    dx = S / 2 - (min(xs) + max(xs)) / 2
    dy = S / 2 - (min(ys) + max(ys)) / 2

    P = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" '
         f'width="{S}" height="{S}">',
         f'<rect width="{S}" height="{S}" rx="{S * 0.22:.1f}" fill="{PETROL}"/>']
    if ring:
        P.append(f'<circle cx="{S / 2}" cy="{S / 2}" r="{S * 0.405:.1f}" fill="none" '
                 f'stroke="{RING_COLOR}" stroke-width="{S * 0.015:.1f}"/>')
    P.append(f'<g transform="translate({dx:.2f} {dy:.2f})">')

    for x, y, w, h, tilt, fill in bodies:
        P.append(f'<g transform="rotate({tilt:.2f} {x:.2f} {y:.2f})">'
                 f'<rect x="{x - w / 2:.2f}" y="{y - h / 2:.2f}" width="{w:.2f}" '
                 f'height="{h:.2f}" rx="{h * 0.30:.2f}" fill="{fill}"/></g>')

    hw, hh = sw / 2, sh / 2
    # The sacrum tapers, but it does not come to a point -- a sharp wedge reads as a
    # downward arrow at avatar size, which is the wrong glyph entirely. Keep the taper,
    # blunt the bottom, and round the corners so it sits as bone rather than as a marker.
    P.append(f'<path d="M {sx - hw:.1f} {sy - hh:.1f} '
             f'L {sx + hw:.1f} {sy - hh:.1f} '
             f'L {sx + hw * 0.44:.1f} {sy + hh:.1f} '
             f'L {sx - hw * 0.34:.1f} {sy + hh:.1f} Z" fill="{BONE}" '
             f'stroke="{BONE}" stroke-width="{S * 0.018:.1f}" stroke-linejoin="round"/>')
    P.append("</g></svg>")
    return "\n".join(P)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/osc_logo")
    ap.add_argument("--size", type=int, default=1024)
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    svg = build(a.size)
    Path(f"{out}.svg").write_text(svg, encoding="utf-8")

    import cairosvg
    px_list = (1024, 512, 256, 64, 40)
    for px in px_list:
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{out}_{px}.png",
                         output_width=px, output_height=px)
    sizes = {px: Path(f"{out}_{px}.png").stat().st_size for px in px_list}
    print(f"  wrote {out}.svg and PNGs: "
          + ", ".join(f"{k}px {v / 1024:.0f} KB" for k, v in sizes.items()))
    print(f"  largest {max(sizes.values()) / 1024:.0f} KB, under Zenodo's 1 MB cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
