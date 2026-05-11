from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class OvfHeader:
    xnodes: int
    ynodes: int
    znodes: int
    valuedim: int
    data_format: Literal["text", "binary4", "binary8"]


def _parse_ovf_header_and_seek_to_data(f) -> OvfHeader:
    xnodes = ynodes = znodes = valuedim = None
    data_format: str | None = None

    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading OVF header.")
        s = line.decode("utf-8", errors="replace").strip()
        if not s.startswith("#"):
            continue

        low = s.lower()
        if "xnodes:" in low:
            xnodes = int(low.split("xnodes:")[1].strip())
        elif "ynodes:" in low:
            ynodes = int(low.split("ynodes:")[1].strip())
        elif "znodes:" in low:
            znodes = int(low.split("znodes:")[1].strip())
        elif "valuedim:" in low:
            valuedim = int(low.split("valuedim:")[1].strip())
        elif "begin: data" in low:
            # Examples:
            #   "# Begin: Data Text"
            #   "# Begin: Data Binary 4"
            #   "# Begin: Data Binary 8"
            if "text" in low:
                data_format = "text"
            elif "binary" in low and "4" in low:
                data_format = "binary4"
            elif "binary" in low and "8" in low:
                data_format = "binary8"
            else:
                raise ValueError(f"Unsupported data section header: {s}")
            break

    if None in (xnodes, ynodes, znodes, valuedim) or data_format is None:
        raise ValueError(
            f"Missing required OVF header fields: "
            f"xnodes={xnodes}, ynodes={ynodes}, znodes={znodes}, valuedim={valuedim}, data_format={data_format}"
        )

    return OvfHeader(
        xnodes=xnodes,
        ynodes=ynodes,
        znodes=znodes,
        valuedim=valuedim,
        data_format=data_format,  # type: ignore[arg-type]
    )


def read_ovf(path: str) -> tuple[OvfHeader, np.ndarray]:
    """
    Reads MuMax3 OVF2 files (Text, Binary 4, Binary 8).

    Returns:
      header, data array of shape (znodes, ynodes, xnodes, valuedim)
    """
    with open(path, "rb") as f:
        header = _parse_ovf_header_and_seek_to_data(f)
        n = header.xnodes * header.ynodes * header.znodes * header.valuedim

        if header.data_format == "text":
            # OVF text: whitespace-separated floats until EOF / End: Data
            # We read the remainder and parse numbers; this is slower but simple.
            rest = f.read().decode("utf-8", errors="replace")
            # Strip possible trailing "# End: Data" and other comments.
            lines = []
            for ln in rest.splitlines():
                st = ln.strip()
                if st.startswith("#"):
                    continue
                if st:
                    lines.append(st)
            arr = np.fromstring(" ".join(lines), sep=" ", dtype=np.float64)
            if arr.size < n:
                raise ValueError(f"OVF text data too short: got {arr.size} floats, expected {n}.")
            arr = arr[:n]
        elif header.data_format == "binary4":
            # OOMMF/OVF binary blocks usually start with a check value 1234567.0 (float32)
            check = np.frombuffer(f.read(4), dtype="<f4", count=1)
            if check.size != 1:
                raise ValueError("OVF binary4 missing check value.")
            arr = np.frombuffer(f.read(4 * n), dtype="<f4", count=n).astype(np.float64, copy=False)
            if arr.size != n:
                raise ValueError(f"OVF binary4 data too short: got {arr.size} floats, expected {n}.")
        elif header.data_format == "binary8":
            check = np.frombuffer(f.read(8), dtype="<f8", count=1)
            if check.size != 1:
                raise ValueError("OVF binary8 missing check value.")
            arr = np.frombuffer(f.read(8 * n), dtype="<f8", count=n)
            if arr.size != n:
                raise ValueError(f"OVF binary8 data too short: got {arr.size} floats, expected {n}.")
        else:
            raise ValueError(f"Unsupported OVF data format: {header.data_format}")

    data = arr.reshape((header.znodes, header.ynodes, header.xnodes, header.valuedim))
    return header, data


def hsv_from_m(mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> np.ndarray:
    """
    A MuMax-like visualization:
      - hue: in-plane angle atan2(my, mx)
      - saturation: 1
      - value: mapped from mz to [0,1] with a mild contrast boost
    """
    ang = np.arctan2(my, mx)
    h = (ang + math.pi) / (2 * math.pi)
    s = np.ones_like(h)
    v = np.clip((mz + 1.0) / 2.0, 0.0, 1.0)
    # MuMax GUI tends to look more contrasty than a linear mz->value map.
    # Gamma < 1 brightens mid-tones while keeping black/white endpoints.
    v = np.power(v, 0.65, dtype=np.float64)

    # HSV -> RGB
    import matplotlib.colors as mcolors

    hsv = np.stack([h, s, v], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot MuMax3 OVF fields in a GUI-like style (HSV + arrows)."
    )
    ap.add_argument("ovf", help="Path to .ovf file (e.g. m_martensite_final.ovf)")
    ap.add_argument("--z", type=int, default=0, help="Z-slice index (default: 0)")
    ap.add_argument(
        "--avg-z",
        action="store_true",
        help="Average over z instead of selecting a slice.",
    )
    ap.add_argument(
        "--downsample",
        type=int,
        default=8,
        help="Arrow downsampling factor (default: 8). Use 0 to disable arrows.",
    )
    ap.add_argument(
        "--arrow-scale",
        type=float,
        default=10.0,
        help="Arrow scale passed to matplotlib quiver (default: 10). Smaller -> longer arrows.",
    )
    ap.add_argument(
        "--arrow-normalize",
        action="store_true",
        help="Normalize in-plane vectors (mx,my) so arrows are visible even when mz dominates.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output png path (default: next to ovf with .png).",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Figure title (default: filename).",
    )

    args = ap.parse_args()

    header, data = read_ovf(args.ovf)

    if header.valuedim == 3:
        if args.avg_z:
            slab = data.mean(axis=0)  # (y,x,3)
        else:
            z = int(np.clip(args.z, 0, header.znodes - 1))
            slab = data[z, ...]  # (y,x,3)
        mx, my, mz = slab[..., 0], slab[..., 1], slab[..., 2]
        rgb = hsv_from_m(mx, my, mz)

        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=140)
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        ds = int(args.downsample)
        if ds > 0:
            # quiver expects x/y in data coordinates
            yy, xx = np.mgrid[0 : mx.shape[0], 0 : mx.shape[1]]
            qmx = mx
            qmy = my
            if args.arrow_normalize:
                mag = np.sqrt(qmx * qmx + qmy * qmy)
                mag = np.where(mag > 0, mag, 1.0)
                qmx = qmx / mag
                qmy = qmy / mag
            ax.quiver(
                xx[::ds, ::ds],
                yy[::ds, ::ds],
                qmx[::ds, ::ds],
                qmy[::ds, ::ds],
                color="black",
                angles="xy",
                scale_units="xy",
                scale=args.arrow_scale,
                width=0.0025,
                headwidth=3.0,
                headlength=4.0,
                headaxislength=3.5,
                pivot="mid",
                alpha=0.7,
            )

        title = args.title if args.title is not None else os.path.basename(args.ovf)
        ax.set_title(title)

    elif header.valuedim == 1:
        if args.avg_z:
            slab = data.mean(axis=0)[..., 0]
        else:
            z = int(np.clip(args.z, 0, header.znodes - 1))
            slab = data[z, ..., 0]

        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=140)
        im = ax.imshow(slab, origin="lower", cmap="RdBu_r", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        title = args.title if args.title is not None else os.path.basename(args.ovf)
        ax.set_title(title)
    else:
        raise ValueError(f"Unsupported valuedim={header.valuedim}. Expected 1 or 3.")

    out = args.out
    if out is None:
        base, _ = os.path.splitext(args.ovf)
        out = base + ".png"

    fig.tight_layout()
    fig.savefig(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

