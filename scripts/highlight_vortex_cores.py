"""
Подсветка возможных ядер вихрей по полю m (OVF с valuedim=3).

Эвристика (не строгая топология): величина |∂m_y/∂x − ∂m_x/∂y| (ротор в плоскости xy)
обычно пикает у вихрей и некоторых стенок; маска порогом + небольшое расширение.

Пример:
  python scripts/highlight_vortex_cores.py simulations/NiFeGaCo_martensite.out/m_martensite_final.ovf -o vortexes.png
  python scripts/highlight_vortex_cores.py .../m_martensite_final.ovf --z-slice 2 --percentile 97
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_mumax_output import _z_index, read_ovf_binary4  # noqa: E402

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except ImportError as e:
    print("Нужен matplotlib: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from e


def _dilate2d(mask: np.ndarray, passes: int = 4) -> np.ndarray:
    m = mask.astype(bool)
    for _ in range(passes):
        p = np.pad(m, 1, mode="constant", constant_values=False)
        m = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return m


def _curl_z(mx: np.ndarray, my: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dmy_dx = np.gradient(my, dx, axis=1)
    dmx_dy = np.gradient(mx, dy, axis=0)
    return dmy_dx - dmx_dy


def highlight_vortex_cores(
    ovf_path: Path,
    *,
    z_slice: int = -1,
    percentile: float = 98.0,
    dilate: int = 5,
    save: Path | None = None,
    dpi: int = 200,
) -> None:
    ovf_path = ovf_path.resolve()
    d, meta = read_ovf_binary4(ovf_path)
    if meta["valuedim"] != 3:
        print(f"Нужен векторный m (valuedim=3), сейчас {meta['valuedim']}", file=sys.stderr)
        raise SystemExit(1)

    zi = _z_index(meta["znodes"], z_slice)
    sl = np.asarray(d[zi, :, :, :3], dtype=np.float64)
    mx, my, mz = sl[:, :, 0], sl[:, :, 1], sl[:, :, 2]

    nx, ny = meta["xnodes"], meta["ynodes"]
    dx = (meta["xmax"] - meta["xmin"]) / max(nx - 1, 1)
    dy = (meta["ymax"] - meta["ymin"]) / max(ny - 1, 1)

    curl = _curl_z(mx, my, dx, dy)
    ac = np.abs(curl)
    thr = np.percentile(ac, percentile)
    if thr <= 0:
        thr = np.max(ac) * 0.5
    mask = ac >= thr
    mask = _dilate2d(mask, passes=dilate)

    xi = np.linspace(meta["xmin"] * 1e9, meta["xmax"] * 1e9, nx)
    yi = np.linspace(meta["ymin"] * 1e9, meta["ymax"] * 1e9, ny)
    X, Y = np.meshgrid(xi, yi, indexing="xy")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=100)

    ext = (
        meta["xmin"] * 1e9,
        meta["xmax"] * 1e9,
        meta["ymin"] * 1e9,
        meta["ymax"] * 1e9,
    )

    for ax, title in zip(
        axes,
        (r"$m_z$", r"$m_z$ + подсветка $|\partial_y m_x - \partial_x m_y|$"),
    ):
        im = ax.imshow(
            mz,
            origin="lower",
            extent=ext,
            aspect="equal",
            cmap="RdBu_r",
            norm=Normalize(-1, 1),
        )
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    yy, xx = np.where(mask)
    if len(xx) > 0:
        axes[1].scatter(
            X[yy, xx],
            Y[yy, xx],
            s=14,
            c="#39ff14",
            marker="o",
            linewidths=0.35,
            edgecolors="black",
            alpha=0.85,
            label="маска ядра (эвристика)",
        )
        axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{ovf_path.name} — z-слой {zi}/{meta['znodes'] - 1}, p={percentile:g}%", fontsize=11)
    fig.tight_layout()

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Сохранено: {save}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Подсветка вихрей по ротору m_xy")
    p.add_argument("ovf", type=Path, help="Векторный m*.ovf")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--z-slice", type=int, default=-1, metavar="K")
    p.add_argument(
        "--percentile",
        type=float,
        default=98.0,
        help="Порог: |curl| выше этого перцентиля (98 = реже точек, 95 = больше)",
    )
    p.add_argument("--dilate", type=int, default=5, help="Расширение маски (шагов соседства)")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    highlight_vortex_cores(
        args.ovf,
        z_slice=args.z_slice,
        percentile=args.percentile,
        dilate=args.dilate,
        save=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
