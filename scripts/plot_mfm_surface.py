"""
Одна MFM-карта MuMax3 как 3D-поверхность: x, y в нм, z — рельеф в тех же нм (масштаб видимый).

Примеры:
  python scripts/plot_mfm_surface.py simulations/NiFeGaCo_austenite.out/MFM_austenite_lift50nm.ovf -o mfm_surface.png
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
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Нужен matplotlib: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from e


def plot_mfm_surface(
    ovf_path: Path,
    *,
    z_slice_ovf: int = -1,
    subsample: int | None = None,
    save: Path | None = None,
    dpi: int = 200,
    cmap: str = "gray",
    relief_frac: float = 0.12,
    center: bool = True,
) -> None:
    """
    relief_frac — амплитуда рельефа как доля min(Lx,Ly) по z (в нм), чтобы z был сопоставим с xy.
    """
    ovf_path = ovf_path.resolve()
    if not ovf_path.is_file():
        print(f"Нет файла: {ovf_path}", file=sys.stderr)
        raise SystemExit(1)

    d, meta = read_ovf_binary4(ovf_path)
    if meta["valuedim"] != 1:
        print(f"Ожидался скаляр MFM (valuedim=1), сейчас {meta['valuedim']}", file=sys.stderr)
        raise SystemExit(1)

    zi = _z_index(meta["znodes"], z_slice_ovf)
    field = np.asarray(d[zi, :, :, 0], dtype=np.float64)
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

    nx, ny = meta["xnodes"], meta["ynodes"]
    xi = np.linspace(meta["xmin"] * 1e9, meta["xmax"] * 1e9, nx)
    yi = np.linspace(meta["ymin"] * 1e9, meta["ymax"] * 1e9, ny)
    X, Y = np.meshgrid(xi, yi, indexing="xy")

    f = field.copy()
    if center:
        f = f - np.mean(f)
    fmin, fmax = float(np.min(f)), float(np.max(f))
    span = max(fmax - fmin, 1e-30)
    fn = (f - fmin) / span

    lx = (meta["xmax"] - meta["xmin"]) * 1e9
    ly = (meta["ymax"] - meta["ymin"]) * 1e9
    lateral = min(lx, ly)
    z_amp = relief_frac * lateral
    Z = z_amp * (fn - 0.5) * 2.0

    if subsample is None:
        subsample = max(1, int(max(nx, ny) / 160))
    rs = max(1, subsample)
    Xs = X[::rs, ::rs]
    Ys = Y[::rs, ::rs]
    Zs = Z[::rs, ::rs]
    sub = f[::rs, ::rs]

    fig = plt.figure(figsize=(10, 7), dpi=100)
    ax = fig.add_subplot(projection="3d")

    smin, smax = float(np.min(sub)), float(np.max(sub))
    if smax <= smin + 1e-18:
        rgba = np.full((sub.shape[0], sub.shape[1], 4), 0.5)
        rgba[..., 3] = 1.0
    else:
        norm = mpl.colors.Normalize(vmin=smin, vmax=smax)
        cm = plt.get_cmap(cmap)
        rgba = cm(norm(sub))

    fc = (
        rgba[:-1, :-1, :]
        + rgba[1:, :-1, :]
        + rgba[:-1, 1:, :]
        + rgba[1:, 1:, :]
    ) * 0.25

    ax.plot_surface(
        Xs,
        Ys,
        Zs,
        rstride=1,
        cstride=1,
        facecolors=fc,
        linewidth=0.08,
        antialiased=True,
        shade=False,
    )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("рельеф MFM (нм)")
    ax.set_title(ovf_path.name)
    if smax > smin + 1e-18:
        m = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(smin, smax), cmap=cmap)
        m.set_array([])
        fig.colorbar(m, ax=ax, shrink=0.55, label="MFM (усл. ед.)")

    dz = float(np.ptp(Zs))
    dz = max(dz, lateral * 1e-4)
    ax.set_box_aspect((lx, ly, dz))

    ax.view_init(elev=28, azim=-135)
    ax.set_xlim(float(np.nanmin(Xs)), float(np.nanmax(Xs)))
    ax.set_ylim(float(np.nanmin(Ys)), float(np.nanmax(Ys)))
    z0, z1 = float(np.nanmin(Zs)), float(np.nanmax(Zs))
    if not np.isfinite(z0) or not np.isfinite(z1) or z0 >= z1:
        z0, z1 = -1.0, 1.0
    ax.set_zlim(z0, z1)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Сохранено: {save}")
    if mpl.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="3D-поверхность по одному MFM OVF")
    p.add_argument("ovf", type=Path, help="Путь к MFM_*.ovf")
    p.add_argument("-o", "--output", type=Path, default=None, help="PNG")
    p.add_argument("--z-slice", type=int, default=-1, metavar="K")
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--cmap", type=str, default="gray")
    p.add_argument(
        "--relief-frac",
        type=float,
        default=0.12,
        help="Амплитуда рельефа как доля min(Lx,Ly) (по умолчанию 0.12)",
    )
    p.add_argument("--no-center", action="store_true")
    args = p.parse_args()

    plot_mfm_surface(
        args.ovf,
        z_slice_ovf=args.z_slice,
        subsample=args.subsample,
        save=args.output,
        dpi=args.dpi,
        cmap=args.cmap,
        relief_frac=args.relief_frac,
        center=not args.no_center,
    )


if __name__ == "__main__":
    main()
