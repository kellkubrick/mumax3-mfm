"""
Три ортогональных среза m_z внутри объёма образца (один 3D-график).

Читает скалярный OVF (например mz_*_final.ovf, valuedim=1).

Пример:
  python scripts/plot_mz_volume_slices.py simulations/NiFeGaCo_austenite.out/mz_austenite_final.ovf -o mz_volume.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_mumax_output import read_ovf_binary4  # noqa: E402

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Нужен matplotlib: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from e


def _facecolors_from_field(
    field: np.ndarray, cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize
) -> np.ndarray:
    """field shape (a, b) -> facecolors (a-1, b-1, 4)."""
    rgba = cmap(norm(field))
    return (
        rgba[:-1, :-1, :]
        + rgba[1:, :-1, :]
        + rgba[:-1, 1:, :]
        + rgba[1:, 1:, :]
    ) * 0.25


def plot_mz_volume_slices(
    ovf_path: Path,
    *,
    save: Path | None = None,
    dpi: int = 220,
    cmap: str = "RdBu_r",
    alpha: float = 0.92,
) -> None:
    ovf_path = ovf_path.resolve()
    if not ovf_path.is_file():
        print(f"Нет файла: {ovf_path}", file=sys.stderr)
        raise SystemExit(1)

    d, meta = read_ovf_binary4(ovf_path)
    if meta["valuedim"] != 1:
        print(f"Ожидался скаляр mz (valuedim=1), сейчас {meta['valuedim']}", file=sys.stderr)
        raise SystemExit(1)

    vol = np.asarray(d[:, :, :, 0], dtype=np.float64)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    nz, ny, nx = vol.shape
    x_nm = np.linspace(meta["xmin"] * 1e9, meta["xmax"] * 1e9, nx)
    y_nm = np.linspace(meta["ymin"] * 1e9, meta["ymax"] * 1e9, ny)
    z_nm = np.linspace(meta["zmin"] * 1e9, meta["zmax"] * 1e9, nz)

    ix, iy, iz = nx // 2, ny // 2, nz // 2

    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    cm = plt.get_cmap(cmap)

    fig = plt.figure(figsize=(11, 9), dpi=100)
    ax = fig.add_subplot(projection="3d")

    # Плоскость z = const (горизонтальный срез)
    X_xy, Y_xy = np.meshgrid(x_nm, y_nm, indexing="xy")
    Z_xy = np.full_like(X_xy, z_nm[iz], dtype=float)
    slab_z = vol[iz, :, :]
    fc_z = _facecolors_from_field(slab_z, cm, norm)
    ax.plot_surface(
        X_xy,
        Y_xy,
        Z_xy,
        facecolors=fc_z,
        rstride=1,
        cstride=1,
        shade=False,
        linewidth=0.02,
        antialiased=True,
        alpha=alpha,
    )

    # Плоскость y = const (передний/задний срез)
    X_xz = np.broadcast_to(x_nm, (nz, nx))
    Z_xz = np.broadcast_to(z_nm[:, np.newaxis], (nz, nx))
    Y_xz = np.full((nz, nx), y_nm[iy], dtype=float)
    slab_y = vol[:, iy, :]
    fc_y = _facecolors_from_field(slab_y, cm, norm)
    ax.plot_surface(
        X_xz,
        Y_xz,
        Z_xz,
        facecolors=fc_y,
        rstride=1,
        cstride=1,
        shade=False,
        linewidth=0.02,
        antialiased=True,
        alpha=alpha,
    )

    # Плоскость x = const
    Y_yz = np.broadcast_to(y_nm, (nz, ny))
    Z_yz = np.broadcast_to(z_nm[:, np.newaxis], (nz, ny))
    X_yz = np.full((nz, ny), x_nm[ix], dtype=float)
    slab_x = vol[:, :, ix]
    fc_x = _facecolors_from_field(slab_x, cm, norm)
    ax.plot_surface(
        X_yz,
        Y_yz,
        Z_yz,
        facecolors=fc_x,
        rstride=1,
        cstride=1,
        shade=False,
        linewidth=0.02,
        antialiased=True,
        alpha=alpha,
    )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("z (nm)")
    ax.set_title(f"{ovf_path.name}\nсрезы: x={x_nm[ix]:.0f}, y={y_nm[iy]:.0f}, z={z_nm[iz]:.0f} nm")

    lx = (meta["xmax"] - meta["xmin"]) * 1e9
    ly = (meta["ymax"] - meta["ymin"]) * 1e9
    lz = (meta["zmax"] - meta["zmin"]) * 1e9
    ax.set_box_aspect((lx, ly, lz))
    ax.view_init(elev=20, azim=-60)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.55, label=r"$m_z$", pad=0.08)

    fig.tight_layout()
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Сохранено: {save}")
    if mpl.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="m_z: три ортогональных среза в 3D")
    p.add_argument("ovf", type=Path, help="mz_*.ovf (valuedim=1)")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--cmap", default="RdBu_r")
    p.add_argument("--alpha", type=float, default=0.92)
    args = p.parse_args()
    plot_mz_volume_slices(args.ovf, save=args.output, dpi=args.dpi, cmap=args.cmap, alpha=args.alpha)


if __name__ == "__main__":
    main()
