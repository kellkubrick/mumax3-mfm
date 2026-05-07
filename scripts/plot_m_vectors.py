"""
Карта намагниченности со стрелками (m_x, m_y).

  --style mumax  — как в веб-GUI mumax (127.0.0.1:35367): цвет от направления в плоскости xy,
                   чёрные стрелки.
  --style mz     — фон по m_z (RdBu), стрелки с цветом по m_z.

Примеры:
  python scripts/plot_m_vectors.py simulations/NiFeGaCo_austenite.out --style mumax -o m_gui.png
  python scripts/plot_m_vectors.py ... --sparse 2        # ещё в ~4 раза меньше стрелок (при авто-stride)
  python scripts/plot_m_vectors.py ... --stride 12      # явный шаг сетки стрелок
  python scripts/plot_m_vectors.py simulations/mfm_vortex.out -o m_classic.png --style mz
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
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe
except ImportError as e:
    print("Нужен matplotlib: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from e


def _find_vector_ovf(target: Path) -> Path:
    if target.is_file():
        return target
    if not target.is_dir():
        raise FileNotFoundError(target)
    for pattern in ("m000000.ovf",):
        p = target / pattern
        if p.is_file():
            return p
    for p in sorted(target.glob("m_*_final.ovf")):
        if not p.name.startswith(("mx_", "my_", "mz_")):
            return p
    for p in sorted(target.glob("m_*.ovf")):
        if not p.name.startswith(("mx_", "my_", "mz_")):
            return p
    raise FileNotFoundError(
        f"В {target} нет векторного m (*.ovf с полным m): "
        "ожидались m000000.ovf или m_*_final.ovf (не mx/my/mz)."
    )


def _rgb_mumax_gui(mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> np.ndarray:
    """
    Цвет фона в духе mumax: оттенок = азимут (mx, my) в плоскости;
    +x → красный, −x → циан; насыщенность от |m_xy|, яркость почти единица.
    """
    h = np.mod(-np.arctan2(my, mx) / (2 * np.pi), 1.0)
    mxy = np.hypot(mx, my)
    s = np.clip(mxy**0.85, 0.0, 1.0)
    # там, где m почти по z, даём лёгкий оттенок по mz, чтобы не уходить в серый шум
    s = np.maximum(s, 0.12 * np.clip(np.abs(mz), 0, 1))
    s = np.clip(s, 0, 1)
    v = np.clip(0.25 + 0.75 * np.sqrt(np.clip(mx * mx + my * my + mz * mz, 0, 1)), 0, 1)
    hsv = np.stack([h, s, v], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def plot_magnetization_vectors(
    ovf_path: Path,
    *,
    z_slice: int = -1,
    stride: int | None = None,
    save: Path | None = None,
    dpi: int = 220,
    unit_inplane: bool = False,
    streamlines: bool = False,
    style: str = "mz",
    sparse_mult: int = 1,
) -> None:
    data, meta = read_ovf_binary4(ovf_path)
    if meta["valuedim"] < 3:
        raise ValueError(f"{ovf_path.name}: нужен векторный OVF (valuedim=3), сейчас {meta['valuedim']}")

    zi = _z_index(meta["znodes"], z_slice)
    sl = np.asarray(data[zi, :, :, :3])
    mx = sl[:, :, 0].astype(np.float64)
    my = sl[:, :, 1].astype(np.float64)
    mz = sl[:, :, 2].astype(np.float64)

    nx, ny = meta["xnodes"], meta["ynodes"]
    xmin, xmax = meta["xmin"], meta["xmax"]
    ymin, ymax = meta["ymin"], meta["ymax"]
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    xc = xmin + (np.arange(nx) + 0.5) * dx
    yc = ymin + (np.arange(ny) + 0.5) * dy
    XX, YY = np.meshgrid(xc, yc, indexing="xy")

    if stride is None:
        # Меньше делитель → больше stride → реже стрелки (раньше /56 было густо).
        base = max(2, int(max(nx, ny) / 22))
        stride = max(1, base * max(1, sparse_mult))

    if unit_inplane:
        norm = np.hypot(mx, my)
        norm = np.maximum(norm, 1e-9)
        mx = mx / norm
        my = my / norm

    # стиль
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "#f8f9fa",
        }
    )

    fig_w = min(12, 4 + nx / stride * 0.09)
    fig_h = min(11, 3.8 + ny / stride * 0.09)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

    ext_nm = (xmin * 1e9, xmax * 1e9, ymin * 1e9, ymax * 1e9)
    use_mumax = style.lower() in ("mumax", "gui", "web")

    if use_mumax:
        rgb = _rgb_mumax_gui(mx, my, mz)
        ax.imshow(
            rgb,
            origin="lower",
            extent=ext_nm,
            aspect="equal",
            interpolation="bilinear",
            zorder=1,
        )
    else:
        im = ax.imshow(
            mz,
            origin="lower",
            extent=ext_nm,
            aspect="equal",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="bilinear",
            zorder=1,
        )

    XS = XX[::stride, ::stride] * 1e9
    YS = YY[::stride, ::stride] * 1e9
    US = mx[::stride, ::stride]
    VS = my[::stride, ::stride]

    mxy_max = float(np.nanmax(np.hypot(mx, my))) or 1.0
    # Длина стрелки в нм ≈ hypot(U,V) / scale (scale_units='xy'). Раньше scale был перевёрнут —
    # получались микроскопические стрелки (визуально «точки»).
    ref_len_nm = max(stride * max(dx, dy) * 1e9 * 0.75, 1e-6)
    scale = max(mxy_max, 0.08) / ref_len_nm

    if use_mumax:
        w = 0.004 * (96 / max(nx // stride, 24))
        q = ax.quiver(
            XS,
            YS,
            US,
            VS,
            color="#0a0a0a",
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=w,
            headwidth=4.0,
            headlength=4.5,
            headaxislength=3.8,
            minshaft=0.5,
            pivot="mid",
            zorder=3,
        )
    else:
        CS = mz[::stride, ::stride]
        w = 0.0045 * (96 / max(nx // stride, 24))
        q = ax.quiver(
            XS,
            YS,
            US,
            VS,
            CS,
            cmap="RdYlBu_r",
            clim=(-1, 1),
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=w,
            headwidth=4.0,
            headlength=4.5,
            headaxislength=3.8,
            minshaft=0.5,
            pivot="mid",
            zorder=3,
            edgecolors="white",
            linewidths=0.35,
        )
        q.set_path_effects([pe.withStroke(linewidth=1.1, foreground="white", alpha=0.55)])

    if streamlines and not use_mumax:
        ax.streamplot(
            XX * 1e9,
            YY * 1e9,
            mx,
            my,
            color=np.clip(mz, -1, 1),
            cmap="binary",
            linewidth=0.6,
            density=1.1,
            arrowsize=0.8,
            zorder=2,
        )

    if not use_mumax:
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=r"$m_z$")
        cb.ax.tick_params(labelsize=9)

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    st_label = "mumax-GUI" if use_mumax else "m_z фон"
    ax.set_title(
        f"{ovf_path.name} — {st_label}, z={zi}/{meta['znodes'] - 1}, stride={stride}",
        fontsize=11,
    )
    ax.set_xlim(ext_nm[0], ext_nm[1])
    ax.set_ylim(ext_nm[2], ext_nm[3])

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
    p = argparse.ArgumentParser(description="Векторная карта m (OVF mumax3)")
    p.add_argument("target", type=Path, help="Каталог *.out или путь к векторному m.ovf")
    p.add_argument("--z-slice", type=int, default=-1, metavar="K", help="Слой z (-1 = середина)")
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        metavar="N",
        help="Брать каждую N-ю ячейку для стрелки (больше N → реже). По умолчанию авто от размера сетки.",
    )
    p.add_argument(
        "--sparse",
        type=int,
        default=1,
        metavar="K",
        help="При авто-stride: умножить шаг на K (K=2 ≈ в 4 раза меньше стрелок по площади). Игнорируется, если задан --stride.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Сохранить PNG (иначе только окно matplotlib)",
    )
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument(
        "--unit-inplane",
        action="store_true",
        help="Нормировать m_x, m_y по длине (только направление в плоскости)",
    )
    p.add_argument("--streamlines", action="store_true", help="Добавить линии тока (только при --style mz)")
    p.add_argument(
        "--style",
        choices=("mz", "mumax", "gui", "web"),
        default="mumax",
        help="mumax/gui/web — как в браузере mumax (цвет по m_xy, чёрные стрелки); mz — фон RdBu по m_z",
    )
    args = p.parse_args()

    try:
        ovf = _find_vector_ovf(args.target.resolve())
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from e

    sm = args.sparse if args.stride is None else 1
    if sm < 1:
        print("--sparse должен быть >= 1", file=sys.stderr)
        raise SystemExit(1)

    plot_magnetization_vectors(
        ovf,
        z_slice=args.z_slice,
        stride=args.stride,
        save=args.output,
        dpi=args.dpi,
        unit_inplane=args.unit_inplane,
        streamlines=args.streamlines,
        style=args.style,
        sparse_mult=sm,
    )


if __name__ == "__main__":
    main()
