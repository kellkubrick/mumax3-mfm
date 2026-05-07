"""
3D-поверхности MFM по нескольким высотам зонда (MFMLift).

В MuMax3 у каждого MFMLift — своя отдельная 2D-карта (один OVF = один «снимок»
на фиксированной высоте). Это нормально: так устроен вывод MFM.

Чтобы увидеть, как меняется MFM с высотой, в .mx3 должно быть несколько
saveas(MFM, ...) с разным MFMLift (как в NiFeGaCo_austenite: 30, 50, 80 nm).
Тогда этот скрипт кладёт несколько слоёв в 3D: по z — физическая высота lift
плюс небольшой рельеф по сигналу на каждом слое.

Одна только карта (один lift) для «стека по высоте» недостаточна — используйте
plot_mfm_surface.py для одной поверхности-рельефа.

Пример:
  python scripts/plot_mfm_3d.py simulations/NiFeGaCo_austenite.out -o mfm_3d.png
"""

from __future__ import annotations

import argparse
import re
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


def _lift_nm_from_stem(stem: str) -> float | None:
    m = re.search(r"(?:^|_)lift[_]?(\d+)\s*nm", stem, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"lift(\d+)nm", stem, re.I)
    if m:
        return float(m.group(1))
    return None


def _collect_mfm_layers(outdir: Path) -> list[tuple[float, Path]]:
    outdir = outdir.resolve()
    items: list[tuple[float, Path]] = []
    for p in sorted(outdir.glob("MFM_*.ovf")):
        h = _lift_nm_from_stem(p.stem)
        if h is not None:
            items.append((h, p))
    for p in sorted(outdir.glob("lift_*.ovf")):
        h = _lift_nm_from_stem(p.stem)
        if h is not None:
            items.append((h, p))
    items.sort(key=lambda t: t[0])
    return items


def plot_mfm_3d(
    outdir: Path,
    *,
    z_slice_ovf: int = -1,
    subsample: int | None = None,
    save: Path | None = None,
    dpi: int = 200,
    cmap: str = "gray",
    relief_nm: float | None = None,
) -> None:
    layers = _collect_mfm_layers(outdir)
    if len(layers) < 2:
        print(
            "Нужно минимум два файла MFM с разными высотами "
            "(MFM_*lift*nm*.ovf или lift_*nm*.ovf).",
            file=sys.stderr,
        )
        if len(layers) == 1:
            print(f"Найден только один: {layers[0][1]}", file=sys.stderr)
        raise SystemExit(1)

    fields: list[tuple[float, np.ndarray, dict]] = []
    for h_nm, path in layers:
        d, meta = read_ovf_binary4(path)
        if meta["valuedim"] != 1:
            print(f"Пропуск {path.name}: ожидался скаляр (valuedim=1).", file=sys.stderr)
            continue
        zi = _z_index(meta["znodes"], z_slice_ovf)
        f = np.asarray(d[zi, :, :, 0], dtype=np.float64)
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        fields.append((h_nm, f, meta))

    if len(fields) < 2:
        raise SystemExit(1)

    global_min = min(f.min() for _, f, _ in fields)
    global_max = max(f.max() for _, f, _ in fields)
    if global_max <= global_min:
        global_max = global_min + 1e-30

    meta0 = fields[0][2]
    nx, ny = meta0["xnodes"], meta0["ynodes"]
    xi = np.linspace(meta0["xmin"] * 1e9, meta0["xmax"] * 1e9, nx)
    yi = np.linspace(meta0["ymin"] * 1e9, meta0["ymax"] * 1e9, ny)
    X, Y = np.meshgrid(xi, yi, indexing="xy")

    hs = [t[0] for t in fields]
    dh = min(hs[i + 1] - hs[i] for i in range(len(hs) - 1)) if len(hs) > 1 else 30.0
    if relief_nm is None:
        relief_nm = max(3.0, 0.12 * dh)

    if subsample is None:
        subsample = max(1, int(max(nx, ny) / 100))
    rs = max(1, subsample)

    lx = (meta0["xmax"] - meta0["xmin"]) * 1e9
    ly = (meta0["ymax"] - meta0["ymin"]) * 1e9

    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(projection="3d")
    norm = mpl.colors.Normalize(vmin=global_min, vmax=global_max)
    cm = plt.get_cmap(cmap)

    zmin_all = 1e30
    zmax_all = -1e30

    for h_nm, field, _meta in fields:
        fn = (field - global_min) / (global_max - global_min)
        Zsurf = h_nm + relief_nm * (fn - 0.5) * 2.0
        Xs = X[::rs, ::rs]
        Ys = Y[::rs, ::rs]
        Zs = Zsurf[::rs, ::rs]
        sub = field[::rs, ::rs]
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
            linewidth=0.05,
            antialiased=True,
            shade=False,
        )
        zmin_all = min(zmin_all, float(np.min(Zs)))
        zmax_all = max(zmax_all, float(np.max(Zs)))

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("MFMLift + рельеф (нм)")
    ax.set_title("MFM: слои при разных высотах зонда")
    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.55, label="MFM (усл. ед.)")

    dz = max(zmax_all - zmin_all, relief_nm, 1.0)
    ax.set_box_aspect((lx, ly, dz))
    ax.view_init(elev=22, azim=-130)
    x0 = float(meta0["xmin"] * 1e9)
    x1 = float(meta0["xmax"] * 1e9)
    y0 = float(meta0["ymin"] * 1e9)
    y1 = float(meta0["ymax"] * 1e9)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if np.isfinite(zmin_all) and np.isfinite(zmax_all) and zmin_all < zmax_all:
        ax.set_zlim(zmin_all, zmax_all)

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
    p = argparse.ArgumentParser(description="3D: несколько MFM по разным MFMLift")
    p.add_argument("outdir", type=Path, help="Каталог *.out")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--z-slice", type=int, default=-1, metavar="K")
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--cmap", type=str, default="gray")
    p.add_argument("--relief-nm", type=float, default=None, metavar="R")
    args = p.parse_args()
    if not args.outdir.is_dir():
        print(f"Нет каталога: {args.outdir}", file=sys.stderr)
        raise SystemExit(1)

    plot_mfm_3d(
        args.outdir.resolve(),
        z_slice_ovf=args.z_slice,
        subsample=args.subsample,
        save=args.output,
        dpi=args.dpi,
        cmap=args.cmap,
        relief_nm=args.relief_nm,
    )


if __name__ == "__main__":
    main()
