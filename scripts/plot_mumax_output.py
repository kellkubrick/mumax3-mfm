"""
Визуализация OVF из каталога вывода mumax3 (m, MFM).

Пример:
  python scripts/plot_mumax_output.py simulations/mfm_vortex.out
  python scripts/plot_mumax_output.py simulations/NiFeGaCo_martensite.out --z-slice -1
  python scripts/plot_mumax_output.py simulations/mfm_vortex.out --save

Для 3D-симуляций (Nz > 1) по умолчанию берётся средний слой по z (--z-slice -1).
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Нужен matplotlib: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from e


def _parse_ovf_header(text: str) -> dict:
    def grab_int(key: str) -> int:
        m = re.search(rf"^#\s*{key}:\s*(\d+)\s*$", text, re.MULTILINE)
        if not m:
            raise ValueError(f"OVF: нет поля {key}")
        return int(m.group(1))

    def grab_float(key: str) -> float:
        m = re.search(rf"^#\s*{key}:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$", text, re.MULTILINE)
        if not m:
            raise ValueError(f"OVF: нет поля {key}")
        return float(m.group(1))

    return {
        "xnodes": grab_int("xnodes"),
        "ynodes": grab_int("ynodes"),
        "znodes": grab_int("znodes"),
        "valuedim": grab_int("valuedim"),
        "xmin": grab_float("xmin"),
        "xmax": grab_float("xmax"),
        "ymin": grab_float("ymin"),
        "ymax": grab_float("ymax"),
        "zmin": grab_float("zmin"),
        "zmax": grab_float("zmax"),
    }


def read_ovf_binary4(path: Path) -> tuple[np.ndarray, dict]:
    raw = path.read_bytes()
    key = b"# Begin: Data Binary 4"
    i = raw.find(key)
    if i < 0:
        raise ValueError(f"{path}: ожидался '# Begin: Data Binary 4'")

    j = i + len(key)
    if raw[j : j + 2] == b"\r\n":
        j += 2
    elif raw[j : j + 1] == b"\n":
        j += 1
    else:
        raise ValueError(f"{path}: неверное окончание строки Data")

    ctrl = struct.unpack_from("<f", raw, j)[0]
    if not np.isclose(ctrl, 1234567.0, rtol=0, atol=1.0):
        raise ValueError(f"{path}: неверный контрольный float ({ctrl}), ожидалось 1234567.0")

    j += 4
    header_text = raw[:i].decode("ascii", errors="strict")
    meta = _parse_ovf_header(header_text)

    nx, ny, nz, vd = meta["xnodes"], meta["ynodes"], meta["znodes"], meta["valuedim"]
    n = nx * ny * nz * vd
    need = j + 4 * n
    if len(raw) < need:
        raise ValueError(f"{path}: обрезанные данные (нужно {need} байт, есть {len(raw)})")

    arr = np.frombuffer(raw, dtype="<f4", count=n, offset=j)
    # Порядок OOMMF/mumax: x быстрее всего, затем y, z; у каждой ячейки valuedim компонент подряд.
    data = arr.reshape((nz, ny, nx, vd), order="C")
    return data, meta


def _z_index(nz: int, z_slice: int) -> int:
    if nz <= 1:
        return 0
    if z_slice < 0:
        return nz // 2
    return min(z_slice, nz - 1)


def _extent_xy_nm(meta: dict) -> tuple[float, float, float, float]:
    return (
        meta["xmin"] * 1e9,
        meta["xmax"] * 1e9,
        meta["ymin"] * 1e9,
        meta["ymax"] * 1e9,
    )


def _slice_scalar_or_mz(data: np.ndarray, meta: dict, z_slice: int) -> np.ndarray:
    """Один слой по z: valuedim 1 или mz из valuedim 3."""
    zi = _z_index(meta["znodes"], z_slice)
    vd = meta["valuedim"]
    if vd == 1:
        return np.asarray(data[zi, :, :, 0])
    if vd >= 3:
        return np.asarray(data[zi, :, :, 2])
    raise ValueError(f"valuedim={vd} не поддержан для карты m")


def plot_outdir_named(outdir: Path, save: bool, z_slice: int) -> None:
    """NiFeGaCo и др.: mz_*_final.ovf + MFM_*.ovf (или только MFM)."""
    outdir = outdir.resolve()
    mz_data = None
    mz_meta = None
    mz_title = ""
    mz_paths = sorted(outdir.glob("mz_*_final.ovf"))
    m_vec_paths = [
        p
        for p in sorted(outdir.glob("m_*_final.ovf"))
        if not p.name.startswith(("mx_", "my_", "mz_"))
    ]
    mfm_paths = sorted(outdir.glob("MFM_*.ovf"))

    mz_data = None
    mz_meta = None
    if mz_paths:
        p = mz_paths[0]
        mz_data, mz_meta = read_ovf_binary4(p)
        mz_title = p.stem
    elif m_vec_paths:
        p = m_vec_paths[0]
        mz_data, mz_meta = read_ovf_binary4(p)
        mz_title = f"{p.stem} ($m_z$, z-слой)"
    elif not mfm_paths:
        print(
            "Нет mz_*_final.ovf, m_*_final.ovf (вектор) и MFM_*.ovf — "
            "попробуйте каталог mfm_vortex.out или проверьте имена файлов.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    ext = _extent_xy_nm(mz_meta if mz_meta is not None else read_ovf_binary4(mfm_paths[0])[1])

    n_mfm = len(mfm_paths)
    n_panels = (1 if (mz_data is not None) else 0) + n_mfm
    if n_panels == 0:
        raise SystemExit(1)

    cols = min(3, max(2, n_panels))
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4 * rows), squeeze=False)

    idx = 0
    if mz_data is not None:
        zi = _z_index(mz_meta["znodes"], z_slice)
        sl = _slice_scalar_or_mz(mz_data, mz_meta, z_slice)
        ax = axes[idx // cols, idx % cols]
        im = ax.imshow(
            np.flipud(sl.T),
            extent=(ext[0], ext[1], ext[2], ext[3]),
            aspect="equal",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"{mz_title}\n(z={zi}/{mz_meta['znodes'] - 1})")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        idx += 1

    for p in mfm_paths:
        d, meta = read_ovf_binary4(p)
        if meta["znodes"] != 1:
            print(
                f"Предупреждение: {p.name} имеет znodes={meta['znodes']}; "
                f"MFM обычно 2D, беру слой z={_z_index(meta['znodes'], z_slice)}.",
                file=sys.stderr,
            )
        zi = _z_index(meta["znodes"], z_slice)
        sl = np.asarray(d[zi, :, :, 0])
        ax = axes[idx // cols, idx % cols]
        im = ax.imshow(
            np.flipud(sl.T),
            extent=(ext[0], ext[1], ext[2], ext[3]),
            aspect="equal",
            cmap="gray",
        )
        ax.set_title(p.stem)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        idx += 1

    for k in range(idx, rows * cols):
        axes[k // cols, k % cols].set_visible(False)

    fig.suptitle(outdir.name, fontsize=12)
    fig.tight_layout()
    if save:
        out_png = outdir / "mumax_preview.png"
        fig.savefig(out_png, dpi=150)
        print(f"Сохранено: {out_png}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def plot_outdir(outdir: Path, save: bool, z_slice: int = 0) -> None:
    outdir = outdir.resolve()
    m_path = outdir / "m000000.ovf"
    if not m_path.is_file():
        print(f"Нет {m_path.name} в {outdir}", file=sys.stderr)
        raise SystemExit(1)

    m_data, m_meta = read_ovf_binary4(m_path)
    zi = _z_index(m_meta["znodes"], z_slice)
    mz = m_data[zi, :, :, 2]

    lifts = sorted(outdir.glob("lift_*.ovf"))
    if not lifts:
        print("Нет lift_*.ovf (MFM)", file=sys.stderr)
        raise SystemExit(1)

    mfm_stack = []
    titles = []
    for p in lifts:
        d, meta = read_ovf_binary4(p)
        zf = _z_index(meta["znodes"], z_slice)
        mfm_stack.append(d[zf, :, :, 0])
        titles.append(p.stem)

    ext = (
        m_meta["xmin"] * 1e9,
        m_meta["xmax"] * 1e9,
        m_meta["ymin"] * 1e9,
        m_meta["ymax"] * 1e9,
    )

    n_mfm = len(mfm_stack)
    cols = 2
    rows = int(np.ceil((1 + n_mfm) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), squeeze=False)

    ax_m = axes[0, 0]
    im0 = ax_m.imshow(
        np.flipud(mz.T),
        extent=(ext[0], ext[1], ext[2], ext[3]),
        aspect="equal",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    ax_m.set_title(rf"Релаксированное $m_z$ (z={zi}/{m_meta['znodes'] - 1})")
    ax_m.set_xlabel("x (nm)")
    ax_m.set_ylabel("y (nm)")
    fig.colorbar(im0, ax=ax_m, fraction=0.046, pad=0.04)

    for idx, (arr, name) in enumerate(zip(mfm_stack, titles)):
        r = (idx + 1) // cols
        c = (idx + 1) % cols
        ax = axes[r, c]
        im = ax.imshow(
            np.flipud(arr.T),
            extent=(ext[0], ext[1], ext[2], ext[3]),
            aspect="equal",
            cmap="gray",
        )
        ax.set_title(f"MFM ({name})")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k in range(1 + n_mfm, rows * cols):
        axes[k // cols, k % cols].set_visible(False)

    fig.suptitle(outdir.name, fontsize=12)
    fig.tight_layout()
    if save:
        out_png = outdir / "mumax_preview.png"
        fig.savefig(out_png, dpi=150)
        print(f"Сохранено: {out_png}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="OVF mumax3 → matplotlib")
    p.add_argument("outdir", type=Path, help="Каталог *.out (например simulations/mfm_vortex.out)")
    p.add_argument("--save", action="store_true", help="Сохранить mumax_preview.png в каталог вывода")
    p.add_argument(
        "--z-slice",
        type=int,
        default=-1,
        metavar="K",
        help="Индекс слоя по z для 3D (по умолчанию -1 = середина). Для Nz=1 игнорируется.",
    )
    args = p.parse_args()
    if not args.outdir.is_dir():
        print(f"Каталог не найден: {args.outdir}", file=sys.stderr)
        raise SystemExit(1)

    od = args.outdir.resolve()
    has_vortex = (od / "m000000.ovf").is_file()
    has_named = bool(list(od.glob("MFM_*.ovf"))) or bool(list(od.glob("mz_*_final.ovf")))

    if has_named:
        plot_outdir_named(od, args.save, args.z_slice)
    elif has_vortex:
        plot_outdir(od, args.save, args.z_slice)
    else:
        print(
            "Не удалось распознать набор OVF (нет m000000.ovf и нет MFM_*.ovf / mz_*_final.ovf).",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
