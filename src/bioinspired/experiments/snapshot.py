"""Snapshots de reproducibilidad (git/system/lockfile) compartidos por los runners."""

from __future__ import annotations

import hashlib
import platform
import socket
import subprocess
from pathlib import Path


def capture_repro_snapshots(run_dir: Path) -> None:
    """Escribe git.txt/system.txt/lockfile_hash.txt en ``run_dir`` (invariante R18).

    No escribe config.yaml: cada runner persiste su propio config self-describing.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_info = subprocess.check_output(
            ["git", "log", "-1", "--pretty=fuller"], stderr=subprocess.DEVNULL, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_info = "(git no disponible)"
    (run_dir / "git.txt").write_text(git_info)

    sys_info = (
        f"hostname: {socket.gethostname()}\n"
        f"platform: {platform.platform()}\n"
        f"python: {platform.python_version()}\n"
        f"cpu: {platform.processor()}\n"
        f"cwd: {Path.cwd()}\n"
    )
    (run_dir / "system.txt").write_text(sys_info)

    lock_path = Path("uv.lock")
    if lock_path.exists():
        digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        (run_dir / "lockfile_hash.txt").write_text(digest)


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _cpu_model() -> str:
    sistema = platform.system()
    if sistema == "Darwin":
        return _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    if sistema == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for linea in cpuinfo.read_text().splitlines():
                if linea.startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    return platform.processor()


def _physical_cores() -> int | None:
    if platform.system() == "Darwin":
        valor = _cmd(["sysctl", "-n", "hw.physicalcpu"])
        return int(valor) if valor and valor.isdigit() else None
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return None
    nucleos = psutil.cpu_count(logical=False)
    return int(nucleos) if nucleos else None


def machine_provenance() -> dict[str, object]:
    """Máquina, código y entorno de una corrida, para ``provenance.json``.

    Los tiempos absolutos sólo valen en la máquina que los midió: sin su
    identificación no se puede saber después dónde corrió cada dataset.
    """
    import os
    import sys

    lock_path = Path("uv.lock")
    return {
        "hostname": socket.gethostname(),
        # En macOS el hostname cambia con la red; ComputerName es el que fija el usuario.
        "computer_name": (
            _cmd(["scutil", "--get", "ComputerName"]) if platform.system() == "Darwin" else None
        )
        or socket.gethostname(),
        "platform": platform.platform(),
        "arch": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_physical_cores": _physical_cores(),
        "cpu_logical_cores": os.cpu_count(),
        "python": sys.version.split()[0],
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_cmd(["git", "status", "--porcelain", "--untracked-files=no"])),
        "uv_lock_sha256": (
            hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.exists() else None
        ),
    }


def same_machine(a: dict[str, object], b: dict[str, object]) -> bool:
    """True si dos procedencias describen el mismo hardware.

    Compara CPU, arquitectura y núcleos, no el hostname: en macOS ése cambia con
    la red y daría falsas alarmas. Distingue equipos con hardware distinto, que
    es lo que importa para no mezclar tiempos entre máquinas.
    """
    claves = ("cpu_model", "arch", "cpu_logical_cores")
    return all(a.get(k) == b.get(k) for k in claves)
