#!/usr/bin/env python3
"""Second "slot" for IMAGE runs: two emulator runs at once, tempo never in parallel.

A slot is a full copy of a variant directory (binary, libMoltenVK, _ShaderCache, _PipelineCache,
_SaveData) at <variant>-slot2, so the two processes share no cache or save state; the frame log and
the frame dump directory are separate per run. The slot is announced by kbench-results/GPU_SHARE-slot2
instead of GPU_LOCK.

Rule (the whole point): the slot may start only while the GPU_LOCK holder declares an image run, that
is its line ends with the word `image` (`rt-agent <pid> <date> image`), the holder is alive, and no
slot is running. A tempo run (frames per second, time) takes GPU_LOCK without `image` and must wait
until GPU_SHARE-slot2 is gone; the slot never starts under a tempo lock or without a lock.

  run_slot.py prepare --variant <dir> [--config <kbench cfg.json>] [--refresh]
  run_slot.py check                        exit 0 when a slot may start now, 1 otherwise (reason on stdout)
  run_slot.py run --config <slot cfg.json> [--dump-dir D] -- <kbench run args...>
  run_slot.py status | release

`run` takes GPU_SHARE-slot2, runs `python -m kbench run --config <cfg> <args>` from tools/bench, and
removes the file when kbench ends (also on Ctrl-C). Calibration still to do by the game agent: see
tools/run_slot.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GPU_LOCK = Path(os.environ.get("GPU_LOCK", REPO.parent / "kbench-results" / "GPU_LOCK"))
GPU_SHARE = Path(os.environ.get("GPU_SHARE_SLOT2", GPU_LOCK.parent / "GPU_SHARE-slot2"))
MIN_FREE_GB = 15


def parse_lock(text: str) -> dict:
    """`<owner> <pid> <date...> [image]` -> {owner, pid, image}; pid is None when absent/garbled."""
    words = text.split()
    pid = None
    if len(words) > 1 and words[1].isdigit():
        pid = int(words[1])
    return {"owner": words[0] if words else "", "pid": pid, "image": bool(words) and words[-1] == "image"}


def can_start(lock_text: str | None, share_text: str | None, pid_alive) -> tuple[bool, str]:
    """May the slot start? lock_text/share_text are file contents or None when the file is absent."""
    if share_text is not None:
        return False, f"slot already running: {share_text.strip()}"
    if lock_text is None:
        return False, "no GPU_LOCK: nothing to share (take the lock yourself, run alone)"
    lock = parse_lock(lock_text)
    if not lock["image"]:
        return False, f"GPU_LOCK holder did not declare an image run: {lock_text.strip()}"
    if lock["pid"] is None or not pid_alive(lock["pid"]):
        return False, f"GPU_LOCK holder is not alive: {lock_text.strip()}"
    return True, "ok"


def make_slot_config(cfg: dict, slot_dir: Path, dump_dir: Path | None) -> dict:
    """Config of the slot: emulator in the copy, own frame dump directory (unless the config sets one)."""
    out = dict(cfg)
    out["label"] = f"{cfg.get('label', 'cfg')}-slot2"
    out["emulator"] = str(slot_dir / Path(cfg["emulator"]).name)
    env = dict(cfg.get("env", {}))
    if dump_dir is not None:
        env["KYTY_FRAME_DUMP_DIR"] = str(dump_dir)
    out["env"] = env
    return out


def slot_dir_of(variant: Path) -> Path:
    return variant.parent / (variant.name + "-slot2")


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except FileNotFoundError:
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cmd_prepare(args) -> int:
    variant = Path(args.variant).resolve()
    if not (variant / "kyty_emulator").is_file():
        print(f"no kyty_emulator in {variant}", file=sys.stderr)
        return 2
    slot = slot_dir_of(variant)
    if slot.exists():
        if not args.refresh:
            print(f"{slot} exists (use --refresh to recopy)", file=sys.stderr)
            return 2
        shutil.rmtree(slot)
    free_gb = shutil.disk_usage(variant).free / 2**30
    if free_gb < MIN_FREE_GB:
        print(f"BLOCKED: {free_gb:.1f} GB free < {MIN_FREE_GB} GB (R-BEZ-9)", file=sys.stderr)
        return 3
    # The caches are the point of the copy; a running process may be writing them, so copy at rest.
    shutil.copytree(variant, slot, symlinks=True)
    print(f"slot: {slot}")
    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        # The emulator path of a kbench config is relative to the config file.
        base = Path(args.config).resolve().parent
        cfg["emulator"] = str((base / cfg["emulator"]).resolve())
        out = Path(args.config).with_name(Path(args.config).stem + "-slot2.json")
        out.write_text(json.dumps(make_slot_config(cfg, slot, None), indent=2) + "\n")
        print(f"config: {out}")
    return 0


def cmd_check(_args) -> int:
    ok, why = can_start(_read(GPU_LOCK), _read(GPU_SHARE), _alive)
    print(("OK " if ok else "NO ") + why)
    return 0 if ok else 1


def cmd_release(_args) -> int:
    text = _read(GPU_SHARE)
    if text is not None and f" {os.getpid()} " not in text:
        print(f"not ours, left in place: {text.strip()}")
        return 1
    GPU_SHARE.unlink(missing_ok=True)
    return 0


def cmd_status(_args) -> int:
    print("GPU_LOCK:", (_read(GPU_LOCK) or "-").strip())
    print("GPU_SHARE-slot2:", (_read(GPU_SHARE) or "-").strip())
    return 0


def cmd_run(args) -> int:
    ok, why = can_start(_read(GPU_LOCK), _read(GPU_SHARE), _alive)
    if not ok:
        print("NO " + why)
        return 1
    cfg = json.loads(Path(args.config).read_text())
    if not str(cfg.get("label", "")).endswith("-slot2"):
        print("config is not a slot config (label must end with -slot2); use `prepare --config`", file=sys.stderr)
        return 2
    try:
        fd = os.open(GPU_SHARE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("NO slot appeared meanwhile")
        return 1
    with os.fdopen(fd, "w") as f:
        f.write(f"slot2 {os.getpid()} {time.strftime('%Y-%m-%d %H:%M')} image\n")
    try:
        rest = [a for a in args.kbench_args if a != "--"]
        env = dict(os.environ)
        if args.dump_dir:
            env["KYTY_FRAME_DUMP_DIR"] = args.dump_dir
        py = REPO / "tools" / "bench" / ".venv" / "bin" / "python"
        cmd = [str(py), "-m", "kbench", "run", "--config", str(Path(args.config).resolve()), *rest]
        return subprocess.call(cmd, cwd=REPO / "tools" / "bench", env=env)
    finally:
        cmd_release(args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--variant", required=True)
    s.add_argument("--config")
    s.add_argument("--refresh", action="store_true")
    sub.add_parser("check")
    sub.add_parser("status")
    sub.add_parser("release")
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--dump-dir")
    r.add_argument("kbench_args", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    return {"prepare": cmd_prepare, "check": cmd_check, "status": cmd_status,
            "release": cmd_release, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
