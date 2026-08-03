#!/usr/bin/env python
"""One-shot: password-authenticate to the GB10 box, install our public key,
and report what hardware/software is actually there.

    python scripts/gb10_setup.py --host 100.64.3.0 --user sit

After this runs, plain `ssh sit@<host>` works without a password and the rest
of the tooling (rsync, torchrun) can be driven normally.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import paramiko

PROBE = r"""
echo "== host =="; hostname; uname -m; cat /etc/os-release 2>/dev/null | head -2
echo "== cpu/mem =="; nproc; free -g 2>/dev/null | head -2
echo "== gpu =="; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4
echo "== cuda =="; (nvcc --version 2>/dev/null | tail -2) || echo "no nvcc"
echo "== python =="; for p in python3 python; do command -v $p >/dev/null && $p -V; done
echo "== torch =="; python3 -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 | tail -1
echo "== disk =="; df -h /home 2>/dev/null | tail -1; df -h / | tail -1
echo "== containers =="; command -v docker >/dev/null && docker --version || echo "no docker"
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="100.64.3.0")
    ap.add_argument("--user", default="sit")
    ap.add_argument("--password", default=os.environ.get("GB10_PASSWORD"))
    ap.add_argument("--pubkey", default=str(Path.home() / ".ssh" / "id_ed25519.pub"))
    ap.add_argument("--no-install-key", action="store_true")
    args = ap.parse_args()

    if not args.password:
        sys.exit("need --password or GB10_PASSWORD")

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(args.host, username=args.user, password=args.password,
                timeout=20, look_for_keys=False, allow_agent=False)
    print(f"connected to {args.user}@{args.host}\n")

    if not args.no_install_key:
        pub = Path(args.pubkey).read_text(encoding="utf-8").strip()
        cmd = (
            "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
            f"grep -qxF '{pub}' ~/.ssh/authorized_keys 2>/dev/null || "
            f"echo '{pub}' >> ~/.ssh/authorized_keys; "
            "chmod 600 ~/.ssh/authorized_keys && echo KEY_OK"
        )
        _, out, err = cli.exec_command(cmd)
        print("key install:", out.read().decode().strip() or err.read().decode().strip())

    _, out, err = cli.exec_command(PROBE)
    print(out.read().decode())
    e = err.read().decode().strip()
    if e:
        print("[stderr]", e[:500])
    cli.close()


if __name__ == "__main__":
    main()
