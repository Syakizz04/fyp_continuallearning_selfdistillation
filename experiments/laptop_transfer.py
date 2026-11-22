"""
Copy what this project needs from the old laptop to this one, over the local
network. Standard library only, Python 3.8+, so it runs on the old laptop's
Python without installing anything.

On the OLD laptop (inside its copy of the project):

    python laptop_transfer.py serve --project .
    #  -> prints this laptop's address and a one-time token, then waits

On THIS laptop:

    python -m experiments.laptop_transfer fetch --host 192.168.1.23 --token <token>
    #  -> downloads into recovered/old_laptop/, verifies every file by sha256
    python -m experiments.recover_scan recovered/old_laptop

What is sent (and nothing else - the server only serves this list):
  * git: a bundle of every branch and commit (unpushed work included), the
    uncommitted diff, `git status`, and untracked source files;
  * data/: everything except m5_raw and _download_kunal (the Kaggle download,
    already here and rebuildable). This includes processed_m5 (v1), which
    cannot be rebuilt;
  * outputs/: checkpoints, result tables, logs and plots. Walk-state
    snapshots, lightning_logs and *.tmp are skipped, as are files over
    --max-mb (default 1500) unless --include-large is given. Skipped files
    are listed so nothing is dropped silently.

Safety: the server is read-only, serves only the listed files by index (no
paths from the client), and rejects any request without the token. It stops
when the fetch finishes or after --idle-minutes with no requests. Windows
Firewall will ask whether to allow Python on the old laptop the first time;
allow it on Private networks only. Use a home network, not public Wi-Fi.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

PORT = 8765
SKIP_DATA = {"m5_raw", "_download_kunal"}
SKIP_DIR_NAMES = {"lightning_logs", "__pycache__", ".ipynb_checkpoints", "node_modules", ".venv"}
CHUNK = 1 << 20


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def skip_dir(name: str) -> bool:
    return name in SKIP_DIR_NAMES or name.startswith("walk_state_")


def walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not skip_dir(d)]
        for fn in filenames:
            if not fn.endswith(".tmp"):
                yield Path(dirpath) / fn


def git(project: Path, *args: str):
    try:
        r = subprocess.run(["git", *args], cwd=project, capture_output=True, timeout=600)
        return r.returncode, r.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc).encode()


def build_list(project: Path, staging: Path, max_mb: float, include_large: bool):
    """-> (files: [(served_name, local_path)], skipped: [(name, reason)])"""
    files, skipped = [], []

    # git state first: the one thing GitHub may not already have.
    code, _ = git(project, "rev-parse", "--git-dir")
    if code == 0:
        bundle = staging / "repo.bundle"
        code, out = git(project, "bundle", "create", str(bundle), "--all")
        if code == 0 and bundle.exists():
            files.append(("_git/repo.bundle", bundle))
        else:
            skipped.append(("_git/repo.bundle", out.decode(errors="replace")[:200]))
        for name, args in (("status.txt", ("status", "--porcelain=v1", "-b")),
                           ("uncommitted.diff", ("diff", "HEAD", "--binary")),
                           ("log.txt", ("log", "--all", "--oneline", "--graph", "-n", "200")),
                           ("stash.txt", ("stash", "list"))):
            _, out = git(project, *args)
            p = staging / name
            p.write_bytes(out)
            files.append(("_git/" + name, p))
        _, out = git(project, "ls-files", "--others", "--exclude-standard", "-z")
        for rel in filter(None, out.decode(errors="replace").split("\0")):
            p, parts = project / rel, Path(rel).parts
            # data/ and outputs/ are sent below under their own rules, even if the
            # old .gitignore did not cover them.
            if parts[0] in ("data", "outputs") or rel.endswith(".tmp"):
                continue
            if p.is_file() and not any(skip_dir(part) for part in parts):
                files.append(("_git/untracked/" + Path(rel).as_posix(), p))
    else:
        skipped.append(("_git", "not a git repository (or git not installed)"))

    for top in ("data", "outputs"):
        base = project / top
        if not base.is_dir():
            skipped.append((top, "folder not found"))
            continue
        for p in walk(base):
            rel = p.relative_to(project).as_posix()
            if top == "data" and Path(rel).parts[1] in SKIP_DATA:
                continue
            mb = p.stat().st_size / 1e6
            if mb > max_mb and not include_large:
                skipped.append((rel, f"{mb:.0f} MB > --max-mb {max_mb:.0f}"))
                continue
            files.append((rel, p))
    return files, skipped


def lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))      # no packet is sent; picks the LAN interface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        ips.update(a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


def serve(a) -> int:
    project = a.project.resolve()
    if not project.is_dir():
        print(f"{project} is not a folder"); return 2
    staging = Path(tempfile.mkdtemp(prefix="fyp_transfer_"))
    print(f"Preparing the file list from {project} ...")
    files, skipped = build_list(project, staging, a.max_mb, a.include_large)
    print(f"Hashing {len(files)} files ({sum(p.stat().st_size for _, p in files) / 1e6:,.0f} MB) ...")
    manifest = {"project": str(project), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "files": [{"i": i, "name": n, "bytes": p.stat().st_size, "sha256": sha256(p)}
                          for i, (n, p) in enumerate(files)],
                "skipped": [{"name": n, "reason": r} for n, r in skipped]}
    body = json.dumps(manifest).encode()
    token = secrets.token_urlsafe(12)
    state = {"last": time.time(), "done": False}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _authorised(self) -> bool:
            if not secrets.compare_digest(self.headers.get("X-Token", ""), token):
                self.send_error(403, "bad token")
                return False
            state["last"] = time.time()
            return True

        def do_GET(self):
            if not self._authorised():
                return
            if self.path == "/manifest":
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/file/"):
                try:
                    _, p = files[int(self.path[len("/file/"):])]
                except (ValueError, IndexError):
                    self.send_error(404); return
                self.send_response(200)
                self.send_header("Content-Length", str(p.stat().st_size))
                self.end_headers()
                with p.open("rb") as f:
                    shutil.copyfileobj(f, self.wfile, CHUNK)
            elif self.path == "/done":
                self.send_response(200); self.end_headers()
                state["done"] = True
            else:
                self.send_error(404)

    srv = http.server.ThreadingHTTPServer((a.bind, a.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("\nReady. On the NEW laptop, from the project folder, run:\n")
    for ip in ([a.bind] if a.bind != "0.0.0.0" else lan_ips()) or ["<this laptop's IP: run ipconfig>"]:
        print(f"    python -m experiments.laptop_transfer fetch --host {ip} --token {token}")
    print(f"\n{len(files)} files offered, {len(skipped)} skipped (listed in the manifest).")
    print(f"Stops when the fetch finishes, or after {a.idle_minutes} idle minutes. Ctrl+C to stop now.")
    try:
        while not state["done"] and time.time() - state["last"] < a.idle_minutes * 60:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    srv.shutdown()
    shutil.rmtree(staging, ignore_errors=True)
    print("done" if state["done"] else "stopped")
    return 0


def fetch(a) -> int:
    dest = a.dest.resolve()
    here = Path(__file__).resolve().parents[1]
    if dest == here or here in dest.parents and dest.relative_to(here).parts[0] in ("data", "outputs"):
        print("refusing: --dest must not be inside this project's data/ or outputs/"); return 2
    dest.mkdir(parents=True, exist_ok=True)
    base = f"http://{a.host}:{a.port}"

    def get(path, timeout=60):
        req = urllib.request.Request(base + path, headers={"X-Token": a.token})
        return urllib.request.urlopen(req, timeout=timeout)

    try:
        manifest = json.loads(get("/manifest").read())
    except Exception as exc:
        print(f"cannot reach {base}: {exc}\n"
              "  - same network? (both on the same Wi-Fi, or joined by a cable)\n"
              "  - did Windows Firewall on the old laptop allow Python on Private networks?\n"
              "  - is the token right?")
        return 1
    (dest / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    todo = manifest["files"]
    total = sum(f["bytes"] for f in todo)
    print(f"{len(todo)} files, {total / 1e6:,.0f} MB from {manifest['project']}")
    got = bad = reused = 0
    t0 = time.time()
    for f in todo:
        out = dest / f["name"]
        if out.exists() and out.stat().st_size == f["bytes"] and sha256(out) == f["sha256"]:
            reused += f["bytes"]
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        with get(f"/file/{f['i']}", timeout=300) as r, part.open("wb") as w:
            shutil.copyfileobj(r, w, CHUNK)
        if sha256(part) != f["sha256"]:
            bad += 1
            print(f"  HASH MISMATCH {f['name']} (kept as .part)")
            continue
        part.replace(out)
        got += f["bytes"]
        done = got + reused
        rate = got / 1e6 / max(time.time() - t0, 1e-3)
        print(f"  {done / max(total, 1):6.1%}  {rate:5.1f} MB/s  {f['name']}", flush=True)
    if not bad:
        try:
            get("/done").read()
        except Exception:
            pass
    print(f"\n{len(todo) - bad} of {len(todo)} files verified in {dest}"
          + (f"; {bad} failed - re-run the same command to retry them" if bad else ""))
    if manifest["skipped"]:
        print(f"{len(manifest['skipped'])} skipped on the old laptop (see _manifest.json), e.g.:")
        for s in manifest["skipped"][:8]:
            print(f"  {s['name']}: {s['reason']}")
    print(f"\nNext: python -m experiments.recover_scan {dest}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Old laptop -> new laptop transfer over the LAN.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run on the OLD laptop")
    s.add_argument("--project", type=Path, default=Path("."))
    s.add_argument("--port", type=int, default=PORT)
    s.add_argument("--bind", default="0.0.0.0", help="127.0.0.1 for a local test")
    s.add_argument("--max-mb", type=float, default=1500)
    s.add_argument("--include-large", action="store_true")
    s.add_argument("--idle-minutes", type=float, default=30)
    f = sub.add_parser("fetch", help="run on the NEW laptop")
    f.add_argument("--host", required=True)
    f.add_argument("--token", required=True)
    f.add_argument("--port", type=int, default=PORT)
    f.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "recovered" / "old_laptop")
    a = ap.parse_args()
    return serve(a) if a.cmd == "serve" else fetch(a)


if __name__ == "__main__":
    sys.exit(main())
