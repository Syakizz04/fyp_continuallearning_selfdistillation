"""
Local accounts for the operator dashboard.

Criterion 3 rests on the decision log: who approved which price. Before this,
"who" was a name typed into the sidebar, so anyone could sign as anyone. Now a
decision is attributed to the session that made it, and the name in the request
body is ignored.

Deliberately local: the demo runs offline on one machine, so an external
identity provider would add an internet dependency and a redirect that can fail
mid-viva, for no gain in what the dashboard demonstrates.

  * passwords - scrypt (stdlib), per-user salt, constant-time compare
  * sessions  - random 256-bit tokens; only their SHA-256 is stored, so a copy of
                the database does not hand out live sessions
  * roles     - "operator" decides, edits rules and drives the simulator;
                "viewer" reads everything and changes nothing

A separate SQLite file from decisions.db, because the launcher archives that
one on every start - accounts must outlive a run.

CLI (for scripted setup; the dashboard can also create the first account):
    python -m edge_system.gateway.auth add <name> [--role viewer]
    python -m edge_system.gateway.auth list
    python -m edge_system.gateway.auth remove <name>
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

ROLES = ("operator", "viewer")
SESSION_TTL_S = 12 * 3600
COOKIE = "shelfsense_session"
_NAME = re.compile(r"^[A-Za-z0-9._-]{2,32}$")
MIN_PASSWORD = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username    TEXT PRIMARY KEY COLLATE NOCASE,
    role        TEXT NOT NULL,
    salt        BLOB NOT NULL,
    pw_hash     BLOB NOT NULL,
    created_at  REAL NOT NULL,
    created_by  TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    username    TEXT NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);
"""


class AuthError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt,
                          n=2 ** 14, r=8, p=1, dklen=32)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


class AuthStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    # ── users ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def add_user(self, username: str, password: str, role: str = "operator",
                 *, created_by: Optional[str] = None) -> Dict:
        username = username.strip()
        if not _NAME.match(username):
            raise AuthError(422, "username: 2-32 letters, digits, '.', '_' or '-'")
        if len(password) < MIN_PASSWORD:
            raise AuthError(422, f"password must be at least {MIN_PASSWORD} characters")
        if role not in ROLES:
            raise AuthError(422, f"role must be one of {ROLES}")
        salt = secrets.token_bytes(16)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
                    (username, role, salt, _hash(password, salt), time.time(), created_by))
            except sqlite3.IntegrityError:
                raise AuthError(409, f"user {username!r} already exists") from None
        return self.user(username)

    def setup_first(self, username: str, password: str) -> Dict:
        """Create the first operator - only while there are no accounts, and
        atomically, so two browsers racing the setup page cannot both win."""
        with self._lock:
            if self.count():
                raise AuthError(409, "setup is already done; sign in")
        return self.add_user(username, password, "operator", created_by="setup")

    def user(self, username: str) -> Optional[Dict]:
        r = self._conn.execute(
            "SELECT username, role, created_at, created_by FROM users WHERE username = ?",
            (username,)).fetchone()
        return dict(r) if r else None

    def users(self) -> List[Dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT username, role, created_at, created_by FROM users ORDER BY username")]

    def remove_user(self, username: str) -> None:
        with self._lock:
            u = self.user(username)
            if u is None:
                raise AuthError(404, f"no user {username!r}")
            if u["role"] == "operator" and self._conn.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'operator'").fetchone()[0] <= 1:
                raise AuthError(409, "cannot remove the last operator")
            self._conn.execute("DELETE FROM sessions WHERE username = ?", (u["username"],))
            self._conn.execute("DELETE FROM users WHERE username = ?", (u["username"],))

    def verify(self, username: str, password: str) -> Optional[Dict]:
        r = self._conn.execute("SELECT * FROM users WHERE username = ?",
                               (username.strip(),)).fetchone()
        # Hash even for an unknown name, so the response time does not say
        # which usernames exist.
        salt = r["salt"] if r else b"\0" * 16
        ok = hmac.compare_digest(_hash(password, salt), r["pw_hash"] if r else b"\1" * 32)
        return self.user(r["username"]) if (r and ok) else None

    # ── sessions ────────────────────────────────────────────────────────────

    def login(self, username: str, password: str) -> tuple:
        u = self.verify(username, password)
        if u is None:
            raise AuthError(401, "wrong username or password")
        return self.new_session(u["username"]), u

    def new_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            self._conn.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                               (_token_hash(token), username, now, now + SESSION_TTL_S))
        return token

    def session_user(self, token: Optional[str]) -> Optional[Dict]:
        if not token:
            return None
        r = self._conn.execute(
            "SELECT username, expires_at FROM sessions WHERE token_hash = ?",
            (_token_hash(token),)).fetchone()
        if r is None or r["expires_at"] < time.time():
            return None
        return self.user(r["username"])

    def logout(self, token: Optional[str]) -> None:
        if token:
            with self._lock:
                self._conn.execute("DELETE FROM sessions WHERE token_hash = ?",
                                   (_token_hash(token),))


def default_path() -> Path:
    """Next to the decisions database, so the demo's accounts live in the demo
    state directory - but in their own file, which the launcher never archives."""
    from ..config import SYSTEM_CONFIG
    return Path(SYSTEM_CONFIG["paths"]["decisions_db"]).parent / "auth.db"


def _cli() -> None:
    import argparse
    import getpass

    p = argparse.ArgumentParser(prog="python -m edge_system.gateway.auth")
    p.add_argument("--db", help="auth database (default: next to decisions_db)")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("username")
    a.add_argument("--role", choices=ROLES, default="operator")
    r = sub.add_parser("remove"); r.add_argument("username")
    sub.add_parser("list")
    args = p.parse_args()

    store = AuthStore(args.db or default_path())
    try:
        if args.cmd == "add":
            pw = getpass.getpass(f"password for {args.username}: ")
            if pw != getpass.getpass("again: "):
                raise SystemExit("passwords differ")
            store.add_user(args.username, pw, args.role, created_by="cli")
            print(f"added {args.username} ({args.role})")
        elif args.cmd == "remove":
            store.remove_user(args.username)
            print(f"removed {args.username}")
        else:
            for u in store.users():
                print(f"{u['username']:<20} {u['role']}")
    except AuthError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    _cli()
