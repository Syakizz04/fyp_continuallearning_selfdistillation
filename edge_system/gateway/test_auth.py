"""The account store on its own: hashing, sessions, validation."""

import time

import pytest

from .auth import AuthError, AuthStore


@pytest.fixture
def a():
    return AuthStore(":memory:")


def test_password_is_not_stored_and_verifies(a):
    a.add_user("alice", "long-enough", "operator")
    row = a._conn.execute("SELECT * FROM users").fetchone()
    assert b"long-enough" not in bytes(row["pw_hash"]) and len(row["salt"]) == 16
    assert a.verify("alice", "long-enough")["role"] == "operator"
    assert a.verify("alice", "wrong-one!") is None
    assert a.verify("nobody", "long-enough") is None


def test_validation(a):
    with pytest.raises(AuthError):
        a.add_user("x", "long-enough")            # name too short
    with pytest.raises(AuthError):
        a.add_user("bob", "short")
    with pytest.raises(AuthError):
        a.add_user("bob", "long-enough", "admin")
    a.add_user("bob", "long-enough")
    with pytest.raises(AuthError) as e:
        a.add_user("BOB", "long-enough")           # names are case-insensitive
    assert e.value.status == 409


def test_session_token_is_hashed_and_expires(a, monkeypatch):
    a.add_user("alice", "long-enough")
    token, _ = a.login("alice", "long-enough")
    assert a._conn.execute("SELECT COUNT(*) FROM sessions WHERE token_hash = ?",
                           (token,)).fetchone()[0] == 0
    assert a.session_user(token)["username"] == "alice"
    later = time.time() + 13 * 3600
    monkeypatch.setattr(time, "time", lambda: later)
    assert a.session_user(token) is None


def test_removing_a_user_ends_their_sessions(a):
    a.add_user("alice", "long-enough")
    a.add_user("bob", "long-enough", "viewer")
    token, _ = a.login("bob", "long-enough")
    a.remove_user("bob")
    assert a.session_user(token) is None
