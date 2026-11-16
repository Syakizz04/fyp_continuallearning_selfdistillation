"""
The gateway's view of the rest of the system: the edge nodes, the inventory
service and the control plane, over HTTP.

One small class so the service can be tested against a fake without any of the
real processes running. Every call raises `UpstreamError` on failure, which the
service turns into a 502 naming the service that is down - the dashboard shows
"inventory service unreachable", not a stack trace.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from ..config import channel_names, control_url, inventory_url, node_url, sim_url


class UpstreamError(Exception):
    def __init__(self, service: str, detail: str) -> None:
        super().__init__(f"{service}: {detail}")
        self.service = service
        self.detail = detail


class Upstream:
    def __init__(self, timeout: float = 10.0) -> None:
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def channels(self):
        return channel_names()

    def _get(self, service: str, url: str, params: Optional[Dict] = None,
             timeout: Optional[float] = None) -> Any:
        try:
            r = self._client.get(url, params=params,
                                 timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT)
        except httpx.HTTPError as exc:
            raise UpstreamError(service, f"unreachable ({type(exc).__name__})") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise UpstreamError(service, f"HTTP {r.status_code}: {detail}")
        return r.json()

    def _post(self, service: str, url: str, body: Optional[Dict] = None,
              timeout: Optional[float] = None) -> Any:
        try:
            r = self._client.post(url, json=body or {},
                                  timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT)
        except httpx.HTTPError as exc:
            raise UpstreamError(service, f"unreachable ({type(exc).__name__})") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise UpstreamError(service, f"HTTP {r.status_code}: {detail}")
        return r.json()

    # ── the interactive simulator ───────────────────────────────────────────
    def sim(self, path: str, params: Optional[Dict] = None) -> Any:
        return self._get("simulator", f"{sim_url()}{path}", params, timeout=3.0)

    def sim_post(self, path: str, body: Optional[Dict] = None) -> Any:
        return self._post("simulator", f"{sim_url()}{path}", body, timeout=10.0)

    # ── edge nodes ──────────────────────────────────────────────────────────
    def node(self, name: str, path: str, params: Optional[Dict] = None,
             timeout: Optional[float] = None) -> Any:
        return self._get(f"edge node {name}", f"{node_url(name)}{path}", params, timeout)

    # ── inventory ───────────────────────────────────────────────────────────
    def inventory(self, path: str, params: Optional[Dict] = None) -> Any:
        return self._get("inventory service", f"{inventory_url()}{path}", params)

    # ── control plane ───────────────────────────────────────────────────────
    def control(self, path: str, params: Optional[Dict] = None) -> Any:
        return self._get("control plane", f"{control_url()}{path}", params)
