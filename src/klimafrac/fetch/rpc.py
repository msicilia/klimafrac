"""JSON-RPC access that fails loudly.

A transient error must never be recorded as an empty result: a silent zero
is indistinguishable from a real finding. Every call here either returns the
node's answer or raises `RpcError` after bounded retries.

Endpoints:
  polygon      Alchemy, free tier (key in ALCHEMY_KEY or .env at the
               repository root)
  base         Alchemy, free tier
"""

import os
import time

import requests

from klimafrac.paths import CODE_ROOT


class RpcError(RuntimeError):
    pass


def alchemy_key() -> str:
    key = os.environ.get("ALCHEMY_KEY")
    if key:
        return key
    env = CODE_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ALCHEMY_KEY="):
                return line.split("=", 1)[1].strip()
    raise RpcError("ALCHEMY_KEY not set (environment or .env)")


ENDPOINTS = {
    "polygon": lambda: f"https://polygon-mainnet.g.alchemy.com/v2/{alchemy_key()}",
    "base": lambda: f"https://base-mainnet.g.alchemy.com/v2/{alchemy_key()}",
}

# Errors worth retrying: rate limits and transient server faults. A request
# error -- an oversized range, invalid parameters, a revert -- is deterministic
# and must raise at once, so the caller can narrow the range or handle it.
TRANSIENT_CODES = {429, -32000, -32005, -32016, -32029, -32603}
PERMANENT_CODES = {3, -32600, -32601, -32602}
TRANSIENT_WORDS = ("rate limit", "rate-limit", "capacity", "too many requests",
                   "timeout", "timed out", "temporarily", "unavailable", "busy")


def _is_transient(error) -> bool:
    """Classify on the code where there is one; fall back to the message."""
    if isinstance(error, dict):
        code = error.get("code")
        if code in PERMANENT_CODES:
            return False
        if code in TRANSIENT_CODES:
            return True
        msg = str(error.get("message", ""))
    else:
        msg = str(error)
    return any(word in msg.lower() for word in TRANSIENT_WORDS)


class Client:
    def __init__(self, endpoint: str, attempts: int = 10, timeout: int = 120):
        self.name = endpoint
        self.url = ENDPOINTS[endpoint]()
        self.attempts = attempts
        self.timeout = timeout
        self.session = requests.Session()

    def _redact(self, text: str) -> str:
        try:
            return text.replace(alchemy_key(), "<key>")
        except RpcError:
            return text

    def batch(self, calls: list[tuple[str, list]], allow_errors: bool = False) -> list:
        """
        Results in call order.

        A per-call error raises, unless `allow_errors`, in which case each
        entry comes back as (result, error) so that calls expected to revert
        -- probing whether a contract implements something -- can share one
        request with calls that must succeed.
        """
        body = [{"jsonrpc": "2.0", "id": i, "method": m, "params": p}
                for i, (m, p) in enumerate(calls)]
        delay, last = 1.0, ""
        for _ in range(self.attempts):
            try:
                r = self.session.post(self.url, json=body, timeout=self.timeout)
                if r.status_code == 200:
                    out = r.json()
                    out = out if isinstance(out, list) else [out]
                    errors = [x["error"] for x in out if "error" in x]
                    if len(out) == len(calls) and (allow_errors or not errors):
                        by_id = {x["id"]: x for x in out}
                        if allow_errors:
                            return [(by_id[i].get("result"), by_id[i].get("error"))
                                    for i in range(len(calls))]
                        return [by_id[i]["result"] for i in range(len(calls))]
                    if errors and not all(_is_transient(e) for e in errors):
                        raise RpcError(f"{self.name}: {errors[0]}")
                    last = f"transient: {errors[:1]}" if errors else \
                        f"{len(out)} responses for {len(calls)} calls"
                else:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
            except (requests.RequestException, ValueError, KeyError) as e:
                last = repr(e)
            time.sleep(delay)
            delay = min(2 * delay, 60.0)
        raise RpcError(self._redact(
            f"{self.name}: giving up after {self.attempts} attempts ({last})"))

    def call(self, method: str, params: list):
        return self.batch([(method, params)])[0]

    def block_number(self) -> int:
        return int(self.call("eth_blockNumber", []), 16)

    def eth_call(self, to: str, data: str, block: int | str = "latest") -> str:
        tag = hex(block) if isinstance(block, int) else block
        return self.call("eth_call", [{"to": to, "data": data}, tag])
