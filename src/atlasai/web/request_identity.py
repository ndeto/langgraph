import hashlib
import hmac
import ipaddress
from collections.abc import Iterable

from fastapi import Request

from atlasai.web.dependencies import DemoWebSettings


def build_request_key_hash(request: Request, settings: DemoWebSettings) -> str:
    """Build a shared network quota key from the normalized client IP."""

    client_ip = _resolve_client_ip(
        request=request,
        trusted_proxies=settings.trusted_proxies,
    )
    normalized_ip = _normalize_ip(client_ip)
    digest = hmac.new(
        settings.ip_hash_secret.encode("utf-8"),
        normalized_ip.encode("utf-8"),
        hashlib.sha256,
    )
    return digest.hexdigest()


def _resolve_client_ip(*, request: Request, trusted_proxies: Iterable[str]) -> str:
    trusted = {proxy.strip() for proxy in trusted_proxies if proxy.strip()}
    immediate_client = request.client.host if request.client else ""
    if immediate_client in trusted:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            first_hop = forwarded_for.split(",")[0].strip()
            if first_hop:
                return first_hop

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

    return immediate_client or "0.0.0.0"


def _normalize_ip(value: str) -> str:
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError:
        return value

    if isinstance(parsed, ipaddress.IPv6Address):
        network = ipaddress.IPv6Network(f"{parsed}/64", strict=False)
        return str(network.network_address) + "/64"

    return str(parsed)
