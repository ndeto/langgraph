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
    immediate_client = request.client.host if request.client else ""
    if _is_trusted_proxy(immediate_client, trusted_proxies):
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            first_hop = forwarded_for.split(",")[0].strip()
            if first_hop:
                return first_hop

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

    return immediate_client or "0.0.0.0"


def _is_trusted_proxy(client_ip: str, trusted_proxies: Iterable[str]) -> bool:
    try:
        parsed_client = ipaddress.ip_address(client_ip)
    except ValueError:
        return False

    for proxy in trusted_proxies:
        value = proxy.strip()
        if not value:
            continue
        try:
            network = ipaddress.ip_network(value, strict=False)
        except ValueError:
            continue
        if parsed_client in network:
            return True
    return False


def _normalize_ip(value: str) -> str:
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError:
        return value

    if isinstance(parsed, ipaddress.IPv6Address):
        network = ipaddress.IPv6Network(f"{parsed}/64", strict=False)
        return str(network.network_address) + "/64"

    return str(parsed)
