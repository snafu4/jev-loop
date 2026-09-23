"""Every Alpaca and Jev call must reuse one kept-alive connection. A fresh
connection per call measured ~240 ms vs ~26 ms reused, made ~40% of ticks
late after ~2 hours, and crashed a 6-hour run with WinError 10055."""

import requests

from jevloop.assets import resolve_symbol
from jevloop.client import GatewayClient, TypeSafeDirectClient
from jevloop.execution.alpaca import AlpacaPaperClient


class _Resp:
    status_code = 200
    text = "{}"

    def json(self):
        return {"answers": {}, "model": "jev-test"}


class _RecordingSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        return _Resp()

    def post(self, url, **kwargs):
        self.calls.append(("POST", url))
        return _Resp()


def test_alpaca_client_holds_one_session_with_auth_headers():
    client = AlpacaPaperClient(api_key="k", secret_key="s", spec=resolve_symbol("BTC/USD"))
    assert isinstance(client._session, requests.Session)
    assert client._session.headers["APCA-API-KEY-ID"] == "k"
    assert client._session.headers["APCA-API-SECRET-KEY"] == "s"


def test_alpaca_calls_all_go_through_the_same_session():
    client = AlpacaPaperClient(api_key="k", secret_key="s", spec=resolve_symbol("BTC/USD"))
    session = _RecordingSession()
    client._session = session
    client.get_account()
    client.get_orderbook()
    client.get_position()
    assert len(session.calls) == 3


def test_jev_clients_reuse_their_session_across_ticks():
    for cls in (TypeSafeDirectClient, GatewayClient):
        client = cls(api_key="k")
        assert isinstance(client._session, requests.Session)
        session = _RecordingSession()
        client._session = session
        client.ask({}, {}, timeout=5.0)
        client.ask({}, {}, timeout=5.0)
        assert len(session.calls) == 2, cls.__name__
