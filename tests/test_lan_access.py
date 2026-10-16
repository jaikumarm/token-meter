import io
import json
import unittest
from unittest import mock

import meter
from token_meter.web.server import serve_local


class ListenHostTests(unittest.TestCase):
    def serve(self, host=None):
        server = mock.Mock()
        server.serve_forever.side_effect = KeyboardInterrupt
        factory = mock.Mock(return_value=server)
        kwargs = {} if host is None else {"host": host}
        with mock.patch("token_meter.web.server.threading.Thread") as thread:
            serve_local(handler_class=meter.H, server_class=factory, port=8722,
                        output=mock.Mock(), **kwargs)
        return factory, server, thread

    def test_default_stays_loopback(self):
        factory, server, _ = self.serve()
        factory.assert_called_once_with(("127.0.0.1", 8722), meter.H)
        server.server_close.assert_called_once()

    def test_wildcard_uses_one_listener(self):
        factory, server, _ = self.serve("0.0.0.0")
        factory.assert_called_once_with(("0.0.0.0", 8722), meter.H)
        server.server_close.assert_called_once()

    def test_explicit_address(self):
        factory, _, _ = self.serve("192.168.1.50")
        factory.assert_called_once_with(("192.168.1.50", 8722), meter.H)

    def test_invalid_host_fails_before_binding_or_workers(self):
        for host in ("", "example.com", "::", "127.1", "garbage"):
            with self.subTest(host=host), mock.patch("token_meter.web.server.threading.Thread") as thread:
                factory = mock.Mock()
                with self.assertRaises(ValueError):
                    serve_local(handler_class=meter.H, server_class=factory,
                                port=8722, host=host, background=(lambda: None,))
                factory.assert_not_called()
                thread.assert_not_called()

    def test_bind_failure_does_not_start_workers(self):
        with mock.patch("token_meter.web.server.threading.Thread") as thread:
            with self.assertRaises(OSError):
                serve_local(handler_class=meter.H, server_class=mock.Mock(side_effect=OSError),
                            port=8722, host="0.0.0.0", background=(lambda: None,))
            thread.assert_not_called()

    def test_server_closes_on_unexpected_error(self):
        server = mock.Mock()
        server.serve_forever.side_effect = RuntimeError
        with self.assertRaises(RuntimeError):
            serve_local(handler_class=meter.H, server_class=mock.Mock(return_value=server),
                        port=8722, output=mock.Mock())
        server.server_close.assert_called_once()

    def request(self, origin, token=None, destination="192.168.1.50", host=None):
        handler = object.__new__(meter.H)
        handler.path = "/settings/updates"
        handler.connection = mock.Mock()
        handler.connection.getsockname.return_value = (destination, 8722)
        handler.headers = {"Origin": origin, "Content-Type": "application/json",
                           "Content-Length": "1", "Host": host or destination + ":8722"}
        if token:
            handler.headers["X-Token-Meter-Action"] = token
        handler.rfile = io.BytesIO(b"{")
        handler._send = mock.Mock()
        handler.do_POST()
        return json.loads(handler._send.call_args.args[0]), handler._send.call_args.kwargs["status"]

    def test_connected_lan_origin_still_requires_token(self):
        payload, status = self.request("http://192.168.1.50:8722")
        self.assertEqual((payload["error"], status), ("Invalid action token.", 403))
        payload, status = self.request("http://192.168.1.50:8722", meter._ACTION_TOKEN)
        self.assertEqual((payload["error"], status), ("Invalid JSON.", 400))

    def test_unrelated_origins_and_spoofed_host_rejected(self):
        for origin in ("http://192.168.1.51:8722", "http://192.168.1.50:9999",
                       "https://192.168.1.50:8722", "http://evil.example:8722",
                       "http://0.0.0.0:8722", "null"):
            with self.subTest(origin=origin):
                payload, status = self.request(origin, meter._ACTION_TOKEN, host="evil.example:8722")
                self.assertEqual(status, 403)
                self.assertIn("origin", payload["error"])

    def test_other_lan_destination_and_loopback(self):
        for destination in ("10.20.30.40", "127.0.0.1"):
            payload, status = self.request("http://" + destination + ":8722", meter._ACTION_TOKEN,
                                           destination=destination)
            self.assertEqual((payload["error"], status), ("Invalid JSON.", 400))
        payload, status = self.request("http://192.168.1.50:8722", meter._ACTION_TOKEN,
                                       destination="127.0.0.1")
        self.assertEqual(status, 403)

    def test_main_passes_configured_host(self):
        for host in (None, "0.0.0.0", "192.168.1.50"):
            with self.subTest(host=host), mock.patch.dict(meter.os.environ, {}, clear=True):
                if host is not None:
                    meter.os.environ["TOKEN_METER_HOST"] = host
                with mock.patch.object(meter, "serve_local") as serve, mock.patch("builtins.print"), \
                        mock.patch.object(meter, "_matched_pace_persist", False):
                    meter.main()
                self.assertEqual(serve.call_args.kwargs["host"], host or "127.0.0.1")
