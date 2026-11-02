import io
import json
import unittest
from unittest import mock
from types import SimpleNamespace

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

    def handler(self, host="192.168.1.50:8722", origin=None, token=None,
                bind="0.0.0.0", port=8722):
        handler = object.__new__(meter.H)
        handler.server = SimpleNamespace(server_address=(bind, port))
        handler.headers = {"Host": host, "Content-Type": "application/json",
                           "Content-Length": "2"}
        if origin is not None:
            handler.headers["Origin"] = origin
        if token is not None:
            handler.headers["X-Token-Meter-Action"] = token
        handler.rfile = io.BytesIO(b"{}")
        handler._send = mock.Mock()
        return handler

    def status(self, handler):
        return handler._send.call_args.kwargs["status"]

    def test_lan_ip_and_machine_names_allowed(self):
        with mock.patch("socket.gethostname", return_value="jay-ml-srv.lan"), \
                mock.patch("socket.getfqdn", return_value="jay-ml-srv.example.lan"):
            for host in ("192.168.1.50", "10.20.30.40", "jay-ml-srv", "JAY-ML-SRV.LAN",
                         "jay-ml-srv.example.lan", "localhost", "127.0.0.1"):
                with self.subTest(host=host):
                    self.assertTrue(self.handler(host + ":8722")._host_allowed())

    def test_explicit_bind_accepts_only_its_ip(self):
        self.assertTrue(self.handler(bind="192.168.1.50")._host_allowed())
        handler = self.handler(host="192.168.1.51:8722", bind="192.168.1.50")
        self.assertFalse(handler._host_allowed())
        self.assertEqual(self.status(handler), 403)

    def test_optional_allowed_names_and_port(self):
        with mock.patch.dict(meter.os.environ, {"TOKEN_METER_ALLOWED_HOSTS": " meter.lan, EXTRA.lan "}):
            for host in ("meter.lan:8722", "extra.LAN:8722"):
                self.assertTrue(self.handler(host)._host_allowed())
            for host in ("meter.lan:9999", "evil.example:8722", "meter.lan"):
                self.assertFalse(self.handler(host)._host_allowed())

    def test_evil_and_malformed_hosts_forbidden_in_all_modes(self):
        for bind in ("127.0.0.1", "0.0.0.0", "192.168.1.50"):
            for host in ("evil.example:8722", "evil.example:8722@192.168.1.50:8722",
                         "192.168.1.50:8722/path", "192.168.1.50:9999", "", "[::1]:8722"):
                with self.subTest(bind=bind, host=host):
                    handler = self.handler(host, bind=bind)
                    self.assertFalse(handler._host_allowed())
                    self.assertEqual(self.status(handler), 403)

    def test_api_same_origin_and_loopback_with_action_token(self):
        for origin in (None, "http://192.168.1.50:8722", "http://localhost:8722",
                       "http://127.0.0.1:8722", "http://[::1]:8722"):
            self.assertTrue(self.handler(origin=origin, token=meter._ACTION_TOKEN)._local_api_allowed())
        with mock.patch("socket.gethostname", return_value="jay-ml-srv"):
            self.assertTrue(self.handler("JAY-ML-SRV:8722", "http://jay-ml-srv:8722",
                                         meter._ACTION_TOKEN)._local_api_allowed())

    def test_cross_origin_forbidden(self):
        for origin in ("http://192.168.1.51:8722", "http://192.168.1.50:9999",
                       "https://192.168.1.50:8722", "http://evil.example:8722",
                       "http://192.168.1.50:bad", "http://[", "null"):
            handler = self.handler(origin=origin, token=meter._ACTION_TOKEN)
            self.assertFalse(handler._local_api_allowed())
            self.assertEqual(self.status(handler), 403)

    def test_api_missing_token_is_unauthorized_and_bearer_unchanged(self):
        handler = self.handler(origin="http://192.168.1.50:8722")
        self.assertFalse(handler._local_api_allowed())
        self.assertEqual(self.status(handler), 401)
        handler.headers["Authorization"] = "Bearer test-api-token"
        with mock.patch.object(meter, "api_token_matches", return_value=True) as matches:
            self.assertTrue(handler._local_api_allowed())
        matches.assert_called_once_with("test-api-token")

    def test_dashboard_actions_work_with_lan_origin_and_token(self):
        for token, status in ((None, 403), (meter._ACTION_TOKEN, 200)):
            handler = self.handler(origin="http://192.168.1.50:8722", token=token)
            handler.path = "/settings/updates"
            with mock.patch.object(meter, "set_update_settings", return_value={"ok": True, "updates": {"enabled": False}}):
                handler.do_POST()
            self.assertEqual(self.status(handler), status)
        handler = self.handler(origin="http://evil.example:8722", token=meter._ACTION_TOKEN)
        handler.path = "/settings/updates"
        handler.do_POST()
        self.assertEqual(self.status(handler), 403)

    def test_loopback_mode_retains_upstream_host_origin_and_token_rules(self):
        with mock.patch.dict(meter.os.environ, {"TOKEN_METER_ALLOWED_HOSTS": "meter.lan"}):
            for host in ("192.168.1.50:8722", "meter.lan:8722"):
                self.assertFalse(self.handler(host, bind="127.0.0.1")._host_allowed())
        for host in ("127.0.0.1:8722", "LOCALHOST:8722"):
            for origin in (None, "http://localhost:9999", "https://127.0.0.1", "http://[::1]"):
                handler = self.handler(host, origin, meter._ACTION_TOKEN, bind="127.0.0.1")
                self.assertTrue(handler._local_api_allowed())
            handler = self.handler(host, "http://192.168.1.50:8722", meter._ACTION_TOKEN,
                                   bind="127.0.0.1")
            self.assertFalse(handler._local_api_allowed())
            self.assertEqual(self.status(handler), 403)
            handler = self.handler(host, bind="127.0.0.1")
            self.assertFalse(handler._local_api_allowed())
            self.assertEqual(self.status(handler), 401)

    def test_main_passes_configured_host(self):
        for host in (None, "0.0.0.0", "192.168.1.50"):
            with self.subTest(host=host), mock.patch.dict(meter.os.environ, {}, clear=True):
                if host is not None:
                    meter.os.environ["TOKEN_METER_HOST"] = host
                with mock.patch.object(meter, "serve_local") as serve, mock.patch("builtins.print"), \
                        mock.patch.object(meter, "_matched_pace_persist", False), \
                        mock.patch.object(meter, "_parallel_index_enabled", False), \
                        mock.patch.object(meter, "_notification_persist", False), \
                        mock.patch.object(meter, "ensure_api_token"):
                    meter.main()
                self.assertEqual(serve.call_args.kwargs["host"], host or "127.0.0.1")
