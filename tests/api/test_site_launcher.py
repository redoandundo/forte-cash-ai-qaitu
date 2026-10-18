import importlib.util
from pathlib import Path
import socket

spec = importlib.util.spec_from_file_location('serve_site', Path(__file__).resolve().parents[2] / 'scripts/serve_site.py')
site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site)


def test_site_key_persists_without_replacing_existing_key(tmp_path):
    first = site.access_key(tmp_path)
    assert len(first) >= 40
    assert site.access_key(tmp_path) == first


def test_busy_port_is_not_reused():
    with socket.socket() as listener:
        listener.bind(('0.0.0.0', 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert site.choose_port(port) != port
