"""Smoke tests for bench.common and the package skeleton."""

from bench.common import timer, write_result


def test_write_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_result("smoke_test", {"value": 1})
    assert path.exists()
    assert "smoke_test" in path.read_text()


def test_timer():
    with timer() as t:
        pass
    assert "elapsed_s" in t
