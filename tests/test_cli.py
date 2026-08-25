from __future__ import annotations

from types import SimpleNamespace

import pytest

from scalarpulse import cli


def test_serve_cli_passes_arguments_and_stops_handle(monkeypatch, tmp_path, capsys):
    captured = {}
    handle = SimpleNamespace(url="http://127.0.0.1:43210", stopped=False)

    def stop():
        handle.stopped = True

    handle.stop = stop

    def fake_start(log_dir, *, host, port, open_browser):
        captured.update(
            log_dir=log_dir,
            host=host,
            port=port,
            open_browser=open_browser,
        )
        return handle

    def interrupt_wait():
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "start_dashboard", fake_start)
    monkeypatch.setattr(cli, "_wait_forever", interrupt_wait)
    if hasattr(cli.signal, "pause"):
        monkeypatch.setattr(cli.signal, "pause", interrupt_wait)

    result = cli.main(
        [
            "serve",
            "--log-dir",
            str(tmp_path),
            "--host",
            "0.0.0.0",
            "--port",
            "43210",
            "--open",
        ]
    )

    assert result == 0
    assert captured == {
        "log_dir": tmp_path,
        "host": "0.0.0.0",
        "port": 43210,
        "open_browser": True,
    }
    assert handle.stopped
    output = capsys.readouterr().out
    assert "http://127.0.0.1:43210" in output
    assert str(tmp_path.resolve()) in output


@pytest.mark.parametrize(
    "arguments",
    [
        ["demo", "--steps", "0"],
        ["demo", "--steps", "-1"],
        ["demo", "--interval", "-0.1"],
    ],
)
def test_demo_cli_rejects_invalid_ranges_without_starting_a_run(monkeypatch, arguments, capsys):
    def fail_if_constructed(**kwargs):
        raise AssertionError("Tracker must not be created for invalid demo arguments")

    monkeypatch.setattr(cli, "Tracker", fail_if_constructed)
    assert cli.main(arguments) == 2
    assert "--steps must be positive" in capsys.readouterr().err


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "scalarpulse 0.1.0"
