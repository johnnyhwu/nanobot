"""Exercise the Landlock policy that landrun applies, not just its arguments.

test_sandbox.py covers argument generation on every platform. These tests run
only where landrun is installed and the kernel supports Landlock TCP rules,
for example inside the deployment image. They use isolated fixtures and a
loopback HTTP service only.
"""

import os
import shlex
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from nanobot.agent.tools.sandbox import _landlock_abi
from nanobot.agent.tools.shell import ExecTool

pytestmark = pytest.mark.skipif(
    shutil.which("landrun") is None or _landlock_abi() < 4,
    reason="requires landrun and Landlock ABI 4+",
)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    paths = {
        "workspace": root / "project",
        "config": root / "config.json",
        "media": root / "media",
        "ro": root / "readonly",
        "rw": root / "cache",
        "peer": root / "peer",
    }
    for name in ("workspace", "media", "ro", "rw", "peer"):
        paths[name].mkdir()
        (paths[name] / "sentinel").write_text("synthetic-data")
    paths["config"].write_text("synthetic-config")
    paths["link"] = paths["workspace"] / "outside-link"
    paths["link"].symlink_to(paths["peer"], target_is_directory=True)
    monkeypatch.setattr("nanobot.agent.tools.sandbox.get_media_dir", lambda: paths["media"])
    return paths


async def run_script(paths, script, *, allow_network=True):
    workspace = paths["workspace"]
    (workspace / "probe.sh").write_text(script)
    tool = ExecTool(
        working_dir=str(workspace), sandbox="landrun",
        sandbox_ro_binds=[str(paths["ro"])], sandbox_rw_binds=[str(paths["rw"])],
        sandbox_allow_network=allow_network,
    )
    return str(await tool.execute(command="sh probe.sh", timeout=10))


@pytest.mark.parametrize("target", ["workspace", "media", "ro", "rw"])
async def test_native_allowed_reads(layout, target):
    result = await run_script(layout, f"cat {shlex.quote(str(layout[target] / 'sentinel'))}")
    assert result == "synthetic-data\n\nExit code: 0"


@pytest.mark.parametrize("target", ["config", "peer", "link"])
async def test_native_outside_reads_denied(layout, target):
    path = layout[target] if target == "config" else layout[target] / "sentinel"
    result = await run_script(layout, f"cat {shlex.quote(str(path))}")
    assert "Permission denied" in result
    assert "synthetic-" not in result


@pytest.mark.parametrize("target", ["workspace", "rw"])
async def test_native_explicit_writes_allowed(layout, target):
    path = layout[target] / "sentinel"
    result = await run_script(layout, f"printf changed > {shlex.quote(str(path))}")
    assert result == "\nExit code: 0"
    assert path.read_text() == "changed"


@pytest.mark.parametrize("target", ["config", "peer", "link", "ro", "media"])
async def test_native_other_writes_denied(layout, target):
    path = layout[target] if target == "config" else layout[target] / "sentinel"
    before = path.read_text()
    result = await run_script(layout, f"printf changed > {shlex.quote(str(path))}")
    assert "Permission denied" in result
    assert path.read_text() == before


async def test_native_home_and_tmpdir_use_workspace(layout):
    result = await run_script(layout, 'printf "%s\\n%s\\n" "$HOME" "$TMPDIR"; mktemp')
    lines = result.splitlines()
    assert result.endswith("Exit code: 0")
    assert all(line.startswith(str(layout["workspace"])) for line in lines[:3])


async def test_native_gateway_process_cannot_be_inspected(layout):
    # Landlock blocks ptrace-mode access to processes outside the sandbox,
    # which protects the gateway's environment (API keys) from /proc.
    result = await run_script(layout, f"cat /proc/{os.getpid()}/environ")
    assert "Permission denied" in result


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"loopback-ok")

    def log_message(self, *args):
        pass


@pytest.fixture
def loopback_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()


@pytest.mark.parametrize("allow_network", [True, False])
async def test_native_tcp_follows_network_setting(layout, loopback_url, allow_network):
    probe = (
        "python3 -c 'import sys, urllib.request; "
        "print(urllib.request.urlopen(sys.argv[1], timeout=3).read().decode())' "
        f"{shlex.quote(loopback_url)}"
    )
    result = await run_script(layout, probe, allow_network=allow_network)
    assert ("loopback-ok" in result) is allow_network
