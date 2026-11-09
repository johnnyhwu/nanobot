"""Sandbox backends for shell command execution.

To add a backend, match the callable signature of _bwrap and _seatbelt,
including keyword-only sandbox_ro_binds, sandbox_rw_binds, allow_network and
env_keys, and register it in _BACKENDS below. env_keys names the variables the caller passes
to the command; a backend that starts the command with a cleared environment
must forward them.
"""

import ctypes
import os
import shlex
import sys
from pathlib import Path
from typing import Iterable

from nanobot.config.paths import get_media_dir


def _normalize_bind_paths(
    paths: Iterable[str] | None,
    *,
    workspace: Path | None = None,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in paths or []:
        value = str(raw).strip()
        if not value:
            continue
        path = Path(os.path.expandvars(value)).expanduser()
        if not path.is_absolute():
            continue
        resolved_path = path.resolve(strict=False)
        if workspace is not None:
            try:
                workspace.relative_to(resolved_path)
            except ValueError:
                pass
            else:
                # A later bind of the workspace or one of its parents could
                # cover the tmpfs that hides the config directory.
                continue
        resolved = str(resolved_path)
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _bwrap(
    command: str,
    workspace: str,
    cwd: str,
    *,
    sandbox_ro_binds: Iterable[str] | None = None,
    sandbox_rw_binds: Iterable[str] | None = None,
    allow_network: bool = True,
    env_keys: Iterable[str] | None = None,  # inherited by the wrapped command
) -> str:
    """Wrap command in a bubblewrap sandbox (requires bwrap in container).

    The workspace is bind-mounted read-write; its parent directory is hidden
    behind a fresh tmpfs. Media is mounted read-only for uploaded attachments.
    Explicit sandbox_ro_binds and sandbox_rw_binds add read-only and read-write
    roots, respectively. Without allow_network the command gets a new network
    namespace that only has loopback.
    """
    ws = Path(workspace).resolve()
    media = get_media_dir().resolve()

    try:
        sandbox_cwd = str(ws / Path(cwd).resolve().relative_to(ws))
    except ValueError:
        sandbox_cwd = str(ws)

    required = ["/usr"]
    optional = [
        "/bin",
        "/lib",
        "/lib64",
        "/etc/alternatives",
        "/etc/ssl/certs",
        "/etc/pki/tls/certs",
        "/etc/pki/ca-trust",
        "/etc/crypto-policies",
        "/etc/resolv.conf",
        "/etc/ld.so.cache",
    ]

    args = ["bwrap", "--new-session", "--die-with-parent", "--setenv", "HOME", str(ws)]
    if not allow_network:
        args.append("--unshare-net")
    for p in required:
        args += ["--ro-bind", p, p]
    for p in optional:
        args += ["--ro-bind-try", p, p]
    args += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--tmpfs", str(ws.parent),        # mask config dir
        "--dir", str(ws),                 # recreate workspace mount point
        "--bind", str(ws), str(ws),
        "--ro-bind-try", str(media), str(media),  # read-only access to media
    ]
    for p in _normalize_bind_paths(sandbox_ro_binds, workspace=ws):
        args += ["--ro-bind-try", p, p]
    for p in _normalize_bind_paths(sandbox_rw_binds, workspace=ws):
        args += ["--bind-try", p, p]
    args += ["--chdir", sandbox_cwd, "--", "sh", "-c", command]
    return shlex.join(args)


# System code and certificates needed by command-line tools. Do not grant
# /var, /tmp, /Library or /etc wholesale: those contain host application data,
# not an isolated scratch filesystem as in bwrap. Seatbelt matches real paths.
_SEATBELT_SYSTEM_READ_SUBPATHS = (
    "/usr",
    "/bin",
    "/sbin",
    "/dev",
    "/System",
    "/Library/Developer/CommandLineTools",
    "/private/etc/ssl",
    "/private/var/db/dyld",
    "/private/var/db/timezone",
)

_SEATBELT_SYSTEM_READ_LITERALS = (
    "/private/etc/hosts",
    "/private/etc/services",
    "/private/etc/protocols",
    "/private/etc/gitconfig",
    "/etc/gitconfig",
    "/private/var/run/resolv.conf",
    "/private/var/select/sh",
    "/private/var/select/developer_dir",
    "/private/var/db/xcode_select_link",
    "/var/select/developer_dir",
    "/var/db/xcode_select_link",
)

# Ancestors that must stay searchable but are not covered by any allowed
# subpath.  Resolving "/private/var/..." stats "/private", and
# (subpath "/private/var") does not cover its own parent.
_SEATBELT_TRAVERSABLE_LITERALS = ("/private",)


def _seatbelt_ancestors(*paths: Path) -> list[str]:
    """Collect every ancestor directory of *paths*, closest first.

    Seatbelt checks each path component while resolving, so a sandbox that
    allows a deep directory but not its ancestors fails with ENOTDIR on every
    command.  Only ``file-read-metadata`` is granted for these: the directories
    stay searchable, but listing or reading them is still denied.
    """
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for ancestor in path.parents:
            name = str(ancestor)
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def _seatbelt_is_within(path: Path, root: Path) -> bool:
    """Match existing path aliases without lowercasing case-sensitive volumes."""
    if path.is_relative_to(root):
        return True
    for ancestor in (path, *path.parents):
        try:
            if ancestor.samefile(root):
                return True
        except OSError:
            # Missing/inaccessible paths retain conservative lexical matching.
            continue
    return False


def _sbpl_quote(path: str) -> str:
    """Render *path* as an SBPL string literal.

    SBPL uses C-style string escaping, so a backslash or a double quote in a
    path would otherwise terminate the literal early and change the meaning of
    every rule that follows it.
    """
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _seatbelt(
    command: str,
    workspace: str,
    cwd: str,
    *,
    sandbox_ro_binds: Iterable[str] | None = None,
    sandbox_rw_binds: Iterable[str] | None = None,
    allow_network: bool = True,
    env_keys: Iterable[str] | None = None,  # inherited by the wrapped command
) -> str:
    """Wrap command in a macOS Seatbelt sandbox (requires sandbox-exec(1)).

    The workspace is read-write and media is read-only. The workspace's parent
    (the config directory in the default layout) is denied. Host temporary and
    application-data directories are not shared; TMPDIR and HOME use the
    workspace instead. Additional tool installations need explicit read binds.

    Seatbelt has no mount namespace, so masking the parent is expressed as a
    deny rule placed *before* the workspace allow — the last matching rule
    wins, which re-exposes the workspace subtree while the rest of the parent
    stays denied.  Unlike ``bwrap``, a denied parent still appears in directory
    listings; only its contents are inaccessible.

    Network access is allowed unless allow_network is false.
    """
    ws = Path(workspace).resolve()
    media = get_media_dir().resolve()

    try:
        sandbox_cwd = str(ws / Path(cwd).resolve().relative_to(ws))
    except ValueError:
        sandbox_cwd = str(ws)

    def subpaths(paths: Iterable[str]) -> str:
        return " ".join(f"(subpath {_sbpl_quote(p)})" for p in paths)

    ro_binds = _normalize_bind_paths(sandbox_ro_binds, workspace=ws)
    rw_binds = _normalize_bind_paths(sandbox_rw_binds, workspace=ws)

    rules = [
        "(version 1)",
        "(deny default)",
        "(allow process*)",
        "(allow sysctl-read)",
        "(allow mach*)",
        "(allow ipc*)",
        *(["(allow network*)"] if allow_network else []),
        # sh(1) reads the root directory while resolving a path, and
        # (subpath "/usr") does not cover "/" itself.  Without this rule the
        # shell aborts during startup.
        '(allow file-read* (literal "/"))',
        f"(allow file-read* {subpaths(_SEATBELT_SYSTEM_READ_SUBPATHS)})",
        "(allow file-read* "
        + " ".join(f"(literal {_sbpl_quote(p)})" for p in _SEATBELT_SYSTEM_READ_LITERALS)
        + ")",
        '(allow file-write* (literal "/dev/null") (literal "/dev/zero"))',
    ]

    # Mask the config directory.  The workspace allow below re-exposes the
    # workspace subtree, because the last matching rule wins.
    parent = ws.parent
    if parent != ws and str(parent) != "/":
        rules.append(f"(deny file-read* file-write* (subpath {_sbpl_quote(str(parent))}))")

    # Keep every ancestor of an allowed path searchable, including the masked
    # parent: metadata only, so it cannot be listed or read.
    traversable = _seatbelt_ancestors(
        ws, media, *(Path(p) for p in ro_binds), *(Path(p) for p in rw_binds),
        *(Path(p) for p in _SEATBELT_SYSTEM_READ_SUBPATHS),
        *(Path(p) for p in _SEATBELT_SYSTEM_READ_LITERALS),
    )
    literals = [
        *(p for p in _SEATBELT_TRAVERSABLE_LITERALS if p not in traversable),
        *traversable,
    ]
    rules.append(
        "(allow file-read-metadata "
        + " ".join(f"(literal {_sbpl_quote(p)})" for p in literals)
        + ")"
    )

    rules.append(f"(allow file-read* file-write* (subpath {_sbpl_quote(str(ws))}))")
    rules.append(f"(allow file-read* (subpath {_sbpl_quote(str(media))}))")
    rules.append(f"(deny file-write* (subpath {_sbpl_quote(str(media))}))")

    for p in ro_binds:
        rules.append(f"(allow file-read* (subpath {_sbpl_quote(p)}))")
        # A read allow does not revoke a broader write grant (e.g. workspace).
        rules.append(f"(deny file-write* (subpath {_sbpl_quote(p)}))")
    for p in rw_binds:
        rules.append(f"(allow file-read* file-write* (subpath {_sbpl_quote(p)}))")

    # Path-based read-only rules do not follow a renamed ancestor. Keep those
    # directory entries fixed without denying writes to their other children.
    # A RW bind covering the entire RO root intentionally overrides protection;
    # a writable descendant must not unlock the rest of the read-only tree.
    writable_roots = [Path(p) for p in rw_binds]
    protected_roots = [
        root for root in (media, *(Path(p) for p in ro_binds))
        if not any(_seatbelt_is_within(root, writable) for writable in writable_roots)
    ]
    ancestors = _seatbelt_ancestors(*protected_roots)
    if ancestors:
        rules.append(
            "(deny file-write-unlink "
            + " ".join(f"(literal {_sbpl_quote(p)})" for p in ancestors)
            + ")"
        )

    # sandbox-exec(1) has no --chdir. Exit on failure before evaluating any of
    # the submitted shell list; `cd ... && a; b` would still execute b on failure.
    args = [
        # Resolve the security boundary before consulting operator tool PATHs.
        "/usr/bin/sandbox-exec",
        "-p",
        "\n".join(rules),
        "/usr/bin/env",
        f"HOME={ws}",
        f"TMPDIR={ws}",
        "sh",
        "-c",
        f"cd {shlex.quote(sandbox_cwd)} || exit\n{command}",
    ]
    return shlex.join(args)


# Landlock rights that read-only roots must withhold arrived in ABI 3
# (truncate, Linux 6.2); earlier ABIs cannot keep media read-only.  TCP
# restriction arrived in ABI 4 (Linux 6.7).
_LANDLOCK_MIN_ABI = 3
_LANDLOCK_MIN_ABI_NO_NETWORK = 4
_SYS_LANDLOCK_CREATE_RULESET = 444  # same number on every Linux architecture
_LANDLOCK_CREATE_RULESET_VERSION = 1

_LANDRUN_SYSTEM_EXEC = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64")
_LANDRUN_SYSTEM_READ = (
    "/etc/alternatives",
    "/etc/ssl",
    "/etc/pki",
    "/etc/ca-certificates",
    "/etc/crypto-policies",
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/passwd",
    "/etc/group",
    "/etc/localtime",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/proc",
    "/dev/urandom",
    "/dev/random",
)
_LANDRUN_SYSTEM_WRITE = ("/dev/null", "/dev/zero", "/dev/tty")
_LANDRUN_FORWARDED_ENV = ("PATH", "LANG", "TERM")


def _landlock_abi() -> int:
    """Return the kernel's Landlock ABI version, or 0 when unavailable.

    seccomp-filtered, disabled or missing Landlock all report 0.
    """
    if not sys.platform.startswith("linux"):
        return 0
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(
        _SYS_LANDLOCK_CREATE_RULESET,
        None,
        ctypes.c_size_t(0),
        ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    return max(abi, 0)


def _landrun_path(path: str) -> str:
    # landrun splits every list flag value on commas, so "/a,b" would grant "/a".
    if "," in path:
        raise ValueError(f"landrun sandbox cannot express a path containing ',': {path!r}")
    return path


def _landrun(
    command: str,
    workspace: str,
    cwd: str,
    *,
    sandbox_ro_binds: Iterable[str] | None = None,
    sandbox_rw_binds: Iterable[str] | None = None,
    allow_network: bool = True,
    env_keys: Iterable[str] | None = None,
) -> str:
    """Wrap command with landrun(1), a Landlock CLI that needs no privileges.

    Landlock needs no namespaces or capabilities, so this backend works in
    containers that cannot run bwrap (no CAP_SYS_ADMIN, default seccomp). The
    process must run with no_new_privs, which landrun sets itself.

    Access is an allow-list: system code is read/execute, the workspace is
    read/write/execute, media is read-only, and explicit binds add roots.  The
    workspace parent (the config directory in the default layout) is simply
    not granted.  There is no mount namespace: nothing is hidden from
    directory listings outside granted roots, and host /tmp is not shared;
    HOME and TMPDIR use the workspace.  Landlock rules only add access, so a
    read-only bind inside the workspace or a read-write bind stays writable.
    Landlock also blocks ptrace and signals toward processes outside the
    sandbox, such as the gateway itself.

    Without allow_network, TCP bind and connect are denied (ABI 4+).  UDP is
    denied only from ABI 10, and pathname UNIX sockets only from ABI 9, so
    on older kernels DNS and other UDP traffic still work.  landrun clears
    the environment, so only env_keys plus PATH, LANG and TERM are forwarded.

    landrun is run with --best-effort because by default it requires the
    newest Landlock ABI; the minimum ABI is enforced here so an older kernel
    fails instead of running commands without the requested isolation.
    """
    required = _LANDLOCK_MIN_ABI if allow_network else _LANDLOCK_MIN_ABI_NO_NETWORK
    abi = _landlock_abi()
    if abi < required:
        kernel = "6.2" if required == _LANDLOCK_MIN_ABI else "6.7"
        raise RuntimeError(
            f"landrun sandbox requires Landlock ABI {required}+ "
            f"(Linux {kernel}+ with Landlock enabled); "
            f"this kernel reports {abi or 'no Landlock support'}"
        )

    ws = Path(workspace).resolve()
    media = get_media_dir().resolve()

    try:
        sandbox_cwd = str(ws / Path(cwd).resolve().relative_to(ws))
    except ValueError:
        sandbox_cwd = str(ws)

    args = ["landrun", "--best-effort", "--ignore-missing"]
    if allow_network:
        args.append("--unrestricted-network")
    for p in _LANDRUN_SYSTEM_EXEC:
        args += ["--rox", p]
    for p in _LANDRUN_SYSTEM_READ:
        args += ["--ro", p]
    for p in _LANDRUN_SYSTEM_WRITE:
        args += ["--rw", p]
    args += ["--rwx", _landrun_path(str(ws)), "--ro", _landrun_path(str(media))]
    for p in _normalize_bind_paths(sandbox_ro_binds, workspace=ws):
        args += ["--rox", _landrun_path(p)]
    for p in _normalize_bind_paths(sandbox_rw_binds, workspace=ws):
        args += ["--rwx", _landrun_path(p)]

    args += ["--env", f"HOME={ws}", "--env", f"TMPDIR={ws}"]
    forwarded = dict.fromkeys([*_LANDRUN_FORWARDED_ENV, *(env_keys or ())])
    for key in forwarded:
        if key not in ("HOME", "TMPDIR"):
            args += ["--env", key]

    # landrun has no --chdir.  Exit before evaluating any of the submitted
    # shell list; `cd ... && a; b` would still execute b on failure.
    args += ["--", "sh", "-c", f"cd {shlex.quote(sandbox_cwd)} || exit\n{command}"]
    return shlex.join(args)


_BACKENDS = {"bwrap": _bwrap, "seatbelt": _seatbelt, "landrun": _landrun}
SANDBOX_BACKENDS = tuple(_BACKENDS)


def wrap_command(
    sandbox: str,
    command: str,
    workspace: str,
    cwd: str,
    *,
    sandbox_ro_binds: Iterable[str] | None = None,
    sandbox_rw_binds: Iterable[str] | None = None,
    allow_network: bool = True,
    env_keys: Iterable[str] | None = None,
) -> str:
    """Wrap *command* using the named sandbox backend."""
    if backend := _BACKENDS.get(sandbox):
        return backend(
            command,
            workspace,
            cwd,
            sandbox_ro_binds=sandbox_ro_binds,
            sandbox_rw_binds=sandbox_rw_binds,
            allow_network=allow_network,
            env_keys=env_keys,
        )
    raise ValueError(f"Unknown sandbox backend {sandbox!r}. Available: {list(_BACKENDS)}")
