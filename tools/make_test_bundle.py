#!/usr/bin/env python3
"""Pack a cross-compiled build's tests for a machine without build tools.

    tools/make_test_bundle.py <build dir> <out.zip>

Reads the tests from `ctest --show-only=json-v1`, copies every program a test
runs (and the files its arguments name inside the build folder) into the
bundle, and writes run_tests.ps1, which runs each test with its environment
and prints one PASS/FAIL/SKIP line per test and a summary. Tests run by an
emulator or by Python (the conformance harness) are left out.
"""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def main(argv):
    build = Path(argv[0]).resolve()
    out = Path(argv[1]).resolve()
    info = json.loads(subprocess.run(["ctest", "--show-only=json-v1"], cwd=build,
                                     capture_output=True, text=True, check=True).stdout)
    files, lines, skipped = {}, [], []
    for t in info["tests"]:
        cmd = list(t.get("command", []))
        if not cmd:
            continue
        # Emulator prefix (cross builds run tests through Wine): drop it.
        if not cmd[0].endswith(".exe") and len(cmd) > 1 and cmd[1].endswith(".exe"):
            cmd = cmd[1:]
        if not cmd[0].endswith(".exe"):
            skipped.append(t["name"])
            continue
        args = []
        for a in cmd:
            p = Path(a)
            if p.is_absolute() and p.is_file() and build in p.parents:
                rel = p.relative_to(build)
                files[str(rel)] = p
                args.append(".\\" + str(rel).replace("/", "\\"))
            elif p.is_absolute() and str(p).startswith(str(build)):
                rel = p.relative_to(build)
                args.append(".\\" + str(rel).replace("/", "\\"))
            else:
                args.append(a)
        env = []
        skip_code = None
        for prop in t.get("properties", []):
            if prop["name"] == "ENVIRONMENT":
                env = [e for e in prop["value"] if not e.startswith("OCU_DRIVER_LAUNCHER")]
            if prop["name"] == "SKIP_RETURN_CODE":
                skip_code = prop["value"]
        lines.append((t["name"], args, env, skip_code))
    ps = ["$ErrorActionPreference = 'Continue'", "$pass = 0; $fail = 0; $skip = 0",
          "Set-Location $PSScriptRoot"]
    for name, args, env, skip_code in lines:
        ps.append("& {")
        for e in env:
            k, _, v = e.partition("=")
            ps.append(f"  $env:{k} = '{v}'")
        quoted = " ".join("'" + a.replace("'", "''") + "'" for a in args[1:])
        ps.append(f"  $out = & '{args[0]}' {quoted} 2>&1; $rc = $LASTEXITCODE")
        cond = f"$rc -eq {skip_code} -or " if skip_code is not None else ""
        ps.append(f"  if ({cond}($out -match 'SKIP')) {{ if ($rc -eq 0 -or $rc -eq {skip_code or 0}) "
                  f"{{ Write-Output 'SKIP  {name}'; $script:skip++ }} else {{ Write-Output \"FAIL  {name} (rc $rc)\"; "
                  f"$out | Select-Object -Last 20; $script:fail++ }} }}")
        ps.append(f"  elseif ($rc -eq 0) {{ Write-Output 'PASS  {name}'; $script:pass++ }}")
        ps.append(f"  else {{ Write-Output \"FAIL  {name} (rc $rc)\"; $out | Select-Object -Last 20; $script:fail++ }}")
        for e in env:
            k = e.partition("=")[0]
            ps.append(f"  Remove-Item Env:{k} -ErrorAction SilentlyContinue")
        ps.append("}")
    ps.append("Write-Output \"summary: $pass passed, $fail failed, $skip skipped of "
              f"{len(lines)}\"")
    if skipped:
        ps.append("Write-Output 'not in this bundle (need Python or an emulator): "
                  + ", ".join(skipped) + "'")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, p in sorted(files.items()):
            z.write(p, rel)
        z.writestr("run_tests.ps1", "\r\n".join(ps) + "\r\n")
    print(f"{out}: {len(lines)} tests, {len(files)} files, left out: {len(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
