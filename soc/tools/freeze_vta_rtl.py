#!/usr/bin/env python3
"""Generate and freeze the VTA RTL package selected by a SoC config.

This is an explicit release-preparation command. Normal LiteX builds only
consume the imported package and must never invoke this tool or Chisel.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from soc_config import load_config, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vta-config", type=Path,
                        help="VTA hardware config; defaults to config/vta_config.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data = validate(load_config(args.config), require_ip=False)
    repo = Path(__file__).resolve().parents[2]
    vta = data["vta"]
    memory_protocol = vta["memory_protocol"]
    memory_width = vta["memory_data_width"]
    variant = f"apb32-{'axi' if memory_protocol == 'axi4' else 'ahb'}{memory_width}"
    ip_dir = data["_vta_ip_dir"]
    default_vta_config = repo / "soc" / "config" / f"vta_axi{memory_width}_config.json"
    if not default_vta_config.is_file():
        default_vta_config = repo / "config" / "vta_config.json"
    vta_config = args.vta_config or default_vta_config
    generated = repo / "build" / "soc-rtl" / variant / "VTAShellAPB.sv"
    if memory_protocol == "ahb-lite":
        generated = generated.with_name("VTAShellAPBAHB.sv")

    chisel_dir = repo / "hardware" / "chisel"
    env = os.environ.copy()
    chisel_dep = chisel_dir / "dep"
    for directory in ("coursier", "ivy", "sbt-boot", "sbt-global"):
        (chisel_dep / directory).mkdir(parents=True, exist_ok=True)
    env["COURSIER_CACHE"] = str(chisel_dep / "coursier")
    local_sbt_options = (
        f"-Dsbt.global.base={chisel_dep / 'sbt-global'} "
        f"-Dsbt.boot.directory={chisel_dep / 'sbt-boot'} "
        f"-Dsbt.ivy.home={chisel_dep / 'ivy'} "
        f"-Dsbt.coursier.home={chisel_dep / 'coursier'} "
        "-Dsbt.offline=true"
    )
    env["SBT_OPTS"] = " ".join(
        option for option in (env.get("SBT_OPTS", ""), local_sbt_options) if option)
    tool_prefix = Path(sys.base_prefix)
    java_home = tool_prefix / "lib" / "jvm"
    if java_home.is_dir():
        env["JAVA_HOME"] = str(java_home)
        env["JAVA_LD_LIBRARY_PATH"] = str(java_home / "lib" / "server")
    env["PATH"] = os.pathsep.join(
        (str(tool_prefix / "bin"), env.get("PATH", "")))
    env["VTA_SOC_MEMORY_PROTOCOL"] = memory_protocol
    env["VTA_SOC_MEMORY_BITS"] = str(memory_width)
    sbt = shutil.which("sbt")
    if sbt is None:
        candidate = Path(sys.base_prefix) / "bin" / "sbt"
        if candidate.is_file():
            sbt = str(candidate)
    if sbt is None:
        raise SystemExit(
            "sbt not found; install it in tvm_py310 or add it to PATH")
    sbt_command = (
        "runMain vta.socgen.SoCVariant "
        f"--target-dir {generated.parent} -o {generated.stem}"
    )
    add_generator = (
        "set Compile / unmanagedSourceDirectories += "
        f'file("{repo / "soc" / "generator"}")'
    )
    generate = [sbt, add_generator, sbt_command]
    importer = [
        sys.executable, str(repo / "soc" / "tools" / "import_vta_ip.py"),
        "--variant", variant, "--rtl", str(generated), "--vta-config",
        str(vta_config), "--output-root", str(ip_dir.parent),
        "--generator", "vta.socgen.SoCVariant",
    ]
    print("RTL variant:", variant)
    print("Generate:", " ".join(generate))
    print("Freeze:  ", " ".join(importer))
    if args.dry_run:
        return
    subprocess.run(generate, cwd=chisel_dir, env=env, check=True)
    subprocess.run(importer, check=True)


if __name__ == "__main__":
    main()
