"""Build and verify the pinned RNNoise blocker recipe and public receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


RECEIPT_PATH = Path("reports/public/rnnoise_build.json")
CI_COMMAND = "python scripts/verify_rnnoise_build.py --rebuild"
REQUIRED_OUTPUTS = ("rnnoise_demo", "SHA256SUMS", "pinned_commit.txt", "COPYING", "README", "toolchain_packages.txt")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_receipt_sha256(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _registry_row(root: Path) -> dict[str, Any]:
    registry = _load(root / "configs/research_baselines.json")
    return next(row for row in registry["baselines"] if row["name"] == "RNNoise")


def _ref(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _file_sha256(root / relative)}


def _static_inputs(root: Path) -> dict[str, Any]:
    row = _registry_row(root)
    recipe = row["reproduction"]
    toolchain = _load(root / recipe["toolchain_manifest"])
    return {
        "registry": _ref(root, "configs/research_baselines.json"),
        "dockerfile": _ref(root, recipe["dockerfile"]),
        "toolchain_manifest": _ref(root, recipe["toolchain_manifest"]),
        "setup_script": _ref(root, recipe["setup_script"]),
        "platform": toolchain["platform"],
        "base_image": toolchain["base_image"],
        "apt_snapshot": toolchain["apt_snapshot"],
        "source": {
            "commit": toolchain["source_commit"],
            "archive_url": toolchain["source_archive_url"],
            "archive_sha256": toolchain["source_archive_sha256"],
        },
        "model": {
            "archive_url": toolchain["model_archive_url"],
            "archive_sha256": toolchain["model_archive_sha256"],
        },
        "packages": dict(sorted(toolchain["packages"].items())),
    }


def _parse_packages(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            raise ValueError("invalid toolchain package line")
        name, version = line.split("=", 1)
        packages[name] = version
    return dict(sorted(packages.items()))


def _parse_source_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", line.strip())
        if match is None:
            raise ValueError("invalid SHA256SUMS line")
        hashes[match.group(2)] = match.group(1)
    return dict(sorted(hashes.items()))


def build_receipt(root: Path, output_dir: Path) -> dict[str, Any]:
    inputs = _static_inputs(root)
    for name in REQUIRED_OUTPUTS:
        if not (output_dir / name).is_file():
            raise ValueError(f"RNNoise build output missing: {name}")
    if (output_dir / "pinned_commit.txt").read_text(encoding="utf-8").strip() != inputs["source"]["commit"]:
        raise ValueError("RNNoise output commit mismatch")
    packages = _parse_packages(output_dir / "toolchain_packages.txt")
    if packages != inputs["packages"]:
        raise ValueError("RNNoise package manifest mismatch")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "rnnoise_reproducible_build",
        "status": "pass",
        "inputs": inputs,
        "outputs": {
            "files": {name: _file_sha256(output_dir / name) for name in REQUIRED_OUTPUTS},
            "source_files": _parse_source_hashes(output_dir / "SHA256SUMS"),
            "packages": packages,
        },
    }
    receipt["receipt_sha256"] = canonical_receipt_sha256(receipt)
    return receipt


def _docker_build(root: Path, output_dir: Path) -> None:
    row = _registry_row(root)
    recipe = row["reproduction"]
    toolchain = _load(root / recipe["toolchain_manifest"])
    tag = f"clearhop-rnnoise-builder:{toolchain['source_commit'][:12]}"
    package_args = {
        "AUTOCONF_VERSION": toolchain["packages"]["autoconf"],
        "AUTOMAKE_VERSION": toolchain["packages"]["automake"],
        "BUILD_ESSENTIAL_VERSION": toolchain["packages"]["build-essential"],
        "CA_CERTIFICATES_VERSION": toolchain["packages"]["ca-certificates"],
        "CURL_VERSION": toolchain["packages"]["curl"],
        "LIBTOOL_VERSION": toolchain["packages"]["libtool"],
        "PKG_CONFIG_VERSION": toolchain["packages"]["pkg-config"],
    }
    build_args = {
        "BASE_IMAGE": toolchain["base_image"],
        "APT_SNAPSHOT_URL": toolchain["apt_snapshot"],
        "RNNOISE_COMMIT": toolchain["source_commit"],
        "RNNOISE_ARCHIVE_URL": toolchain["source_archive_url"],
        "RNNOISE_ARCHIVE_SHA256": toolchain["source_archive_sha256"],
        "RNNOISE_MODEL_URL": toolchain["model_archive_url"],
        "RNNOISE_MODEL_SHA256": toolchain["model_archive_sha256"],
        **package_args,
    }
    command = [
        "docker", "build", "--pull=false", "--platform", toolchain["platform"],
        "--file", str(root / recipe["dockerfile"]), "--tag", tag,
    ]
    for name, value in build_args.items():
        command.extend(("--build-arg", f"{name}={value}"))
    command.append(str(root))
    subprocess.run(command, cwd=root, check=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command = ["docker", "run", "--rm"]
    if os.name != "nt":
        run_command.extend(("--user", f"{os.getuid()}:{os.getgid()}"))
    run_command.extend(("--volume", f"{output_dir.resolve()}:/export", tag, "sh", "-c", "cp -a /out/. /export/"))
    subprocess.run(
        run_command,
        cwd=root,
        check=True,
    )


def audit_rnnoise_build(
    root: Path,
    *,
    receipt: dict[str, Any] | None = None,
    workflow_text: str | None = None,
    rebuilt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    try:
        public = receipt if receipt is not None else _load(root / RECEIPT_PATH)
        current_inputs = _static_inputs(root)
        row = _registry_row(root)
        recipe = row["reproduction"]
        toolchain = _load(root / recipe["toolchain_manifest"])
        if workflow_text is None:
            workflow_text = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        linux_job = workflow_text.split("\n  windows:", 1)[0]
        receipt_outputs = public.get("outputs") if isinstance(public, dict) else None
        checks = {
            "receipt_hash": isinstance(public, dict) and public.get("receipt_sha256") == canonical_receipt_sha256(public),
            "receipt_inputs": isinstance(public, dict) and public.get("inputs") == current_inputs,
            "output_contract": (
                isinstance(receipt_outputs, dict)
                and isinstance(receipt_outputs.get("files"), dict)
                and set(receipt_outputs["files"]) == set(REQUIRED_OUTPUTS)
                and all(re.fullmatch(r"[0-9a-f]{64}", value or "") for value in receipt_outputs["files"].values())
                and receipt_outputs.get("packages") == current_inputs["packages"]
            ),
            "snapshot_pin": (
                re.fullmatch(r"https://snapshot\.ubuntu\.com/ubuntu/[0-9]{8}T[0-9]{6}Z/", str(toolchain.get("apt_snapshot"))) is not None
                and re.fullmatch(r"ubuntu@sha256:[0-9a-f]{64}", str(toolchain.get("base_image"))) is not None
            ),
            "registry_bindings": (
                recipe.get("setup_script_sha256") == current_inputs["setup_script"]["sha256"]
                and recipe.get("dockerfile_sha256") == current_inputs["dockerfile"]["sha256"]
                and recipe.get("toolchain_manifest_sha256") == current_inputs["toolchain_manifest"]["sha256"]
            ),
            "workflow_contract": CI_COMMAND in linux_job,
            "rebuild_match": rebuilt is None or rebuilt == public,
        }
    except Exception as exc:
        return {"schema_version": 1, "status": "fail", "checks": {}, "error": f"{type(exc).__name__}: {exc}"}
    return {"schema_version": 1, "status": "pass" if all(checks.values()) else "fail", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    rebuilt = None
    if args.rebuild:
        with tempfile.TemporaryDirectory(prefix="clearhop-rnnoise-") as tmp:
            output = Path(tmp) / "out"
            _docker_build(root, output)
            rebuilt = build_receipt(root, output)
        if args.write_receipt:
            destination = root / RECEIPT_PATH
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(rebuilt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    receipt = rebuilt if args.write_receipt and rebuilt is not None else None
    result = audit_rnnoise_build(root, receipt=receipt, rebuilt=None if args.write_receipt else rebuilt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
