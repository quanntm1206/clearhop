"""Create an atomic, hash-bound CPU deployment bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from src.checkpoint import file_sha256


def package_cpu_bundle(
    root: Path,
    output: Path,
    *,
    checkpoint: Path,
    torchscript: Path,
    onnx: Path,
    config: Path,
) -> dict[str, object]:
    root = Path(root).resolve()
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if root not in output.parents:
        raise ValueError("bundle output must stay inside project root")
    inputs = {
        "checkpoint.pth": Path(checkpoint),
        "model.ts": Path(torchscript),
        "model.onnx": Path(onnx),
        "config.yaml": Path(config),
        "enhance_cpu.py": root / "scripts/enhance_cpu.py",
        "pyproject.toml": root / "pyproject.toml",
    }
    for name, path in inputs.items():
        path = path if path.is_absolute() else root / path
        inputs[name] = path.resolve()
        if not path.is_file() or root not in path.parents:
            raise ValueError(f"missing or out-of-root bundle input: {name}")
    source_package = root / "src"
    if not source_package.is_dir():
        raise ValueError("missing src runtime package")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    backup: Path | None = None
    try:
        manifest: dict[str, object] = {"schema_version": 1, "files": {}}
        for name, source in inputs.items():
            target = temp / name
            shutil.copy2(source, target)
            manifest["files"][name] = {"sha256": file_sha256(target), "bytes": target.stat().st_size}  # type: ignore[index]
        shutil.copytree(source_package, temp / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for runtime_file in sorted((temp / "src").rglob("*.py")):
            name = runtime_file.relative_to(temp).as_posix()
            manifest["files"][name] = {"sha256": file_sha256(runtime_file), "bytes": runtime_file.stat().st_size}  # type: ignore[index]
        readme = temp / "README.txt"
        readme.write_text("CPU bundle. Run: python enhance_cpu.py --checkpoint checkpoint.pth --input input.wav --output output.wav\nInput contract: mono 16 kHz WAV.\n", encoding="ascii")
        manifest["files"]["README.txt"] = {"sha256": file_sha256(readme), "bytes": readme.stat().st_size}  # type: ignore[index]
        manifest["checkpoint_sha256"] = manifest["files"]["checkpoint.pth"]["sha256"]  # type: ignore[index]
        (temp / "bundle.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output.exists():
            if not output.is_dir():
                raise ValueError("bundle output exists and is not a directory")
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup.", dir=str(output.parent)))
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(temp, output)
        except Exception:
            if backup is not None and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        temp = None  # type: ignore[assignment]
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
        return {"schema_version": 1, "status": "pass", "bundle": str(output), **manifest}
    finally:
        if temp is not None and temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--torchscript", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cpu_bundle"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = package_cpu_bundle(root, args.output, checkpoint=args.checkpoint, torchscript=args.torchscript, onnx=args.onnx, config=args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
