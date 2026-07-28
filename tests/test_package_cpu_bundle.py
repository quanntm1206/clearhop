import json
import tempfile
import unittest
from pathlib import Path

from scripts.package_cpu_bundle import package_cpu_bundle


class TestPackageCpuBundle(unittest.TestCase):
    def test_package_is_hash_bound_and_replaces_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "scripts/enhance_cpu.py").write_text("print('ok')\n", encoding="ascii")
            (root / "src").mkdir()
            (root / "src/__init__.py").write_text("", encoding="ascii")
            (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="ascii")
            sources = {}
            for name, content in (("checkpoint.pth", b"ckpt"), ("model.ts", b"ts"), ("model.onnx", b"onnx"), ("config.yaml", b"loss: x\n")):
                path = root / name
                path.write_bytes(content)
                sources[name] = path
            output = root / "bundle"
            result = package_cpu_bundle(root, output, checkpoint=sources["checkpoint.pth"], torchscript=sources["model.ts"], onnx=sources["model.onnx"], config=sources["config.yaml"])
            self.assertEqual(result["status"], "pass")
            manifest = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["files"]["model.onnx"]["bytes"], 4)
            self.assertTrue((output / "checkpoint.pth").is_file())
            sources["model.onnx"].write_bytes(b"onnx-v2")
            package_cpu_bundle(root, output, checkpoint=sources["checkpoint.pth"], torchscript=sources["model.ts"], onnx=sources["model.onnx"], config=sources["config.yaml"])
            self.assertEqual((output / "model.onnx").read_bytes(), b"onnx-v2")

    def test_rejects_output_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "scripts/enhance_cpu.py").write_text("print('ok')\n", encoding="ascii")
            (root / "src").mkdir()
            (root / "src/__init__.py").write_text("", encoding="ascii")
            (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="ascii")
            files = []
            for index in range(4):
                path = root / f"f{index}"
                path.write_bytes(b"x")
                files.append(path)
            with self.assertRaises(ValueError):
                package_cpu_bundle(root, root.parent / "escape", checkpoint=files[0], torchscript=files[1], onnx=files[2], config=files[3])


if __name__ == "__main__":
    unittest.main()
