"""Packaged CPU streaming runtime shared by CLI and desktop entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from .model import MobileDeepFilterNet


def enhance_waveform_cpu(model: "MobileDeepFilterNet", waveform: "torch.Tensor") -> "torch.Tensor":
    """Enhance one finite mono waveform with a fresh streaming state."""
    import torch

    from .inference import StreamingEnhancer

    with torch.no_grad():
        if waveform.ndim != 1 or not torch.is_floating_point(waveform) or not torch.isfinite(waveform).all():
            raise ValueError("waveform must be a finite one-dimensional floating-point tensor")
        model = model.cpu().eval()
        enhancer = StreamingEnhancer(model)
        outputs: list[torch.Tensor] = []
        for start in range(0, waveform.numel(), 160):
            hop = waveform[start : start + 160]
            if hop.numel() < 160:
                hop = torch.nn.functional.pad(hop, (0, 160 - hop.numel()))
            outputs.append(enhancer.push(hop.cpu()))
        return torch.cat(outputs)[: waveform.numel()] if outputs else waveform.clone()
