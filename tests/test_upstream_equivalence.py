import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mvt.data.processing_utils import sample_target
from mvt.losses.target import generate_heatmap
from mvt.models.backbone.mvt_backbone import MVTBackbone
from mvt.models.head.mvt_head import CenterPredictor as LocalHead
from mvt.models.neck.mvt_neck import FeatureFusor, MVTNeck


UPSTREAM_ENV = "MVT_UPSTREAM_DIR"


def _upstream_root() -> Path:
    value = os.environ.get(UPSTREAM_ENV)
    if not value:
        pytest.skip(f"{UPSTREAM_ENV} is not set")
    root = Path(value)
    if not root.exists():
        pytest.skip(f"{UPSTREAM_ENV} does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _copy_state_by_tensor_order(upstream, local):
    upstream_parameters = list(upstream.parameters())
    local_parameters = list(local.parameters())
    upstream_buffers = list(upstream.buffers())
    local_buffers = list(local.buffers())
    assert [p.shape for p in upstream_parameters] == [
        p.shape for p in local_parameters
    ]
    assert [b.shape for b in upstream_buffers] == [
        b.shape for b in local_buffers
    ]
    with torch.no_grad():
        for source, target in zip(upstream_parameters, local_parameters):
            target.copy_(source)
        for source, target in zip(upstream_buffers, local_buffers):
            target.copy_(source)


def test_pinned_upstream_backbone_forward_equivalence():
    _upstream_root()
    from lib.models.mobilevit_track.mobilevit import MobileViT

    opts = {
        "mode": "small",
        "head_dim": None,
        "number_heads": 4,
        "conv_layer_normalization_name": "batch_norm",
        "conv_layer_activation_name": "relu",
    }
    upstream = MobileViT(opts)
    local = MVTBackbone()
    _copy_state_by_tensor_order(upstream, local)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream = upstream.eval().to(device)
    local = local.eval().to(device)
    generator = torch.Generator(device=device).manual_seed(17)
    search = torch.randn(
        1, 3, 256, 256, generator=generator, device=device
    )
    template = torch.randn(
        1, 3, 128, 128, generator=generator, device=device
    )

    with torch.inference_mode():
        upstream_template = upstream.layer_2(
            upstream.layer_1(upstream.conv_1(template))
        )
        upstream_search, upstream_template = upstream(
            search,
            upstream_template,
        )
        local_search, local_template = local(search, template)

    assert torch.equal(upstream_search, local_search)
    assert torch.equal(upstream_template, local_template)


def test_pinned_upstream_neck_fusor_and_head_equivalence():
    _upstream_root()
    from lib.models.mobilevit_track.layers.head import (
        CenterPredictor as UpstreamHead,
    )
    from lib.models.mobilevit_track.layers.neck_lighttrack import (
        BN_adj,
        PW_Corr_adj,
    )

    upstream_neck = BN_adj(128)
    local_neck = MVTNeck(128)
    upstream_fusor = PW_Corr_adj(num_kernel=64, adj_channel=256)
    local_fusor = FeatureFusor(128, 64, 256)
    upstream_head = UpstreamHead(
        inplanes=256,
        channel=256,
        feat_sz=16,
        stride=16,
    )
    local_head = LocalHead(
        in_channels=256,
        hidden_channels=256,
        feat_size=16,
    )
    pairs = [
        (upstream_neck, local_neck),
        (upstream_fusor, local_fusor),
        (upstream_head, local_head),
    ]
    for upstream, local in pairs:
        _copy_state_by_tensor_order(upstream, local)
        upstream.eval()
        local.eval()

    search = torch.randn(2, 128, 16, 16)
    template = torch.randn(2, 128, 8, 8)
    with torch.inference_mode():
        upstream_search, upstream_template = upstream_neck(search, template)
        local_search, local_template = local_neck(search, template)
        upstream_fused = upstream_fusor(
            upstream_template,
            upstream_search,
        )
        local_fused = local_fusor(local_template, local_search)
        upstream_outputs = upstream_head(upstream_fused)
        local_outputs = local_head(local_fused)

    assert torch.equal(upstream_search, local_search)
    assert torch.equal(upstream_template, local_template)
    assert torch.equal(upstream_fused, local_fused)
    for upstream_output, local_output in zip(
        upstream_outputs,
        local_outputs,
    ):
        assert torch.equal(upstream_output, local_output)


def test_pinned_upstream_crop_and_boundary_mask_equivalence():
    root = _upstream_root()
    path = root / "lib" / "train" / "data" / "processing_utils.py"
    spec = importlib.util.spec_from_file_location("upstream_prutils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, (73, 91, 3), dtype=np.uint8)
    for box in (
        [20.3, 11.7, 17.2, 24.6],
        [-4.2, -3.0, 12.0, 9.0],
        [82.0, 65.0, 15.0, 14.0],
    ):
        upstream = module.sample_target(image, box, 4.0, 64)
        local = sample_target(image, box, 4.0, 64)
        assert np.array_equal(upstream[0], local[0])
        assert upstream[1] == local[1]
        assert np.array_equal(upstream[2], local[2])


def test_pinned_upstream_heatmap_compatibility_mode():
    _upstream_root()
    from lib.utils.heapmap_utils import generate_heatmap as upstream_heatmap

    boxes = torch.tensor(
        [
            [0.10, 0.20, 0.35, 0.45],
            [0.80, 0.75, 0.15, 0.20],
        ],
        dtype=torch.float32,
    )
    upstream = upstream_heatmap(boxes.unsqueeze(0), 256, 16)[0]
    local = generate_heatmap(
        boxes,
        heatmap_size=16,
        upstream_bug_compatible=True,
    ).squeeze(1)
    assert torch.equal(upstream, local)
