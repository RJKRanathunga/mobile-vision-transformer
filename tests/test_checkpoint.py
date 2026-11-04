from collections import OrderedDict

import torch

from mvt.checkpoint import UPSTREAM_COMMIT, load_mvt_checkpoint
from mvt.models.mvt import MVT


def test_state_dict_has_no_duplicate_mobilenet_paths():
    state = MVT().state_dict()
    assert len(state) == 334
    assert not any(".expansion." in key for key in state)
    assert not any(".depthwise." in key for key in state)
    assert not any(".projection." in key for key in state)


def test_order_validated_upstream_checkpoint_conversion():
    source_model = MVT()
    target_model = MVT()
    source_state = source_model.state_dict()
    upstream_state = OrderedDict()
    for index, value in enumerate(source_state.values()):
        if index == 0:
            key = "backbone.conv_1.block.conv.weight"
        else:
            key = f"upstream.tensor_{index:04d}"
        upstream_state[key] = value.clone()

    report = load_mvt_checkpoint(
        target_model,
        {"net": upstream_state},
    )

    assert report.format == "upstream"
    assert report.upstream_commit == UPSTREAM_COMMIT
    for expected, actual in zip(
        source_model.state_dict().values(),
        target_model.state_dict().values(),
    ):
        assert torch.equal(expected, actual)
