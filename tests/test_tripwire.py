import pytest
from mvt.applications.tripwire import VirtualTripwire


def box(x):
    return [x-5, 45, 10, 10]


def run(xs, margin=5):
    w = VirtualTripwire((100, 0), (100, 200), margin)
    for i, x in enumerate(xs, 1):
        w.update(box(x), i, 10)
    return w.events


def test_no_crossing():
    assert run([20, 50, 80]) == []


def test_bidirectional():
    events = run([80, 98, 120, 121, 80])
    assert [e.frame_number for e in events] == [3, 5]
    assert [e.direction for e in events] == ['positive_to_negative', 'negative_to_positive']
    assert events[0].timestamp == pytest.approx(.2)


def test_jitter():
    assert run([80, 97, 101, 99, 105, 80]) == []
    assert len(run([80, 101, 99, 103, 97, 80], 0)) == 4


def test_neutral_initialization():
    assert run([100, 101, 120]) == []


def test_invalid_and_gap():
    w = VirtualTripwire((100, 0), (100, 200))
    w.update(box(80), 1, 10)
    w.update([0, 0, -1, 2], 2, 10)
    assert w.update(box(120), 3, 10) is None
    assert w.update(box(80), 5, 10) is None
    with pytest.raises(ValueError):
        w.update(box(80), 5, 10)


def test_invalid_config():
    with pytest.raises(ValueError):
        VirtualTripwire((1, 1), (1, 1))
    with pytest.raises(ValueError):
        VirtualTripwire((1, 1), (2, 2), -1)
