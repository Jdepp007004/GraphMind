"""tests/test_event_dataset.py — EventDataset interface + SyntheticDataset."""
import pytest
from src.data.event_dataset import EventDataset, SyntheticDataset, DeviceAnalyzerDataset


def test_synthetic_dataset_loads_without_crash():
    ds = SyntheticDataset()
    ds.load()
    meta = ds.metadata()
    assert meta["loaded"] is True
    assert meta["total_events"] >= 0


def test_synthetic_dataset_idempotent_load():
    ds = SyntheticDataset()
    ds.load()
    total_1 = ds.metadata()["total_events"]
    ds.load()  # second call
    total_2 = ds.metadata()["total_events"]
    assert total_1 == total_2


def test_chronological_split_ratios():
    events = [{"timestamp": float(i), "app_id": f"app_{i}", "battery": 100.0,
               "time_bucket": 0, "headphones": False, "calendar_event_in_mins": None,
               "weekend": False} for i in range(100)]
    splits = EventDataset._chronological_split(events)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10


def test_chronological_split_preserves_order():
    events = [{"timestamp": float(i)} for i in range(100)]
    splits = EventDataset._chronological_split(events)
    # First train event timestamp < first val event timestamp
    assert splits["train"][-1]["timestamp"] < splits["val"][0]["timestamp"]
    assert splits["val"][-1]["timestamp"] < splits["test"][0]["timestamp"]


def test_synthetic_dataset_event_schema():
    ds = SyntheticDataset()
    ds.load()
    required = {"app_id", "battery", "time_bucket", "headphones", "weekend"}
    for event in ds.iter_events("train"):
        assert required.issubset(event.keys()), f"Missing keys in event: {event}"
        break  # only check first event


def test_device_analyzer_dataset_fallback():
    """Without raw data, DeviceAnalyzerDataset falls back to synthetic."""
    ds = DeviceAnalyzerDataset(fallback_to_synthetic=True)
    ds.load()
    meta = ds.metadata()
    assert meta["loaded"] is True
    # Either real data or fallback
    assert meta["source"] in ("device_analyzer", "synthetic_fallback")


def test_get_splits_returns_dict():
    ds = SyntheticDataset()
    ds.load()
    splits = ds.get_splits()
    assert set(splits.keys()) == {"train", "val", "test"}
    assert isinstance(splits["train"], list)


def test_iter_events_all():
    ds = SyntheticDataset()
    ds.load()
    all_events = list(ds.iter_events("all"))
    train = list(ds.iter_events("train"))
    val = list(ds.iter_events("val"))
    test = list(ds.iter_events("test"))
    assert len(all_events) == len(train) + len(val) + len(test)
