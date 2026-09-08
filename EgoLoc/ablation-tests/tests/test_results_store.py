import json

import pytest

import results_store


def record(key, config_hash, status="success"):
    return {
        "key": key,
        "status": status,
        "config_hash": config_hash,
        "prediction_frame": 3 if status == "success" else None,
        "request_traces": [],
    }


def test_append_resume_and_duplicate_rejection(tmp_path):
    path = tmp_path / "results.jsonl"
    config_hash = results_store.stable_config_hash({"locked": True})
    key = results_store.result_key("7", "contact", "speed", 0)
    store = results_store.ResultsStore(path, config_hash=config_hash)
    store.append(record(key, config_hash))

    resumed = results_store.ResultsStore(path, config_hash=config_hash)
    assert key in resumed
    assert resumed.resume_index[key]["prediction_frame"] == 3
    with pytest.raises(results_store.DuplicateResultError):
        resumed.append(record(key, config_hash))
    assert len(path.read_text().strip().splitlines()) == 1


def test_torn_final_line_is_ignored_then_repaired_on_append(tmp_path):
    path = tmp_path / "results.jsonl"
    config_hash = results_store.stable_config_hash({"version": 1})
    key_a = results_store.result_key(0, "contact", "speed", 0)
    key_b = results_store.result_key(0, "contact", "pinch", 0)
    first = json.dumps(record(key_a, config_hash), separators=(",", ":"))
    path.write_bytes((first + "\n" + '{"key":"torn').encode())

    store = results_store.ResultsStore(path, config_hash=config_hash)
    assert key_a in store
    assert "torn" not in store
    store.append(record(key_b, config_hash))

    loaded, torn = results_store.read_jsonl_tolerant(path)
    assert torn is False
    assert [item["key"] for item in loaded] == [key_a, key_b]


def test_config_mismatch_refuses_resume(tmp_path):
    path = tmp_path / "results.jsonl"
    first_hash = results_store.stable_config_hash({"version": 1})
    second_hash = results_store.stable_config_hash({"version": 2})
    store = results_store.ResultsStore(path, config_hash=first_hash)
    store.append(
        record(
            results_store.result_key(0, "separation", "pinch", 2),
            first_hash,
        )
    )

    with pytest.raises(results_store.ConfigMismatchError, match="new output path"):
        results_store.ResultsStore(path, config_hash=second_hash)
