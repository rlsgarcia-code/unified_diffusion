import json

from unified_diffusion.telemetry import log_event


def test_log_event_emits_json(capsys) -> None:
    log_event("test.event", model="sdxl.base")

    captured = capsys.readouterr().err.strip()
    payload = json.loads(captured)
    assert payload["event"] == "test.event"
    assert payload["model"] == "sdxl.base"
