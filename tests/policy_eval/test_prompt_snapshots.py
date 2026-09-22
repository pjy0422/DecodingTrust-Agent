from __future__ import annotations

import asyncio
import json

from dt_arena.policy_eval.prompt_snapshots import PromptSnapshotWriter


def test_prompt_snapshot_wrapper_retains_prompt_but_not_provider_state(tmp_path):
    writer = PromptSnapshotWriter(tmp_path)

    async def complete(prompt: str) -> dict[str, str]:
        return {"echo": prompt}

    wrapped = writer.wrap(
        complete,
        component="digestor",
        label="Digestor request prompt",
        role="user",
        source="feedback.digestor",
    )
    assert asyncio.run(wrapped("exact request")) == {"echo": "exact request"}
    value = json.loads((tmp_path / "prompt-snapshots.jsonl").read_text(encoding="utf-8"))
    assert value == {
        "component": "digestor",
        "exact": True,
        "label": "Digestor request prompt",
        "prompt": "exact request",
        "role": "user",
        "schema": "dtap-policy-eval-prompt-snapshot",
        "schema_version": 1,
        "sequence": 1,
        "source": "feedback.digestor",
    }


def test_prompt_snapshot_records_active_submission_attempt(tmp_path):
    writer = PromptSnapshotWriter(tmp_path)

    async def complete(prompt: str) -> str:
        return prompt

    wrapped = writer.wrap(
        complete,
        component="digestor",
        label="Digestor request prompt",
        source="feedback.digestor",
    )
    with writer.attempt(2):
        assert asyncio.run(wrapped("H2 request")) == "H2 request"
    value = json.loads((tmp_path / "prompt-snapshots.jsonl").read_text(encoding="utf-8"))
    assert value["attempt_index"] == 2
