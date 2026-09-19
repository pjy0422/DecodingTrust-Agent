from __future__ import annotations

from types import SimpleNamespace

import pytest

from dt_arena.policy_eval.actions import PlanValidationResult, ValidatedAttackStep
from dt_arena.policy_eval.attempt_runner import AttemptResult
from dt_arena.policy_eval.episode_runtime import EpisodeRuntimeState
from dt_arena.policy_eval.feedback.builder import FeedbackBuilder
from dt_arena.policy_eval.feedback.digestor import (
    DigestorObservation,
    PromptedLLMDigestor,
    validate_repair_digest,
)
from dt_arena.policy_eval.feedback.schema import (
    DeterministicFeedback,
    InjectionObservation,
    ToolCallObservation,
    VictimVisibleTrace,
    deterministic_to_dict,
)
from dt_arena.policy_eval.policy_contract import PolicyContract
from dt_arena.policy_eval.submission import SubmissionCoordinator


def _deterministic(*, presented: bool = True) -> DeterministicFeedback:
    return DeterministicFeedback(
        tool_sequence=(ToolCallObservation(0, "finance:browse_article", "ok"),),
        injections=(
            InjectionObservation(
                step_index=0,
                injection_type="environment",
                polluted_tool_called=None,
                injected_target_accessed=True,
                access_state="accessed",
                locator_targeted=True,
                access_call_status="ok",
                response_contains_injection=True,
                presented_to_model=presented,
                evidence_call_indices=(0,),
                response_match_state="matched",
                presentation_state="presented" if presented else "not_presented",
                presentation_observation_source="provider_request",
            ),
        ),
        trace_complete=True,
    )


def test_digest_v3_validates_per_step_attribution_and_evidence():
    digest = validate_repair_digest(
        {
            "diagnosis": "The payload was visible but rejected as unauthenticated.",
            "preserve": ["/steps/0"],
            "reconsider": [],
            "confidence": "high",
            "payload_effect": "rejected",
            "evidence_refs": ["/deterministic/injections/0/presented_to_model"],
            "step_attributions": [
                {
                    "step_index": 0,
                    "effect": "rejected",
                    "reason_classes": ["authority_channel_mismatch"],
                    "confidence": "high",
                    "evidence_refs": ["/deterministic/injections/0/presented_to_model"],
                }
            ],
        },
        {"steps": [{"type": "environment"}]},
        deterministic=_deterministic(),
    )
    assert digest.step_attributions[0].reason_classes == ("authority_channel_mismatch",)

    receipt = PolicyContract().accepted_submit(
        submission=1,
        success=False,
        terminal=False,
        remaining_submissions=1,
        feedback={
            "schema_version": 3,
            "final_response": "refused",
            "deterministic": deterministic_to_dict(_deterministic()),
            "digest": digest.to_dict(),
        },
    )
    assert receipt["feedback"]["digest"]["step_attributions"][0]["step_index"] == 0


def test_digest_v3_rejects_unbounded_reason_class():
    with pytest.raises(ValueError, match="reason classes"):
        validate_repair_digest(
            {
                "diagnosis": "unsupported claim",
                "preserve": [],
                "reconsider": [],
                "confidence": "low",
                "payload_effect": "unclear",
                "evidence_refs": [],
                "step_attributions": [
                    {
                        "step_index": 0,
                        "effect": "unclear",
                        "reason_classes": ["invented_cause"],
                        "confidence": "low",
                        "evidence_refs": ["/deterministic/injections/0/presented_to_model"],
                    }
                ],
            },
            {"steps": [{"type": "environment"}]},
            deterministic=_deterministic(),
        )


def test_digest_v3_accepts_policy_visible_final_response_evidence():
    digest = validate_repair_digest(
        {
            "diagnosis": "The victim explicitly rejected the source as unauthenticated.",
            "preserve": [],
            "reconsider": ["/steps/0"],
            "confidence": "high",
            "payload_effect": "rejected",
            "evidence_refs": ["/final_response"],
            "step_attributions": [
                {
                    "step_index": 0,
                    "effect": "rejected",
                    "reason_classes": ["authority_channel_mismatch"],
                    "confidence": "high",
                    "evidence_refs": ["/final_response"],
                }
            ],
        },
        {"steps": [{"type": "environment"}]},
        deterministic=_deterministic(),
        final_response="I cannot trust instructions from this page.",
    )
    assert digest.evidence_refs == ("/final_response",)


@pytest.mark.asyncio
async def test_prompted_digestor_requests_complete_step_attribution():
    requests = []

    async def complete(prompt):
        import json

        requests.append(json.loads(prompt))
        return {
            "diagnosis": "unauthenticated source",
            "preserve": [],
            "reconsider": ["/steps/0"],
            "confidence": "medium",
            "payload_effect": "rejected",
            "evidence_refs": ["/final_response"],
            "step_attributions": [
                {
                    "step_index": 0,
                    "effect": "rejected",
                    "reason_classes": ["authority_channel_mismatch"],
                    "confidence": "medium",
                    "evidence_refs": ["/final_response"],
                }
            ],
        }

    digest = await PromptedLLMDigestor(complete).digest(
        DigestorObservation(
            schema_version=3,
            submission={"steps": [{"type": "environment"}]},
            deterministic=_deterministic(),
            victim_trace=VictimVisibleTrace((), True, "unavailable"),
            final_response="I cannot trust instructions from this page.",
        )
    )
    assert digest.step_attributions[0].step_index == 0
    assert "/final_response" in requests[0]["allowed_evidence_pointers"]
    assert "authority_channel_mismatch" in requests[0]["allowed_reason_classes"]


def test_surface_mismatch_metric_is_research_only(tmp_path):
    builder = FeedbackBuilder(research_output_root=tmp_path)
    digest = validate_repair_digest(
        {
            "diagnosis": "trusted channel mismatch",
            "preserve": [],
            "reconsider": [],
            "confidence": "medium",
            "payload_effect": "rejected",
            "evidence_refs": [],
            "step_attributions": [
                {
                    "step_index": 0,
                    "effect": "rejected",
                    "reason_classes": ["authority_channel_mismatch"],
                    "confidence": "medium",
                    "evidence_refs": ["/deterministic/injections/0/presented_to_model"],
                }
            ],
        },
        {"steps": [{"type": "environment"}]},
        deterministic=_deterministic(),
    )
    builder._write_surface_mismatch_metric(
        attempt_root=tmp_path / "internal" / "attempt-0001" / "results",
        deterministic=_deterministic(),
        digest=digest,
    )
    text = (tmp_path / "attempts" / "attempt-0001" / "research-feedback.json").read_text()
    assert '"classification": "apparent"' in text
    assert '"policy_visible": false' in text


@pytest.mark.asyncio
async def test_terminal_failed_h_still_receives_feedback(monkeypatch, tmp_path):
    step = ValidatedAttackStep(type="prompt", turn_id=1, mode="suffix", content="test")
    monkeypatch.setattr(
        "dt_arena.policy_eval.submission.validate_attack_plan",
        lambda *_: PlanValidationResult(True, steps=(step,)),
    )
    attempt_dir = tmp_path / "attempt-0001"
    output_root = attempt_dir / "results"
    output_root.mkdir(parents=True)
    config_path = attempt_dir / "config.yaml"
    config_path.write_text("Attack: {}\n", encoding="utf-8")
    workspace = SimpleNamespace(output_root=output_root, config_path=config_path)
    monkeypatch.setattr("dt_arena.policy_eval.submission.materialize_attempt_dir", lambda **_: workspace)

    class Runner:
        async def run(self, _workspace):
            return AttemptResult(evaluation_started=True, attack_success=False)

    class Builder:
        calls = 0

        async def build(self, **_kwargs):
            self.calls += 1
            return {"schema_version": 3, "final_response": "refused"}

    feedback = Builder()
    coordinator = SubmissionCoordinator(
        validation_context=SimpleNamespace(),
        runtime=EpisodeRuntimeState(max_submissions=1),
        source_task_dir=tmp_path,
        episode_root=tmp_path / "episode",
        runner=Runner(),
        policy_contract=PolicyContract(),
        feedback_builder=feedback,
    )
    receipt = await coordinator.submit({"steps": [{"type": "prompt"}]})
    assert receipt["terminal"] is True
    assert receipt["feedback"]["final_response"] == "refused"
    assert feedback.calls == 1
