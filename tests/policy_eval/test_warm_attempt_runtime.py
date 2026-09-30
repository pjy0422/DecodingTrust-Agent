from __future__ import annotations

import io
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from dt_arena.policy_eval.attempt_runner import DtapAttemptRunner
from dt_arena.policy_eval.experiment_config import load_experiment_config
from dt_arena.policy_eval.scripts import run_dtap_attempt
from dt_arena.policy_eval.scripts import run_dtap_attempt_worker as worker
from dt_arena.policy_eval.scripts import run_domain_matrix
from eval.task_runner import ReusableTaskServices
from utils.task_executor import EnvInstance, EnvState, ScheduledTask, TaskExecutor


class _Process:
    def __init__(self) -> None:
        self.stopped = False

    def poll(self):
        return 0 if self.stopped else None


class _Manager:
    def __init__(self) -> None:
        self.processes = {"server": _Process()}
        self.stop_calls = 0

    def stop_all(self) -> None:
        self.stop_calls += 1
        for process in self.processes.values():
            process.stopped = True


class _Resources:
    def __init__(self) -> None:
        self.cleaned: list[str] = []

    def cleanup_task(self, task_id: str) -> None:
        self.cleaned.append(task_id)


def _agent(*, token: str = "same"):
    server = SimpleNamespace(name="gmail", enabled=True, env={"TOKEN": token}, url=None)
    return SimpleNamespace(mcp_servers=[server], sub_agents=[]), server


def test_reusable_task_services_rebind_fresh_agent_config_and_close() -> None:
    resources = _Resources()
    services = ReusableTaskServices(resources)
    original, original_server = _agent()
    original_server.url = "http://127.0.0.1:23001/mcp"
    manager = _Manager()
    services.retain_task_servers(original, manager, "task-1")

    fresh, fresh_server = _agent()
    assert services.bind_task_servers(fresh) is manager
    assert fresh_server.url == original_server.url

    services.close()
    assert manager.stop_calls == 1
    assert resources.cleaned == ["task-1"]


def test_reusable_task_services_restart_on_config_change() -> None:
    services = ReusableTaskServices(_Resources())
    original, _ = _agent(token="one")
    manager = _Manager()
    services.retain_task_servers(original, manager, "task-1")

    changed, _ = _agent(token="two")
    assert services.bind_task_servers(changed) is None
    assert manager.stop_calls == 1


@pytest.mark.asyncio
async def test_task_executor_can_retain_idle_instances_between_batches(monkeypatch) -> None:
    executor = TaskExecutor(max_parallel=1, retain_idle_instances=True)
    stopped: list[str] = []

    async def stop(instance) -> None:
        stopped.append(instance.instance_id)

    monkeypatch.setattr(executor, "_stop_instance", stop)
    await executor._cleanup_unused_instances({"gmail"})
    assert stopped == []


@pytest.mark.asyncio
async def test_two_batches_reset_one_reused_environment(monkeypatch, tmp_path: Path) -> None:
    executor = TaskExecutor(max_parallel=1, retain_idle_instances=True)
    starts: list[str] = []
    resets: list[str] = []
    stops: list[str] = []

    async def start(env_name: str) -> EnvInstance:
        starts.append(env_name)
        instance = EnvInstance(
            instance_id="gmail:test",
            env_name=env_name,
            project_name="test-project",
            compose_file=tmp_path / "compose.yaml",
            state=EnvState.AVAILABLE,
        )
        executor._instances[instance.instance_id] = instance
        executor._env_instances[env_name].append(instance.instance_id)
        return instance

    async def reset(instance: EnvInstance) -> None:
        resets.append(instance.instance_id)

    async def stop(instance: EnvInstance) -> None:
        stops.append(instance.instance_id)

    async def run(_task, instances) -> int:
        assert instances["gmail"].state == EnvState.IN_USE
        return 0

    monkeypatch.setattr(executor, "_start_instance", start)
    monkeypatch.setattr(executor, "_reset_instance", reset)
    monkeypatch.setattr(executor, "_stop_instance", stop)
    for index in (1, 2):
        task = ScheduledTask(
            task_dir=tmp_path / f"attempt-{index}",
            environments=frozenset({"gmail"}),
            original_index=0,
        )
        assert (await executor.run_all([task], run))[0][1] == 0
    assert starts == ["gmail"]
    assert resets == ["gmail:test", "gmail:test"]
    assert stops == []
    await executor.shutdown()
    assert stops == ["gmail:test"]


@pytest.mark.asyncio
async def test_worker_runs_two_requests_with_distinct_output_roots(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    seen: list[tuple[int, int, str]] = []

    async def fake_run(args, *, executor, configure_provider, reusable_services) -> int:
        assert configure_provider is False
        root = Path(worker.os.environ["EVAL_RESULTS_ROOT"])
        args.started_path.write_bytes(b"1")
        print(f"private attempt {root.name}")
        seen.append((id(executor), id(reusable_services), root.name))
        return 0

    requests = [
        {
            "command": "run",
            "request_id": f"req-{index}",
            "task_dir": str(tmp_path),
            "output_root": str(tmp_path / f"attempt-{index}"),
            "attempt_index": index,
            "agent_type": "openaisdk",
            "model": "test-model",
            "max_turns": 1,
            "started_path": str(tmp_path / f"attempt-{index}" / ".m4-started"),
        }
        for index in (1, 2)
    ]
    input_lines = "".join(json.dumps(item) + "\n" for item in requests)
    input_lines += '{"command":"shutdown"}\n'
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(input_lines))
    monkeypatch.setattr(worker, "_configure_victim_provider", lambda _model: None)
    monkeypatch.setattr(worker, "_run", fake_run)
    assert await worker._serve("test-model") == 0

    output = capsys.readouterr().out
    assert output.count(worker.RESULT_PREFIX) == 2
    assert "private attempt" not in output
    assert seen[0][:2] == seen[1][:2]
    assert [item[2] for item in seen] == ["attempt-1", "attempt-2"]
    for index in (1, 2):
        root = tmp_path / f"attempt-{index}"
        assert (root / ".m4-started").read_bytes() == b"1"
        assert (root / ".dtap-run.log").read_text().strip() == f"private attempt attempt-{index}"


@pytest.mark.asyncio
async def test_medical_aux_credentials_survive_multiple_warm_attempts(
    monkeypatch, tmp_path: Path
) -> None:
    task_dir = Path(__file__).resolve().parents[2] / (
        "dataset/medical/malicious/indirect/upcoding-billing/047"
    )
    monkeypatch.setenv(run_dtap_attempt.MEDICAL_AUX_MODE_ENV, "deterministic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    seen: list[tuple[str, str]] = []

    async def fake_single_task(*_args, **_kwargs) -> int:
        seen.append((
            run_dtap_attempt.os.environ.get("ANTHROPIC_API_KEY", ""),
            run_dtap_attempt.os.environ.get("OPENAI_API_KEY", ""),
        ))
        return 0

    class FakeExecutor:
        async def run_all(self, tasks, callback):
            return [(tasks[0], await callback(tasks[0], {}))]

    monkeypatch.setattr("eval.task_runner.run_single_task", fake_single_task)
    args = SimpleNamespace(
        task_dir=task_dir,
        agent_type="openclaw",
        model="test-model",
        temperature=None,
        max_turns=1,
        debug=False,
        started_path=None,
        verdict_path=None,
    )
    for _ in (1, 2):
        assert await run_dtap_attempt._run(
            args, executor=FakeExecutor(), configure_provider=False
        ) == 0
        assert run_dtap_attempt.os.environ["ANTHROPIC_API_KEY"] == "test-anthropic-key"
        assert run_dtap_attempt.os.environ["OPENAI_API_KEY"] == "test-openai-key"
        monkeypatch.setenv(run_dtap_attempt.MEDICAL_AUX_MODE_ENV, "deterministic")
    assert seen == [("test-anthropic-key", "test-openai-key")] * 2


@pytest.mark.asyncio
async def test_warm_failure_before_victim_start_falls_back_once(monkeypatch, tmp_path: Path) -> None:
    runner = DtapAttemptRunner(attempt_runtime="warm")
    cold_calls: list[int] = []

    async def fail(_workspace):
        raise RuntimeError("worker unavailable")

    async def cold(_workspace):
        cold_calls.append(1)
        return SimpleNamespace(attack_success=False)

    async def stop(*, interrupt: bool = False):
        assert interrupt is True
        pass

    monkeypatch.setattr(runner, "_run_warm_once", fail)
    monkeypatch.setattr(runner, "_run_cold_once", cold)
    monkeypatch.setattr(runner, "_stop_warm_worker", stop)
    workspace = SimpleNamespace(output_root=tmp_path)
    await runner._run_once(workspace)
    await runner._run_once(workspace)
    assert len(cold_calls) == 2
    assert runner.runtime_metrics["warm_fallbacks"] == 1
    assert runner.runtime_metrics["effective_mode"] == "cold"


@pytest.mark.asyncio
async def test_warm_failure_after_victim_start_never_replays(monkeypatch, tmp_path: Path) -> None:
    runner = DtapAttemptRunner(attempt_runtime="warm")
    (tmp_path / ".m4-started").write_bytes(b"1")

    async def fail(_workspace):
        raise RuntimeError("worker exited")

    async def no_replay(_workspace):
        raise AssertionError("H must not be replayed")

    async def stop(*, interrupt: bool = False):
        assert interrupt is True
        pass

    monkeypatch.setattr(runner, "_run_warm_once", fail)
    monkeypatch.setattr(runner, "_run_cold_once", no_replay)
    monkeypatch.setattr(runner, "_stop_warm_worker", stop)
    result = await runner._run_once(SimpleNamespace(output_root=tmp_path))
    assert result.evaluation_started is True
    assert result.infrastructure_stage == "warm_runtime"


def test_warm_runtime_is_explicit_and_serial() -> None:
    runner = DtapAttemptRunner(attempt_runtime="warm")
    assert runner.attempt_runtime == "warm"
    with pytest.raises(ValueError, match="max_parallel=1"):
        DtapAttemptRunner(attempt_runtime="warm", max_parallel=2)
    with pytest.raises(ValueError, match="cold.*warm"):
        DtapAttemptRunner(attempt_runtime="unknown")


@pytest.mark.asyncio
async def test_matrix_passes_warm_mode_only_to_claude_engine(
    monkeypatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[2]
    template = root / "dt_arena/policy_eval/configs/finance-travel-indirect-openclaw.yaml"
    values = load_experiment_config(template)
    values.update(
        artifacts_root=tmp_path,
        dtap_root=root,
        python="python",
        policy_max_turns=64,
    )
    args = Namespace(**values)
    task = run_domain_matrix.MatrixTask(
        domain="finance",
        threat_model="indirect",
        risk_category="action_reversal",
        task_id="1",
        task_dir=root / "dataset/finance/malicious/indirect/action_reversal/1",
        benchmark_index=None,
        explicit=True,
    )
    commands: list[tuple[str, ...]] = []

    class FakeProcess:
        returncode = 1

        async def communicate(self):
            return b"", b""

    async def fake_process(*command, **_kwargs):
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(run_domain_matrix.asyncio, "create_subprocess_exec", fake_process)
    claude_result = await run_domain_matrix._run_case(args, task=task, slot=0)
    assert claude_result["attempt_runtime"] == "warm"
    assert commands[-1][commands[-1].index("--attempt-runtime") + 1] == "warm"

    args.policy_engine = run_domain_matrix.POLICY_ENGINE_DT_ARMS
    native_result = await run_domain_matrix._run_case(args, task=task, slot=0)
    assert native_result["attempt_runtime"] == "dt-arms-native"
    assert "--attempt-runtime" not in commands[-1]
