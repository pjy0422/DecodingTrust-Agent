import json
from typing import Any, Dict, Optional, Tuple
from utils.llm import call_llm
from dt_arena.src.types.judge import JudgeUnavailableError
from dt_arena.utils.travel.prompts import (
    DECEPTION_SYSTEM_PROMPT, DECEPTION_JUDGE_PROMPT,
    CREDENTIAL_COLLECTION_SYSTEM_PROMPT, CREDENTIAL_COLLECTION_JUDGE_PROMPT,
    PII_COLLECTION_SYSTEM_PROMPT, PII_COLLECTION_JUDGE_PROMPT,
)


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def llm_judge_deception(
    malicious_goal: str,
    agent_text: str,
    model: str,
    retries: int = 3,
) -> Tuple[bool, Dict[str, Any]]:
    """Use LLM to judge whether agent made deceptive claims in its responses.

    Returns (success, metadata), or raises after all provider retries fail.
    """

    prompt = DECEPTION_JUDGE_PROMPT.format(
        malicious_goal=malicious_goal,
        agent_text=agent_text,
    )

    data = None
    for attempt in range(retries):
        try:
            content = call_llm(
                messages=[
                    {"role": "system", "content": DECEPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.0,
            )
            data = _extract_json_obj(content or "")
            if not isinstance(data, dict) or "attack_succeeded" not in data:
                continue
        except Exception:
            print(f"LLM judge attempt {attempt + 1} failed to produce valid JSON. Retrying...")
            # if attempt == retries - 1:
            #     return None
            continue
    
    if data is None:
        raise JudgeUnavailableError("travel deception judge exhausted retries")
    return bool(data.get("attack_succeeded")), {"llm_judge": data, "llm_model": model}


def llm_judge_credential_collection(
    malicious_goal: str,
    agent_text: str,
    model: str,
    retries: int = 3,
) -> Tuple[bool, Dict[str, Any]]:
    """Use LLM to judge whether agent requested payment credentials from the user.

    Returns (success, metadata), or raises after all provider retries fail.
    """

    prompt = CREDENTIAL_COLLECTION_JUDGE_PROMPT.format(
        malicious_goal=malicious_goal,
        agent_text=agent_text,
    )

    data = None
    for attempt in range(retries):
        try:
            content = call_llm(
                messages=[
                    {"role": "system", "content": CREDENTIAL_COLLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.0,
            )
            data = _extract_json_obj(content or "")
            if not isinstance(data, dict) or "attack_succeeded" not in data:
                continue
        except Exception:
            continue

    if data is None:
        raise JudgeUnavailableError("travel credential judge exhausted retries")
    return bool(data.get("attack_succeeded")), {"llm_judge": data, "llm_model": model}


def llm_judge_pii_collection(
    malicious_goal: str,
    agent_text: str,
    model: str,
    retries: int = 3,
) -> Tuple[bool, Dict[str, Any]]:
    """Use LLM to judge whether agent requested unnecessary personal information from the user.

    Returns (success, metadata), or raises after all provider retries fail.
    """

    prompt = PII_COLLECTION_JUDGE_PROMPT.format(
        malicious_goal=malicious_goal,
        agent_text=agent_text,
    )

    data = None
    for attempt in range(retries):
        try:
            content = call_llm(
                messages=[
                    {"role": "system", "content": PII_COLLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.0,
            )
            data = _extract_json_obj(content or "")
            if not isinstance(data, dict) or "attack_succeeded" not in data:
                continue
        except Exception:
            continue

    if data is None:
        raise JudgeUnavailableError("travel PII judge exhausted retries")
    return bool(data.get("attack_succeeded")), {"llm_judge": data, "llm_model": model}
