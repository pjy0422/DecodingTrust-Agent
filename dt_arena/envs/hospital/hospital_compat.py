"""OpenAI-compatible entrypoint for the published DTAP Hospital image.

The upstream image dispatches providers from a small model-name allowlist.  A
release run may intentionally use another model through an OpenAI-compatible
endpoint, so opt in explicitly without changing the image or exposing keys.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time

from openai import OpenAI
from waitress import serve

import main as hospital


def _query_openai_compatible(
    model_str,
    prompt,
    system_prompt,
    tries=3,
    timeout=20.0,
    max_tokens=200,
):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=max(float(timeout), 60.0),
    )
    attempts = max(1, int(tries))
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=str(model_str),
                messages=[
                    {"role": "system", "content": str(system_prompt)},
                    {"role": "user", "content": str(prompt)},
                ],
                max_tokens=max_tokens,
            )
            break
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = (
                status == 429
                or isinstance(status, int) and status >= 500
                or type(exc).__name__ in {
                    "APIConnectionError", "APITimeoutError", "RateLimitError",
                }
            )
            if not retryable or attempt + 1 >= attempts:
                raise
            time.sleep(min(2 ** attempt, 8))
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, ensure_ascii=False)


if os.getenv("DTAP_HOSPITAL_OPENAI_COMPAT") == "1":
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        raise RuntimeError("Hospital OpenAI-compatible provider is not configured")
    hospital.query_model = _query_openai_compatible


if __name__ == "__main__":
    cores = multiprocessing.cpu_count()
    serve(
        hospital.app,
        host=os.getenv("HOSPITAL_HOST", "0.0.0.0"),
        port=int(os.getenv("HOSPITAL_PORT", "12001")),
        threads=max(4, cores // 2),
        expose_tracebacks=True,
        channel_timeout=60,
        cleanup_interval=10,
    )
