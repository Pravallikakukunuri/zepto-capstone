"""
llm_client.py
Optional MOCK_LLM=0 extension only. This file is NOT used, and not required,
when MOCK_LLM is left at its default (mock mode) -- the graded baseline never
imports or calls anything in this file.

If you want to try the optional real-LLM extension: set the environment
variable MOCK_LLM=0 and GROQ_API_KEY=<your free-tier key from console.groq.com>
before running the app. This requires `pip install groq` in addition to
requirements.txt.
"""

import os


def call_llm(prompt: str) -> str:
    """
    Calls Groq's free-tier API (or any other free-tier LLM API as a substitute).
    Only reached when MOCK_LLM=0 is explicitly set -- never called in the
    graded mock-mode baseline.
    """
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. This function is only used in the "
            "optional MOCK_LLM=0 extension; leave MOCK_LLM unset (or =1) "
            "to use the graded mock baseline instead, which never calls this."
        )

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content