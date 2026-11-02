"""Golden-set evaluation against the LIVE graph (needs keys + an ingested KB).

Checks that each question takes the expected path and that KB answers mention
an expected fact. Run after every prompt or model change:
    python -m scripts.eval
"""
import time

from app.core.logging import configure_logging
from app.rag.workflow import ask

CASES = [
    # (question, expected source_used, a fact the answer must contain or None)
    ("How many PTO days can I carry over into next year?", "private_kb", "5"),
    ("How long is parental leave for a non-birthing parent?", "private_kb", "12"),
    ("What is the hotel cap in San Francisco?", "private_kb", "350"),
    ("When is open enrollment?", "private_kb", "November"),
    ("What is the 2026 IRS 401(k) employee contribution limit?", "web", None),
    ("hi there!", "direct", None),
]


def main() -> None:
    passed = 0
    for question, expected, fact in CASES:
        t = time.perf_counter()
        r = ask(question)
        ms = int((time.perf_counter() - t) * 1000)
        ok_path = r["source_used"] == expected
        ok_fact = fact is None or fact.lower() in r["answer"].lower()
        ok = ok_path and ok_fact
        passed += ok
        path = " -> ".join(s["node"] for s in r["trace"])
        print(f"{'PASS' if ok else 'FAIL'} [{ms} ms] {question}\n     got={r['source_used']} want={expected} | {path}")
    print(f"\n{passed}/{len(CASES)} passed")


if __name__ == "__main__":
    configure_logging()
    main()
