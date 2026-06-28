import argparse
import json
from datetime import datetime
from pathlib import Path

import requests


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/react/chat")
    parser.add_argument("--cases", default="eval_cases/react_cases.jsonl")
    parser.add_argument("--out-dir", default="eval_results")
    parser.add_argument("--session-prefix", default="eval-day13")
    parser.add_argument("--enable-reflection", action="store_true")
    args = parser.parse_args()

    cases = load_jsonl(Path(args.cases))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"react_eval_{timestamp}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for case in cases:
            case_id = case["case_id"]
            payload = {
                "session_id": f"{args.session_prefix}-{case_id}",
                "question": case["question"],
                "goal": "Build an AI Agent project in 10 days.",
                "level": "Beginner",
                "hours": 4,
                "enable_reflection": args.enable_reflection,
                "rag_top_k": 3,
                "rag_fetch_k": 8,
                "retrieval_mode": "lightweight",
                "max_memory_rounds": 10,
            }

            try:
                resp = requests.post(args.api_url, json=payload, timeout=120)
                data = resp.json()
                status_code = resp.status_code
            except Exception as exc:
                data = {"error": str(exc)}
                status_code = -1

            record = {
                "case_id": case_id,
                "question": case["question"],
                "expected_skills": case.get("expected_skills", []),
                "status_code": status_code,
                "matched_skills": data.get("matched_skills", []),
                "recommended_tools": data.get("recommended_tools", []),
                "routing_reason": data.get("routing_reason", ""),
                "final_answer": data.get("final_answer", ""),
                "draft_answer": data.get("draft_answer", ""),
                "used_reflection": data.get("used_reflection", False),
                "reflection": data.get("reflection", ""),
                "trace": data.get("trace", []),
                "redis_memory": data.get("redis_memory", {}),
                "long_term_memory_result": data.get("long_term_memory_result", {}),
                "error": data.get("detail") or data.get("error", ""),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{case_id}] status={status_code}, matched={record['matched_skills']}")

    print(f"\nEval results saved to: {out_path}")


if __name__ == "__main__":
    main()
