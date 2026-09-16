from __future__ import annotations

import argparse
import json

from .scenarios import (
    attention_experiment,
    echo_experiment,
    fork_experiment,
    haunting_experiment,
    history_experiment,
    morrow_experiment,
    run_all,
    speech_experiment,
)


SCENARIOS = {
    "echo": echo_experiment,
    "history": history_experiment,
    "haunting": haunting_experiment,
    "attention": attention_experiment,
    "fork": fork_experiment,
    "morrow": morrow_experiment,
    "speech": speech_experiment,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MADMAN digital-metaphysics experiments")
    parser.add_argument("scenario", choices=["all", *SCENARIOS], nargs="?", default="all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    results = run_all(args.seed) if args.scenario == "all" else (SCENARIOS[args.scenario](args.seed),)
    if args.as_json:
        print(json.dumps([{"name": r.name, "metrics": r.metrics} for r in results], indent=2, sort_keys=True))
        return
    for result in results:
        print(f"[{result.name}]")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
