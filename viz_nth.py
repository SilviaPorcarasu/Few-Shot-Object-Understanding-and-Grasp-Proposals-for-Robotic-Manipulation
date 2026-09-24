#!/usr/bin/env python3
"""Visualize the Nth ranked grasp by swapping it into best_candidate slot."""
import json, os, sys, tempfile
from pathlib import Path

scores_path = os.environ.get("GRASP_SCORES_PATH", "../grasp_scores_flexible.json")
rank = int(os.environ.get("GRASP_RANK", "0"))   # 0 = best, 1 = second, ...

data = json.loads(Path(scores_path).read_text())
candidate = data["all_candidates"][rank]
data["best_candidate"] = candidate

tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
json.dump(data, tmp)
tmp.close()
os.environ["GRASP_SCORES_PATH"] = tmp.name

import visualize_grasp
visualize_grasp.main()
Path(tmp.name).unlink(missing_ok=True)
