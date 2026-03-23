from test_flows import SCENARIOS, run_scenario
import json

print("Running test 1 with Gemini...")
res = run_scenario(SCENARIOS[0])
print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
