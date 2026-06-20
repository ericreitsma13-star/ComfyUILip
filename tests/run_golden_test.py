#!/usr/bin/env python3
"""run_golden_test.py — Run LTX golden test via ComfyUI API."""
import json, sys, time, urllib.request

def run_test(workflow_path, server="http://127.0.0.1:8188", timeout=600):
    wf = json.load(open(workflow_path))
    payload = json.dumps({"prompt": wf, "client_id": "golden_test"}).encode()
    req = urllib.request.Request(f"{server}/prompt", data=payload, headers={"Content-Type": "application/json"})
    pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    print(f"Queued: {pid}")
    start = time.time()
    while time.time() - start < timeout:
        try:
            d = json.loads(urllib.request.urlopen(f"{server}/history/{pid}", timeout=10).read())
            if pid in d and d[pid].get("status",{}).get("completed"):
                outputs = d[pid].get("outputs",{})
                for nid, no in outputs.items():
                    for k, v in no.items():
                        if isinstance(v, list):
                            for item in v:
                                if isinstance(item, dict) and "filename" in item:
                                    print(f"Output: {item['filename']}")
                return True
        except: pass
        time.sleep(3)
    return False

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--workflow", default="tests/golden_test_flow_api.json")
    a.add_argument("--timeout", type=int, default=600)
    args = a.parse_args()
    ok = run_test(args.workflow, timeout=args.timeout)
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
