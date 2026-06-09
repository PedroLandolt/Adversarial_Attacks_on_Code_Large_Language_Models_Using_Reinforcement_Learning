"""
Combine MBPP + HumanEval and upload as adversarial-code-buggy.jsonl.
Uses requests (urllib3) to avoid the httpx WinError 10054 TLS issue.
Payload format: application/x-ndjson (one JSON object per line).
"""
import base64
import json
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).parent.parent.parent

try:
    from dotenv import dotenv_values
    token = dotenv_values(_ROOT / "V3" / ".env").get("HF_TOKEN", "")
except Exception:
    token = ""

if not token:
    print("ERROR: HF_TOKEN not found in V3/.env")
    sys.exit(1)

repo_id = "PedroLandolt/adversarial-code-buggy"
branch = "main"
dest_path = "adversarial-code-buggy.jsonl"

# --- Build combined file ---
records = []
for fname in ["datasets/mbpp_pregenerated.jsonl", "datasets/humaneval_pregenerated.jsonl"]:
    p = _ROOT / fname
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

content_bytes = b"\n".join(json.dumps(r).encode() for r in records)
print(f"Combined: {len(records)} records, {len(content_bytes):,} bytes")

b64 = base64.b64encode(content_bytes).decode("ascii")

# --- Build ndjson payload (exact format used by huggingface_hub library) ---
ndjson_lines = [
    json.dumps({"key": "header", "value": {
        "summary": f"Add {dest_path} ({len(records)} records: 790 MBPP + 150 HumanEval)",
        "description": "",
    }}),
    json.dumps({"key": "file", "value": {
        "content": b64,
        "path": dest_path,
        "encoding": "base64",
    }}),
]
ndjson_payload = "\n".join(ndjson_lines).encode("utf-8")
print(f"ndjson payload size: {len(ndjson_payload):,} bytes")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/x-ndjson",
}

# --- Commit with retries ---
print("Committing (up to 5 attempts)...")
for attempt in range(5):
    try:
        resp = requests.post(
            f"https://huggingface.co/api/datasets/{repo_id}/commit/{branch}",
            headers=headers,
            data=ndjson_payload,
            timeout=120,
        )
        print(f"  Attempt {attempt+1}: status {resp.status_code}")
        try:
            d = resp.json()
            if d.get("success"):
                print(f"SUCCESS — {d.get('commitUrl')}")
                break
            else:
                print(f"  API error: {d}")
                break
        except Exception:
            print(f"  Raw response: {resp.text[:300]}")
            break
    except Exception as e:
        print(f"  Attempt {attempt+1} network error: {type(e).__name__}")
        if attempt < 4:
            time.sleep(3)
else:
    print("All attempts failed.")

# --- Verify ---
print("\nVerifying tree...")
time.sleep(2)
for attempt in range(3):
    try:
        r = requests.get(
            f"https://huggingface.co/api/datasets/{repo_id}/tree/{branch}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if r.status_code == 200:
            files = r.json()
            for f in files:
                print(f"  {f['path']} ({f['size']:,} bytes)")
        break
    except Exception as e:
        print(f"  Tree attempt {attempt+1}: {type(e).__name__}")
        time.sleep(2)
