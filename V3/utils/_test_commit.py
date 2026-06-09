import requests, sys, base64, time, json
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import dotenv_values
from pathlib import Path

token = dotenv_values(Path(__file__).parent.parent.parent / "V3" / ".env").get("HF_TOKEN", "")
h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
repo = "PedroLandolt/adversarial-code-buggy"

# Test 1: tiny txt file
b64 = base64.b64encode(b"hello world").decode()
for attempt in range(3):
    try:
        r = requests.post(
            f"https://huggingface.co/api/datasets/{repo}/commit/main",
            headers=h,
            json={"summary": "test txt file", "operations": [{"op": "addOrUpdate", "path": "test.txt", "content": b64}]},
            timeout=30,
        )
        d = r.json()
        print(f"test.txt commit: status={r.status_code} success={d.get('success')} commit={d.get('commitUrl','')[-10:]}")
        break
    except Exception as e:
        print(f"Attempt {attempt+1}: {type(e).__name__}")
        time.sleep(2)

time.sleep(2)

# Check tree
for attempt in range(3):
    try:
        r = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main", headers=h, timeout=30)
        files = [f["path"] for f in r.json()]
        print(f"Tree: {files}")
        break
    except Exception as e:
        print(f"Tree {attempt+1}: {type(e).__name__}")
        time.sleep(2)
