import requests, sys, time
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import dotenv_values
from pathlib import Path

token = dotenv_values(Path(__file__).parent.parent.parent / "V3" / ".env").get("HF_TOKEN", "")
h = {"Authorization": f"Bearer {token}"}
repo = "PedroLandolt/adversarial-code-buggy"

for attempt in range(3):
    try:
        # Repo info
        r = requests.get(f"https://huggingface.co/api/datasets/{repo}", headers=h, timeout=30)
        info = r.json()
        print(f"Repo: {info.get('id')} private={info.get('private')} disabled={info.get('disabled')}")
        print(f"SHA: {info.get('sha')}")
        break
    except Exception as e:
        print(f"Attempt {attempt+1}: {type(e).__name__}")
        time.sleep(2)

for attempt in range(3):
    try:
        # Raw tree response
        r = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main", headers=h, timeout=30)
        raw = r.text
        print(f"Raw tree response ({len(raw)} chars): {raw[:500]}")
        break
    except Exception as e:
        print(f"Tree attempt {attempt+1}: {type(e).__name__}")
        time.sleep(2)
