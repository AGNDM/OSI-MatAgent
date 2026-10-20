"""Read authorized model IDs; never print or persist the environment credential."""
import json
import os
import httpx


def main():
    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        raise SystemExit("SILICONFLOW_API_KEY is not exported in this process")
    with httpx.Client(timeout=20, trust_env=False) as client:
        response = client.get("https://api.siliconflow.cn/v1/models", headers={"Authorization": "Bearer " + key})
        print("Model catalog HTTP", response.status_code)
        response.raise_for_status()
        models = response.json()["data"]
    print(json.dumps([m["id"] for m in models if "deepseek" in m["id"].lower() and "flash" in m["id"].lower()], ensure_ascii=False))


if __name__ == "__main__":
    main()
