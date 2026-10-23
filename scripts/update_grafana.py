import base64
import json
import subprocess
import urllib.request
from pathlib import Path

def main():
    dash_path = Path(__file__).resolve().parent.parent / "k8s" / "grafana_dashboard.json"
    with open(dash_path, "r", encoding="utf-8") as f:
        dash_data = json.load(f)

    payload = {
        "dashboard": dash_data,
        "overwrite": True,
        "message": "Enhanced live Edge AI Camouflage Surveillance telemetry"
    }

    auth_header = "Basic " + base64.b64encode(b"admin:kqC9ivmEIJgMiAwNgLU3jHTn3sReBiOdPQxGpz0L").decode("ascii")

    # Retrieve Grafana pod IP using kubectl
    try:
        cmd = ["kubectl", "get", "pod", "-n", "monitoring", "-l", "app.kubernetes.io/name=grafana", "-o", "jsonpath={.items[0].status.podIP}"]
        pod_ip = subprocess.check_output(cmd).decode().strip()
        print(f"Discovered Grafana Pod IP: {pod_ip}")
        target_url = f"http://{pod_ip}:3000/api/dashboards/db"
    except Exception as e:
        print(f"Could not get pod IP: {e}")
        target_url = "http://127.0.0.1:3000/api/dashboards/db"

    try:
        req = urllib.request.Request(
            target_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": auth_header,
                "Content-Type": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            body = res.read().decode("utf-8")
            print(f"[SUCCESS] Dashboard deployed to {target_url}!")
            print(body)
    except Exception as exc:
        print(f"[ERROR] Failed to post to {target_url}: {exc}")

if __name__ == "__main__":
    main()
