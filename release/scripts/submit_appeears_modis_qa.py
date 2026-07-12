import os, json, time, zipfile, glob
from pathlib import Path
import pandas as pd
import requests

BASE = "https://appeears.earthdatacloud.nasa.gov/api"
OUT = Path("data/raw/appeears")
OUT.mkdir(parents=True, exist_ok=True)

user = os.environ["APPEEARS_USER"]
pw = os.environ["APPEEARS_PASS"]

points = pd.read_csv("data/raw/gee/stable_grassland_points.csv")
lat_col = "lat" if "lat" in points.columns else "latitude"
lon_col = "lon" if "lon" in points.columns else "longitude"
id_col = "point_id"

features = []
for _, r in points.iterrows():
    features.append({
        "type": "Feature",
        "properties": {"point_id": str(r[id_col])},
        "geometry": {"type": "Point", "coordinates": [float(r[lon_col]), float(r[lat_col])]}
    })

token = None
for attempt in range(1, 8):
    try:
        print(f"Login attempt {attempt}/7...")
        login = requests.post(f"{BASE}/login", auth=(user, pw), timeout=120)
        print("Login status:", login.status_code)
        if login.status_code == 504:
            print("AppEEARS gateway timeout. Waiting 60 seconds...")
            time.sleep(60)
            continue
        login.raise_for_status()
        token = login.json()["token"]
        break
    except requests.exceptions.RequestException as e:
        print("Login failed:", e)
        print("Waiting 60 seconds...")
        time.sleep(60)

if token is None:
    raise SystemExit("Could not log into AppEEARS after retries. Try again later or check Earthdata credentials.")
Path("appeears_token.txt").write_text(token)

headers = {"Authorization": f"Bearer {token}"}

task = {
    "task_type": "point",
    "task_name": "grassland_modis_gpp_et_qa_2001_2024",
    "params": {
        "dates": [{"startDate": "01-01-2001", "endDate": "12-31-2024"}],
        "layers": [
            {"product": "MOD17A2HGF.061", "layer": "Gpp_500m"},
            {"product": "MOD17A2HGF.061", "layer": "Psn_QC_500m"},
            {"product": "MOD16A2GF.061", "layer": "ET_500m"},
            {"product": "MOD16A2GF.061", "layer": "ET_QC_500m"}
        ],
        "geo": {"type": "FeatureCollection", "features": features}
    }
}

r = requests.post(f"{BASE}/task", headers=headers, json=task)
print(r.status_code, r.text[:1000])
r.raise_for_status()

task_id = r.headers.get("Location", "").split("/")[-1] or r.json().get("task_id") or r.json().get("taskId")
Path("data/raw/appeears/task_id.txt").write_text(str(task_id))
print("TASK_ID", task_id)

while True:
    s = requests.get(f"{BASE}/task/{task_id}", headers=headers)
    s.raise_for_status()
    js = s.json()
    status = js.get("status")
    print("STATUS", status)
    if status in ["done", "failed", "error"]:
        print(json.dumps(js, indent=2)[:4000])
        break
    time.sleep(60)

if status != "done":
    raise SystemExit("AppEEARS task did not finish successfully.")

bundle = requests.get(f"{BASE}/bundle/{task_id}", headers=headers)
bundle.raise_for_status()
files = bundle.json()["files"]

for f in files:
    fid = f["file_id"]
    name = f["file_name"]
    print("downloading", name)
    d = requests.get(f"{BASE}/bundle/{task_id}/{fid}", headers=headers, stream=True)
    d.raise_for_status()
    out = OUT / name
    with open(out, "wb") as w:
        for chunk in d.iter_content(chunk_size=1024*1024):
            if chunk:
                w.write(chunk)

for z in OUT.glob("*.zip"):
    with zipfile.ZipFile(z) as zz:
        zz.extractall(OUT)

print("DONE. Files:")
for p in OUT.glob("*"):
    print(p)
