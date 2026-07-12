from pathlib import Path
import requests

OUT = Path("/Users/me/Downloads/grassland_wue_nature_repo/data/external/_downloads/traits/liu_2021")
OUT.mkdir(parents=True, exist_ok=True)

ARTICLE_ID = "13350713"
VERSION = "2"
url = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/{VERSION}"

print("Fetching Figshare metadata:", url)
r = requests.get(url, timeout=60)
r.raise_for_status()
meta = r.json()

files = meta.get("files", [])
print("\nFiles found:", len(files))

with open(OUT / "figshare_files.txt", "w") as f:
    for i, item in enumerate(files):
        name = item.get("name")
        size = item.get("size")
        dl = item.get("download_url")
        line = f"{i}\t{name}\t{size}\t{dl}"
        print(line)
        f.write(line + "\n")

for item in files:
    name = item.get("name")
    dl = item.get("download_url")
    if not name or not dl:
        continue

    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 0:
        print("Already exists:", dest)
        continue

    print("\nDownloading:", name)
    with requests.get(dl, stream=True, timeout=120) as rr:
        rr.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in rr.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print("Wrote:", dest, dest.stat().st_size, "bytes")

print("\nDONE. Downloaded files are in:")
print(OUT)
