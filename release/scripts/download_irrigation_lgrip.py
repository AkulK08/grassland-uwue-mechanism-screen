from pathlib import Path
import earthaccess

Path("data/external/irrigation").mkdir(parents=True, exist_ok=True)

earthaccess.login(strategy="interactive", persist=True)

results = earthaccess.search_data(
    short_name="LGRIP30_L2_IRRI",
    version="002",
    temporal=("2020-01-01", "2020-12-31"),
    count=50
)

print("FOUND", len(results), "LGRIP files")
earthaccess.download(results, "data/external/irrigation")
print("DONE")
