"""Download pinned BorealTC sensor files; verify Git blob SHA-1 and save SHA-256."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen, Request
import time

ROOT = Path(__file__).resolve().parents[1]
REVISION = "0146dcd9fa08c34a9075a80448ac04c0a947b568"
BASE = f"https://raw.githubusercontent.com/norlab-ulaval/BorealTC/{REVISION}/"

def fetch(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "terrain-ml-student-project"}), timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)

def main():
    manifest_path = ROOT / "data/source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        tree = json.loads(fetch(f"https://api.github.com/repos/norlab-ulaval/BorealTC/git/trees/{REVISION}?recursive=1"))
        entries = [x for x in tree["tree"] if x["type"] == "blob" and x["path"].startswith("data/borealtc/") and "/MIXED/" not in x["path"]]
        manifest = {"repository": "https://github.com/norlab-ulaval/BorealTC", "revision": REVISION, "files": entries}
    def download(entry):
        relative = Path(entry["path"]).relative_to("data/borealtc")
        target = ROOT / "data/raw/borealtc" / relative
        content = target.read_bytes() if target.exists() else fetch(BASE + entry["path"])
        actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if actual != entry["sha"]:
            raise ValueError(f"Integrity check failed: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entry["sha256"] = hashlib.sha256(content).hexdigest()
        return str(relative)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for name in pool.map(download, manifest["files"]):
            print(name, flush=True)
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified {len(manifest['files'])} files from {REVISION}")

if __name__ == "__main__":
    main()
