"""Re-creates data/prostate_cancer.csv from its public source (see data/README.md).

    python data/fetch_prostate.py

The source repository has no CSV file; the data are
embedded in the output of a notebook cell that uploaded the file. They are read here with
ast.literal_eval (a literal parser, no code is executed).
"""
import ast
import hashlib
import json
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/SAZZZO99/PROSTATE-CANCER-DETECTION/master/"
       "PROSTATE_CANCER_KAGGLE.ipynb")
SHA256 = "1db89cad00041ab8840d6913f0075c1e2f6cf2f4ae4060c14f59d689686e4476"
OUT = Path(__file__).resolve().parent / "prostate_cancer.csv"


def main():
    with urllib.request.urlopen(URL, timeout=30) as r:
        nb = json.load(r)
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = "".join(out.get("data", {}).get("text/plain", []))
            if text.startswith("{'PTRAIN.csv'"):
                data = ast.literal_eval(text)["PTRAIN.csv"]
                digest = hashlib.sha256(data).hexdigest()
                if digest != SHA256:
                    raise RuntimeError(f"checksum mismatch: {digest}")
                OUT.write_bytes(data)
                print(f"wrote {OUT} ({len(data)} bytes, sha256 ok)")
                return
    raise RuntimeError("uploaded CSV not found in the notebook outputs")


if __name__ == "__main__":
    main()
