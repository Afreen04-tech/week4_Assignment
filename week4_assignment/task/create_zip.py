
import zipfile
import os

ROOT = r"E:\task"
EXCLUDE_NAMES = {".env", ".git", "__pycache__", "create_zip.py", "Week4_LLM_Assignment.zip"}
EXCLUDE_EXT = {".pyc"}

out_path = os.path.join(ROOT, "Week4_LLM_Assignment.zip")

with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Skip excluded directories in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_NAMES]
        for filename in filenames:
            if filename in EXCLUDE_NAMES:
                continue
            if os.path.splitext(filename)[1] in EXCLUDE_EXT:
                continue
            full_path = os.path.join(dirpath, filename)
            arcname = os.path.relpath(full_path, ROOT)
            zf.write(full_path, arcname)
            print(f"  Added: {arcname}")

size_mb = os.path.getsize(out_path) / (1024 * 1024)
print(f"\nZIP created: {out_path}  ({size_mb:.2f} MB)")

