"""
downcast_npz.py
================
Usage:
    python downcast_npz.py path/to/DRegular.npz
    python downcast_npz.py path/to/complete_bipartite.npz
    python downcast_npz.py path/to/Gilbert.npz
    # etc. for complete.npz, linear.npz, circular.npz
"""
 
import sys
from pathlib import Path
 
import numpy as np
 
 
def downcast_array(arr, target_dtype=np.int16):
    info = np.iinfo(target_dtype)
    if arr.size and (arr.min() < info.min or arr.max() > info.max):
        raise ValueError(
            f"Values out of range for {target_dtype}: "
            f"min={arr.min()}, max={arr.max()}, allowed=[{info.min}, {info.max}]"
        )
    return arr.astype(target_dtype)
 
 
def downcast_npz(path, target_dtype=np.uint16):
    path = Path(path)
    data = np.load(path)
 
    flat = {}
    for name in data.files:
        arr = data[name]
        if np.issubdtype(arr.dtype, np.integer):
            flat[name] = downcast_array(arr, target_dtype)
        else:
            flat[name] = arr
 
    out_path = path.with_name(f"{path.stem}_{np.dtype(target_dtype).name}.npz")
    np.savez_compressed(out_path, **flat)
 
    orig_size = path.stat().st_size
    new_size = out_path.stat().st_size
    print(f"{path.name}: {orig_size/1e6:.1f} MB -> {out_path.name}: {new_size/1e6:.1f} MB "
          f"({orig_size/new_size:.2f}x smaller)")
 
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python downcast_npz.py file1.npz [file2.npz ...]")
        sys.exit(1)
 
    for f in sys.argv[1:]:
        try:
            downcast_npz(f)
        except ValueError as e:
            print(f"SKIPPED {f}: {e}")
