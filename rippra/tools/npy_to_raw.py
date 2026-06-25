#!/usr/bin/env python
"""
npy_to_raw.py — export .npy arrays to raw doubles for the C pipeline.

Usage: python tools/npy_to_raw.py [input_dir] [output_dir]

Defaults:  input_dir  = ../RAW_DATA/extracted_data
           output_dir = ./data_raw
"""
import numpy as np, sys, os

def main():
    indir  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', '..', 'RAW_DATA', 'extracted_data')
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', 'data_raw')
    os.makedirs(outdir, exist_ok=True)

    for name in ['sh_flat', 'img', 'sh_flat_bg']:
        npy = os.path.join(indir, f'{name}.npy')
        raw = os.path.join(outdir, f'{name}.raw')
        if not os.path.exists(npy):
            print(f'  SKIP {npy} (not found)')
            continue
        arr = np.load(npy).astype(np.float64, order='C')  # row-major C order
        arr.tofile(raw)
        print(f'  {npy} -> {raw}  (shape={arr.shape}, size={os.path.getsize(raw)} bytes)')

    # write a tiny metadata file so the C side can verify
    meta = os.path.join(outdir, 'meta.txt')
    with open(meta, 'w') as f:
        f.write('# shape and range of each exported frame\n')
        for name in ['sh_flat', 'img', 'sh_flat_bg']:
            npy = os.path.join(indir, f'{name}.npy')
            if os.path.exists(npy):
                arr = np.load(npy)
                f.write(f'{name}: shape={arr.shape} min={float(arr.min()):.6e} max={float(arr.max()):.6e}\n')

    print(f'Done. Metadata in {meta}')

if __name__ == '__main__':
    main()
