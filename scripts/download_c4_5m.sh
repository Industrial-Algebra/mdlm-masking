#!/bin/bash
# Download 5M C4 en examples for larger training runs
# Run on hyperion-proto before bed — network-only, no GPU needed
# Output: examples/datasets/c4-en-5m.jsonl (~10-12 GB)

set -euo pipefail
cd /home/lucien/working/industrial-algebra/Quantizon

echo "Downloading 5M C4 en examples..."
echo "This will take 30-60 minutes and use ~10 GB disk."
echo "Run: nohup bash scripts/download_c4_5m.sh > logs/c4-download.log 2>&1 &"

/home/lucien/miniconda3/bin/python3 -c "
import json, datasets
from pathlib import Path

ds = datasets.load_dataset('allenai/c4', 'en', split='train', streaming=True, trust_remote_code=True)
ds = ds.take(5_000_000)

output = Path('examples/datasets/c4-en-5m.jsonl')
output.parent.mkdir(parents=True, exist_ok=True)

count = 0
with open(output, 'w', encoding='utf-8') as f:
    for row in ds:
        text = row['text'].strip()
        if not text or len(text) < 50:
            continue
        f.write(json.dumps({'text': text}) + '\n')
        count += 1
        if count >= 5_000_000:
            break
        if count % 50000 == 0:
            print(f'  downloaded {count:,}...')

size_gb = output.stat().st_size / 1024**3
print(f'Done: {count:,} examples, {size_gb:.1f} GB')
"
