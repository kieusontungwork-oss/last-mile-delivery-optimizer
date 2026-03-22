#!/usr/bin/env bash
# Download all datasets for the project.
set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR/nyc_tlc" "$RAW_DIR/amazon" "$RAW_DIR/solomon"

# --- NYC TLC Yellow Taxi (Jan, Jun, Oct 2023) ---
echo "=== Downloading NYC TLC Yellow Taxi data ==="
TLC_BASE="https://d37ci6vzurychx.cloudfront.net/trip-data"
for month in 01 06 10; do
    FILE="yellow_tripdata_2023-${month}.parquet"
    if [ ! -f "$RAW_DIR/nyc_tlc/$FILE" ]; then
        echo "Downloading $FILE..."
        curl -L -C - -o "$RAW_DIR/nyc_tlc/$FILE" "${TLC_BASE}/${FILE}"
    else
        echo "$FILE already exists, skipping."
    fi
done

# --- NYC Taxi Zone Shapefile ---
ZONES_URL="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
if [ ! -f "$RAW_DIR/nyc_tlc/taxi_zones.zip" ]; then
    echo "Downloading taxi zone shapefile..."
    curl -L -C - -o "$RAW_DIR/nyc_tlc/taxi_zones.zip" "$ZONES_URL"
    unzip -o "$RAW_DIR/nyc_tlc/taxi_zones.zip" -d "$RAW_DIR/nyc_tlc/taxi_zones/"
fi

# --- Amazon Last Mile ---
echo ""
echo "=== Downloading Amazon Last Mile dataset ==="
if command -v aws &> /dev/null; then
    aws s3 sync --no-sign-request \
        s3://amazon-last-mile-challenges/almrrc2021/ \
        "$RAW_DIR/amazon/"
else
    echo "AWS CLI not installed. Install it with: brew install awscli"
    echo "Then re-run this script."
fi

# --- Solomon VRPTW Benchmarks ---
echo ""
echo "=== Downloading Solomon VRPTW benchmarks ==="
cd "$RAW_DIR/solomon"
python3 -c "
import vrplib
import os
instances = ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']
for name in instances:
    fname = f'{name}.txt'
    sol_fname = f'{name}.sol'
    if not os.path.exists(fname):
        print(f'Downloading {name}...')
        vrplib.download_instance(f'{name}', fname)
    if not os.path.exists(sol_fname):
        vrplib.download_solution(f'{name}', sol_fname)
print('Solomon instances ready.')
"
cd -

echo ""
echo "=== Download complete ==="
echo "NYC TLC:  $RAW_DIR/nyc_tlc/"
echo "Amazon:   $RAW_DIR/amazon/"
echo "Solomon:  $RAW_DIR/solomon/"
