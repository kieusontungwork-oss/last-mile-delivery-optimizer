#!/usr/bin/env bash
# Setup OSRM with New York State road network data.
# Prerequisites: Docker must be running.
set -euo pipefail

OSRM_DIR="data/external/osrm"
PBF_URL="https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf"
PBF_FILE="new-york-latest.osm.pbf"
OSRM_IMAGE="ghcr.io/project-osrm/osrm-backend"

mkdir -p "$OSRM_DIR"
cd "$OSRM_DIR"

# Step 1: Download PBF if not present
if [ ! -f "$PBF_FILE" ]; then
    echo "Downloading NY State OSM data (~463 MB)..."
    curl -L -C - -o "$PBF_FILE" "$PBF_URL"
else
    echo "PBF file already exists, skipping download."
fi

OSRM_BASE="${PBF_FILE%.osm.pbf}"

# Step 2: Extract
if [ ! -f "${OSRM_BASE}.osrm" ]; then
    echo "Extracting road network..."
    docker run -t -v "${PWD}:/data" "$OSRM_IMAGE" \
        osrm-extract -p /opt/car.lua "/data/$PBF_FILE"
else
    echo "Extract already done, skipping."
fi

# Step 3: Partition
if [ ! -f "${OSRM_BASE}.osrm.partition" ]; then
    echo "Partitioning..."
    docker run -t -v "${PWD}:/data" "$OSRM_IMAGE" \
        osrm-partition "/data/${OSRM_BASE}.osrm"
else
    echo "Partition already done, skipping."
fi

# Step 4: Customize
if [ ! -f "${OSRM_BASE}.osrm.cell_metrics" ]; then
    echo "Customizing..."
    docker run -t -v "${PWD}:/data" "$OSRM_IMAGE" \
        osrm-customize "/data/${OSRM_BASE}.osrm"
else
    echo "Customize already done, skipping."
fi

echo ""
echo "OSRM data ready at: $OSRM_DIR"
echo "Start the server with:"
echo "  docker run -d -p 5000:5000 -v \"\$(pwd)/$OSRM_DIR:/data\" --name osrm \\"
echo "    $OSRM_IMAGE osrm-routed --algorithm mld --max-table-size 10000 /data/${OSRM_BASE}.osrm"
