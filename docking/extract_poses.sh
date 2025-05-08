#!/bin/bash

# This script processes .db files in the results directory,
# converts them to .sdf format with docking scores,
# combines all .sdf files into one, and compresses the result

# Exit on error
set -e

# Create a secure temporary directory that will be automatically cleaned up on exit
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Check if results directory exists
if [ ! -d "results" ]; then
  echo "Error: 'results' directory not found"
  exit 1
fi

# Process each .db file in the results directory
echo "Processing .db files in results directory..."
CURRENT_DIR=$(pwd)
cd results

for db_file in *.db; do
  # Skip if no .db files found
  [ -e "$db_file" ] || { echo "No .db files found in results directory"; exit 1; }

  # Skip experimental.db
  if [ "$db_file" = "experimental.db" ]; then
    echo "Skipping experimental.db"
    continue
  fi
  
  # Get base name without extension
  base_name="${db_file%.db}"
  output_file="${TMP_DIR}/${base_name}.sdf"
  
  # Requires easy dock installed
  echo "Converting $db_file to $output_file"
  get_sdf_from_dock_db -i "$db_file" -o "$output_file" --fields docking_score
done

cd "$CURRENT_DIR"

# Combine all .sdf files into one
COMBINED_FILE="${CURRENT_DIR}/poses.sdf"
echo "Combining all .sdf files into $COMBINED_FILE"
cat "${TMP_DIR}"/*.sdf > "$COMBINED_FILE"

# Check if combined file was created successfully
if [ ! -s "$COMBINED_FILE" ]; then
  echo "Error: Failed to create combined SDF file or file is empty"
  exit 1
fi

# Compress the combined file
COMPRESSED_FILE="${COMBINED_FILE}.gz"
echo "Compressing to $COMPRESSED_FILE"
gzip -c "$COMBINED_FILE" > "$COMPRESSED_FILE"

# Check if compressed file was created successfully
if [ ! -s "$COMPRESSED_FILE" ]; then
  echo "Error: Failed to create compressed file or file is empty"
  exit 1
fi

echo "Done! Combined results are in $COMPRESSED_FILE in the current directory"
# Cleanup is handled automatically by the trap command