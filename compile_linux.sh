#!/bin/bash
set -e

echo "Building Spark Engine Editor for Linux..."
cargo build --release -p spark-editor

echo "Preparing Build_Linux directory..."
rm -rf Build_Linux
mkdir -p Build_Linux

# Copy binary
cp target/release/spark-editor Build_Linux/

# Copy assets
cp -r assets Build_Linux/

echo "Build complete! You can find the Linux build in the 'Build_Linux' folder."
chmod +x Build_Linux/spark-editor
