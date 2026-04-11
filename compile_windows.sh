#!/bin/bash
set -e

echo "Building Spark Engine Editor for Windows..."
# Note: This requires the x86_64-pc-windows-gnu target and cross-compilation tools
rustup target add x86_64-pc-windows-gnu || true
cargo build --release -p spark-editor --target x86_64-pc-windows-gnu

echo "Preparing Build directory..."
rm -rf Build
mkdir -p Build

# Copy binary
cp target/x86_64-pc-windows-gnu/release/spark-editor.exe Build/

# Copy assets
cp -r assets Build/

echo "Build complete! You can find the Windows build in the 'Build' folder."
