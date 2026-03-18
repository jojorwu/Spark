#!/bin/bash
set -e

echo "Running Unit Tests..."
cargo test --workspace

echo "Checking Compilation..."
cargo check --workspace

echo "All tests passed!"
