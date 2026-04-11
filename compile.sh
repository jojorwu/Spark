#!/bin/bash

function print_usage() {
    echo "Usage: ./compile.sh [linux|windows|all]"
    echo "  linux   - Build for Linux (output in Build_Linux/)"
    echo "  windows - Build for Windows (output in Build/)"
    echo "  all     - Build for both platforms"
}

if [ -z "$1" ]; then
    print_usage
    exit 1
fi

case "$1" in
    linux)
        ./compile_linux.sh
        ;;
    windows)
        ./compile_windows.sh
        ;;
    all)
        ./compile_linux.sh
        ./compile_windows.sh
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
