#!/usr/bin/env bash
# Read-only snapshot; RSS values are KiB. No apps/services are stopped.
set -u
printf 'RAM and swap:\n'
free -h
printf '\nLargest processes (RSS in KiB):\n'
ps -eo pid,comm,rss --sort=-rss | head -n 16
printf '\nLoaded Ollama models:\n'
if command -v ollama >/dev/null 2>&1; then
    ollama ps || true
else
    echo 'ollama not found in PATH.'
fi
printf '\nSwap devices:\n'
if command -v swapon >/dev/null 2>&1; then
    swapon --show || true
fi
