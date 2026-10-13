#!/usr/bin/env bash
# One-time memory tuning for an 8 GB Jetson, after Ollama was OOM-killed while MIRAI ran.
#   bash scripts/tune_jetson.sh           # show the plan, ask, then apply (needs sudo)
#   bash scripts/tune_jetson.sh --check   # only report the current state
# Safe to run again: steps already done are skipped. It does not change the power mode.
set -euo pipefail
swap_size_gb=8
swapfile=''
check_only=false
[[ "${1:-}" == --check ]] && check_only=true

disk_swap_kb() {  # zram lives in RAM, so only count swap on real disks/files
    awk 'NR > 1 && $1 !~ /zram/ {sum += $3} END {print sum + 0}' /proc/swaps
}

# Prefer the NVMe drive (Ollama's models already live on /mnt/nvme); never swap to an SD card.
for candidate in /mnt/nvme /; do
    if mountpoint -q "$candidate" || [[ "$candidate" == / ]]; then
        fstype="$(findmnt -no FSTYPE --target "$candidate")"
        source="$(findmnt -no SOURCE --target "$candidate")"
        if [[ "$fstype" =~ ^(ext4|xfs)$ && "$source" == *nvme* ]]; then
            swapfile="${candidate%/}/mirai.swap"
            break
        fi
    fi
done

echo '== Current state'
free -h
echo "Disk swap: $(( $(disk_swap_kb) / 1024 / 1024 )) GB (zram not counted)"
override=/etc/systemd/system/ollama.service.d/mirai.conf
[[ -f "$override" ]] && echo "Ollama memory settings: present ($override)" || echo 'Ollama memory settings: not set'
if command -v nvpmodel >/dev/null 2>&1; then
    echo "Power mode: $(sudo -n nvpmodel -q 2>/dev/null | head -n 1 || echo 'run: sudo nvpmodel -q')"
fi
$check_only && exit 0

echo
echo '== Plan'
if (( $(disk_swap_kb) >= swap_size_gb * 1024 * 1024 - 1024 )); then
    echo "- Swap: already ${swap_size_gb} GB or more, skipping."
    swapfile=''
elif [[ -z "$swapfile" ]]; then
    echo '- Swap: no ext4/xfs NVMe filesystem found; skipping (swap on an SD card wears it out).'
else
    echo "- Swap: create a ${swap_size_gb} GB swap file at $swapfile and enable it at boot."
fi
echo '- Ollama: flash attention + 8-bit KV cache (about half the context memory), one model at a time.'
read -r -p 'Apply? [y/N] ' answer
[[ "$answer" =~ ^[Yy]$ ]] || { echo 'Nothing changed.'; exit 0; }

if [[ -n "$swapfile" ]]; then
    if [[ ! -f "$swapfile" ]]; then
        sudo fallocate -l "${swap_size_gb}G" "$swapfile" \
            || sudo dd if=/dev/zero of="$swapfile" bs=1M count=$(( swap_size_gb * 1024 )) status=progress
        sudo chmod 600 "$swapfile"
        sudo mkswap "$swapfile"
    fi
    grep -q "^$swapfile " /proc/swaps || sudo swapon "$swapfile"
    grep -q "^$swapfile " /etc/fstab || echo "$swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
    echo "Swap enabled: $swapfile"
fi

sudo mkdir -p "$(dirname "$override")"
sudo tee "$override" >/dev/null <<'EOF'
# Written by MIRAI scripts/tune_jetson.sh. Delete this file and restart ollama to undo.
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
echo 'Ollama restarted with the memory settings.'
echo
free -h
echo 'Done. Restart the kiosk: bash scripts/start_kiosk.sh'
