#!/usr/bin/env bash
# Start the MIRAI kiosk automatically when the desktop logs in.
#   bash scripts/install_autostart.sh            # turn auto-start on
#   bash scripts/install_autostart.sh --remove   # turn it off
# Output goes to .mirai/kiosk.log. Pair with automatic login (Settings > Users) so no keyboard is needed.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
entry="$HOME/.config/autostart/mirai-kiosk.desktop"

if [[ "${1:-}" == --remove ]]; then
    rm -f "$entry"
    echo "Auto-start removed ($entry)."
    exit
fi

mkdir -p "$(dirname "$entry")" "$project_dir/.mirai"
cat > "$entry" <<EOF
[Desktop Entry]
Type=Application
Name=MIRAI Kiosk
Exec=bash -c 'exec bash "$project_dir/scripts/start_kiosk.sh" >> "$project_dir/.mirai/kiosk.log" 2>&1'
X-GNOME-Autostart-enabled=true
EOF
echo "Auto-start installed ($entry). MIRAI opens at the next login; log: $project_dir/.mirai/kiosk.log"
