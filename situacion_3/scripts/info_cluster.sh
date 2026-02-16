#!/bin/bash
# ============================================================================
# info_cluster.sh — Información rápida del clúster (ambas máquinas)
# ============================================================================
# Uso: ./scripts/info_cluster.sh
# Muestra en un solo comando: CPU, RAM, disco, Python, Ray de ambas máquinas

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../docs/red/.config_red.sh"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║          INFO RÁPIDA DEL CLÚSTER — Situación 3             ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ──────────────── PORTÁTIL (MASTER) ────────────────
echo -e "${CYAN}┌─── PORTÁTIL (Master) ── $(hostname) ── $(hostname -I | awk '{print $1}') ───${NC}"

# CPU
CPU_MODEL=$(lscpu | grep "Model name" | sed 's/Model name:\s*//')
CPU_CORES=$(nproc)
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{printf "%.1f%%", $2+$4}')
echo -e "│ ${GREEN}CPU:${NC}  $CPU_MODEL ($CPU_CORES hilos) — Uso: $CPU_USAGE"

# RAM
RAM_TOTAL=$(free -h | awk '/Mem:/{print $2}')
RAM_USED=$(free -h | awk '/Mem:/{print $3}')
RAM_AVAIL=$(free -h | awk '/Mem:/{print $7}')
RAM_PCT=$(free | awk '/Mem:/{printf "%.0f", $3/$2*100}')
echo -e "│ ${GREEN}RAM:${NC}  $RAM_USED / $RAM_TOTAL (${RAM_PCT}%) — Disponible: $RAM_AVAIL"

# Disco
DISCO=$(df -h /mnt/datasets 2>/dev/null | awk 'NR==2{printf "%s / %s (%s)", $3, $2, $5}')
echo -e "│ ${GREEN}NFS:${NC}  $DISCO"

# Python + Ray
PY_VER=$(python3 --version 2>&1)
RAY_VER=$(python3 -c "import ray; print(ray.__version__)" 2>/dev/null || echo "no instalado")
echo -e "│ ${GREEN}Py:${NC}   $PY_VER — Ray $RAY_VER"
echo -e "${CYAN}└──────────────────────────────────────────────────────────────${NC}"

echo ""

# ──────────────── SERVIDOR (WORKER) ────────────────
echo -e "${YELLOW}┌─── SERVIDOR (Worker) ── $SSH_HOST ── $SSH_HOST_IP_LAN ───${NC}"

# Verificar conectividad
if ! ping -c 1 -W 2 $SSH_HOST_IP_LAN &>/dev/null; then
    echo -e "│ ${RED}✗ OFFLINE — No responde al ping${NC}"
    echo -e "${YELLOW}└──────────────────────────────────────────────────────────────${NC}"
    exit 1
fi

# Obtener info del servidor en una sola conexión SSH (pipe-delimited output)
REMOTE_RAW=$(sshpass -p "$PASSWORD" ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
    $SSH_USER@$SSH_HOST_IP_LAN 'export PATH=$HOME/.local/bin:$PATH
CPU_M=$(lscpu | grep "Model name" | sed "s/Model name:\s*//")
CPU_C=$(nproc)
CPU_U=$(top -bn1 | grep "Cpu(s)" | awk "{printf \"%.1f%%\", \$2+\$4}")
RAM_T=$(free -h | awk "/Mem:/{print \$2}")
RAM_U=$(free -h | awk "/Mem:/{print \$3}")
RAM_A=$(free -h | awk "/Mem:/{print \$7}")
RAM_P=$(free | awk "/Mem:/{printf \"%.0f\", \$3/\$2*100}")
DISCO=$(df -h /mnt/hdd 2>/dev/null | awk "NR==2{printf \"%s / %s (%s)\", \$3, \$2, \$5}")
PY=$(python3 --version 2>&1)
RV=$(python3 -c "import ray; print(ray.__version__)" 2>/dev/null || echo "no")
UP=$(uptime -p)
echo "${CPU_M}|${CPU_C}|${CPU_U}|${RAM_T}|${RAM_U}|${RAM_A}|${RAM_P}|${DISCO}|${PY}|${RV}|${UP}"
' 2>/dev/null)

if [ -n "$REMOTE_RAW" ]; then
    IFS='|' read -r R_CPU_MODEL R_CPU_CORES R_CPU_USAGE R_RAM_TOTAL R_RAM_USED R_RAM_AVAIL R_RAM_PCT R_DISCO R_PY_VER R_RAY_VER R_UPTIME <<< "$REMOTE_RAW"
    echo -e "│ ${GREEN}CPU:${NC}  $R_CPU_MODEL ($R_CPU_CORES hilos) — Uso: $R_CPU_USAGE"
    echo -e "│ ${GREEN}RAM:${NC}  $R_RAM_USED / $R_RAM_TOTAL (${R_RAM_PCT}%) — Disponible: $R_RAM_AVAIL"
    echo -e "│ ${GREEN}HDD:${NC}  $R_DISCO"
    echo -e "│ ${GREEN}Py:${NC}   $R_PY_VER — Ray $R_RAY_VER"
    echo -e "│ ${GREEN}Up:${NC}   $R_UPTIME"
else
    echo -e "│ ${RED}✗ No se pudo obtener información${NC}"
fi
echo -e "${YELLOW}└──────────────────────────────────────────────────────────────${NC}"

echo ""
REMOTE_CORES=${R_CPU_CORES:-0}
echo -e "${BOLD}Recursos combinados: $(($(nproc) + REMOTE_CORES)) hilos CPU${NC}"
echo ""
