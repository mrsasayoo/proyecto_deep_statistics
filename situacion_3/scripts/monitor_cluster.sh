#!/bin/bash
# ============================================================================
# monitor_cluster.sh — Monitor de recursos en tiempo real (ambas máquinas)
# ============================================================================
# Uso: ./scripts/monitor_cluster.sh [intervalo_segundos]
# Default: actualiza cada 3 segundos
# Ctrl+C para detener

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../docs/red/.config_red.sh"

INTERVALO=${1:-3}

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'

# Función para barra visual de porcentaje
barra() {
    local pct=$1
    local width=30
    local filled=$((pct * width / 100))
    local empty=$((width - filled))
    local color=$GREEN
    [[ $pct -ge 70 ]] && color=$YELLOW
    [[ $pct -ge 90 ]] && color=$RED
    printf "${color}["
    printf '%0.s█' $(seq 1 $filled 2>/dev/null) || true
    printf '%0.s░' $(seq 1 $empty 2>/dev/null) || true
    printf "] %3d%%${NC}" "$pct"
}

# Verificar servidor una vez
SERVIDOR_ONLINE=false
if ping -c 1 -W 2 $SSH_HOST_IP_LAN &>/dev/null; then
    SERVIDOR_ONLINE=true
fi

echo -e "${BOLD}Monitor del Clúster — Ctrl+C para salir — Intervalo: ${INTERVALO}s${NC}"
echo ""

while true; do
    clear
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║  MONITOR DE RECURSOS — $TIMESTAMP            ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # ── LOCAL (Master) ──
    LOCAL_CPU_PCT=$(top -bn1 | grep "Cpu(s)" | awk '{printf "%.0f", $2+$4}')
    LOCAL_RAM_PCT=$(free | awk '/Mem:/{printf "%.0f", $3/$2*100}')
    LOCAL_RAM_USED=$(free -h | awk '/Mem:/{print $3}')
    LOCAL_RAM_TOTAL=$(free -h | awk '/Mem:/{print $2}')
    LOCAL_RAM_AVAIL=$(free -h | awk '/Mem:/{print $7}')
    LOCAL_LOAD=$(cat /proc/loadavg | awk '{print $1, $2, $3}')
    LOCAL_PROCS=$(ps aux | wc -l)
    LOCAL_PY_PROCS=$(ps aux | grep -c "[p]ython3" || echo "0")

    echo -e "  ${BOLD}PORTÁTIL (Master)${NC} — $(hostname) — $(hostname -I | awk '{print $1}')"
    echo -e "  CPU:  $(barra $LOCAL_CPU_PCT)  Load: $LOCAL_LOAD"
    echo -e "  RAM:  $(barra $LOCAL_RAM_PCT)  $LOCAL_RAM_USED / $LOCAL_RAM_TOTAL (Libre: $LOCAL_RAM_AVAIL)"
    echo -e "  ${DIM}Procesos: $LOCAL_PROCS total | Python: $LOCAL_PY_PROCS${NC}"
    echo ""

    # ── REMOTO (Worker) ──
    if $SERVIDOR_ONLINE; then
        REMOTE_DATA=$(sshpass -p "$PASSWORD" ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no \
            $SSH_USER@$SSH_HOST_IP_LAN "
echo \"\$(top -bn1 | grep 'Cpu(s)' | awk '{printf \"%.0f\", \$2+\$4}')\"
echo \"\$(free | awk '/Mem:/{printf \"%.0f\", \$3/\$2*100}')\"
echo \"\$(free -h | awk '/Mem:/{print \$3}')\"
echo \"\$(free -h | awk '/Mem:/{print \$2}')\"
echo \"\$(free -h | awk '/Mem:/{print \$7}')\"
echo \"\$(cat /proc/loadavg | awk '{print \$1, \$2, \$3}')\"
echo \"\$(ps aux | wc -l)\"
echo \"\$(ps aux | grep -c '[p]ython3' || echo 0)\"
" 2>/dev/null)

        if [ $? -eq 0 ] && [ -n "$REMOTE_DATA" ]; then
            IFS=$'\n' read -r -d '' R_CPU R_RAM_PCT R_RAM_USED R_RAM_TOTAL R_RAM_AVAIL R_LOAD R_PROCS R_PY <<< "$REMOTE_DATA" || true
            
            echo -e "  ${BOLD}SERVIDOR (Worker)${NC} — $SSH_HOST — $SSH_HOST_IP_LAN"
            echo -e "  CPU:  $(barra ${R_CPU:-0})  Load: ${R_LOAD:-?}"
            echo -e "  RAM:  $(barra ${R_RAM_PCT:-0})  ${R_RAM_USED:-?} / ${R_RAM_TOTAL:-?} (Libre: ${R_RAM_AVAIL:-?})"
            echo -e "  ${DIM}Procesos: ${R_PROCS:-?} total | Python: ${R_PY:-0}${NC}"
        else
            echo -e "  ${BOLD}SERVIDOR (Worker)${NC} — $SSH_HOST — $SSH_HOST_IP_LAN"
            echo -e "  ${RED}✗ Error al obtener datos (SSH timeout)${NC}"
        fi
    else
        echo -e "  ${BOLD}SERVIDOR (Worker)${NC} — $SSH_HOST — $SSH_HOST_IP_LAN"
        echo -e "  ${RED}✗ OFFLINE — No responde${NC}"
    fi

    echo ""
    echo -e "  ${DIM}─── Resumen del Pipeline ───${NC}"
    
    # Verificar procesos de Ray
    RAY_LOCAL=$(pgrep -c "raylet" 2>/dev/null || echo "0")
    if [ "$RAY_LOCAL" -gt 0 ]; then
        echo -e "  ${GREEN}● Ray activo en Master ($RAY_LOCAL procesos)${NC}"
    else
        echo -e "  ${DIM}○ Ray no activo en Master${NC}"
    fi

    if $SERVIDOR_ONLINE; then
        RAY_REMOTE=$(sshpass -p "$PASSWORD" ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no \
            $SSH_USER@$SSH_HOST_IP_LAN "pgrep -c raylet 2>/dev/null || echo 0" 2>/dev/null || echo "?")
        if [ "$RAY_REMOTE" != "0" ] && [ "$RAY_REMOTE" != "?" ]; then
            echo -e "  ${GREEN}● Ray activo en Worker ($RAY_REMOTE procesos)${NC}"
        else
            echo -e "  ${DIM}○ Ray no activo en Worker${NC}"
        fi
    fi

    # Verificar archivos procesados
    PANEL_DIR="/mnt/datasets/cristian_garcia_eduardo/tareas/proyecto/situacion_3/data_processed/panel_analitico"
    if [ -d "$PANEL_DIR" ]; then
        N_PARQUET=$(find "$PANEL_DIR" -name "*.parquet" 2>/dev/null | wc -l)
        if [ "$N_PARQUET" -gt 0 ]; then
            PARQUET_SIZE=$(du -sh "$PANEL_DIR" 2>/dev/null | awk '{print $1}')
            echo -e "  ${GREEN}● Parquet: $N_PARQUET archivos ($PARQUET_SIZE)${NC}"
        else
            echo -e "  ${DIM}○ Sin archivos Parquet generados${NC}"
        fi
    fi

    echo ""
    echo -e "  ${DIM}Siguiente actualización en ${INTERVALO}s...${NC}"
    
    sleep $INTERVALO
done
