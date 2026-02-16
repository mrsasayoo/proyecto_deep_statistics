#!/bin/bash
# ============================================================================
# ray_start.sh — Iniciar clúster Ray en ambas máquinas
# ============================================================================
# Uso: ./scripts/ray_start.sh        (inicia Master + Worker)
#      ./scripts/ray_start.sh master  (solo Master)
#      ./scripts/ray_start.sh worker  (solo Worker)
#      ./scripts/ray_start.sh stop    (detener todo)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../docs/red/.config_red.sh"

MASTER_IP=$(hostname -I | awk '{print $1}')
MASTER_PORT=6379
MODE=${1:-"all"}

start_master() {
    echo "🚀 Iniciando Ray HEAD en Master ($MASTER_IP:$MASTER_PORT)..."
    ray stop --force 2>/dev/null || true
    ray start --head --port=$MASTER_PORT --num-cpus=8 \
        --object-store-memory=4000000000 \
        --dashboard-host=0.0.0.0
    echo "✅ Master Ray activo"
    echo "   Dashboard: http://$MASTER_IP:8265"
}

start_worker() {
    echo "🚀 Iniciando Ray Worker en servidor ($SSH_HOST_IP_LAN)..."
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no $SSH_USER@$SSH_HOST_IP_LAN "
export PATH=\$HOME/.local/bin:\$PATH
ray stop --force 2>/dev/null || true
ray start --address='$MASTER_IP:$MASTER_PORT' --num-cpus=4 \
    --object-store-memory=2000000000
" 2>&1
    echo "✅ Worker Ray conectado al Master"
}

stop_all() {
    echo "⏹️  Deteniendo Ray en ambas máquinas..."
    ray stop --force 2>/dev/null || true
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no $SSH_USER@$SSH_HOST_IP_LAN \
        "export PATH=\$HOME/.local/bin:\$PATH; ray stop --force" 2>/dev/null || true
    echo "✅ Ray detenido en ambas máquinas"
}

status() {
    echo "📊 Estado de Ray:"
    echo ""
    echo "--- Master ---"
    ray status 2>/dev/null || echo "  No activo"
    echo ""
    echo "--- Worker ---"
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no $SSH_USER@$SSH_HOST_IP_LAN \
        "export PATH=\$HOME/.local/bin:\$PATH; ray status" 2>/dev/null || echo "  No activo"
}

case "$MODE" in
    all)
        start_master
        sleep 3
        start_worker
        echo ""
        echo "🎯 Clúster Ray activo: 12 CPUs combinados"
        ;;
    master)
        start_master
        ;;
    worker)
        start_worker
        ;;
    stop)
        stop_all
        ;;
    status)
        status
        ;;
    *)
        echo "Uso: $0 {all|master|worker|stop|status}"
        exit 1
        ;;
esac
