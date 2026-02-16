#!/bin/bash
# ============================================================================
# test_parallelization.sh — Verificar que la paralelización funciona
# ============================================================================
# Ejecuta un test rápido de Ray entre ambas máquinas

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../docs/red/.config_red.sh"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     TEST DE PARALELIZACIÓN — Ray entre 2 máquinas           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 1. Verificar conectividad
echo "1. Verificando conectividad..."
if ping -c 1 -W 2 $SSH_HOST_IP_LAN &>/dev/null; then
    LATENCY=$(ping -c 3 -W 2 $SSH_HOST_IP_LAN | tail -1 | awk -F'/' '{printf "%.1f", $5}')
    echo "   ✅ Servidor accesible — Latencia: ${LATENCY}ms"
else
    echo "   ❌ Servidor NO accesible. Abortando."
    exit 1
fi

# 2. Verificar SSH
echo "2. Verificando SSH..."
SSH_OK=$(sshpass -p "$PASSWORD" ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no \
    $SSH_USER@$SSH_HOST_IP_LAN "echo OK" 2>/dev/null)
if [ "$SSH_OK" == "OK" ]; then
    echo "   ✅ SSH funciona"
else
    echo "   ❌ SSH fallido. Abortando."
    exit 1
fi

# 3. Verificar NFS
echo "3. Verificando NFS..."
if mount | grep -q "nfs"; then
    NFS_SPEED=$(dd if=/mnt/datasets/cristian_garcia_eduardo/tareas/proyecto/situacion_3/dataset/Performance.zip \
        bs=1M count=100 2>&1 | grep -oP '[\d,.]+\s+[MG]B/s' || echo "?")
    echo "   ✅ NFS montado — Velocidad lectura: $NFS_SPEED"
else
    echo "   ⚠️  NFS no montado"
fi

# 4. Verificar Ray en ambos
echo "4. Verificando Ray..."
LOCAL_RAY=$(python3 -c "import ray; print(ray.__version__)" 2>/dev/null || echo "MISSING")
REMOTE_RAY=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no $SSH_USER@$SSH_HOST_IP_LAN \
    "export PATH=\$HOME/.local/bin:\$PATH; python3 -c 'import ray; print(ray.__version__)'" 2>/dev/null || echo "MISSING")
echo "   Local:  Ray $LOCAL_RAY"
echo "   Remoto: Ray $REMOTE_RAY"

if [ "$LOCAL_RAY" == "MISSING" ] || [ "$REMOTE_RAY" == "MISSING" ]; then
    echo "   ❌ Ray no instalado en alguna máquina. Abortando."
    exit 1
fi
echo "   ✅ Ray instalado en ambas"

# 5. Test funcional de Ray
echo "5. Test funcional de Ray (cómputo distribuido)..."
echo ""

python3 << 'PYEOF'
import ray
import time
import os

print("   Iniciando Ray en modo local (single-node test)...")
ray.init(num_cpus=4, ignore_reinit_error=True)

@ray.remote
def tarea_pesada(n):
    """Simula procesamiento: suma de cuadrados."""
    total = sum(i**2 for i in range(n))
    return {"pid": os.getpid(), "result": total, "node": ray.get_runtime_context().get_node_id()[:8]}

# Ejecutar 8 tareas en paralelo
print("   Lanzando 8 tareas de prueba...")
t0 = time.time()
futures = [tarea_pesada.remote(500_000) for _ in range(8)]
results = ray.get(futures)
elapsed = time.time() - t0

pids = set(r["pid"] for r in results)
nodes = set(r["node"] for r in results)

print(f"   ✅ 8 tareas completadas en {elapsed:.2f}s")
print(f"   PIDs usados: {len(pids)} — Nodos: {len(nodes)}")
print(f"   Resultado ejemplo: {results[0]['result']:,}")

# Info del cluster
cluster = ray.cluster_resources()
print(f"\n   Recursos del clúster: {cluster.get('CPU', '?')} CPUs, "
      f"{cluster.get('memory', 0)/1e9:.1f} GB memory")

ray.shutdown()
print("\n   ✅ Ray funciona correctamente")
PYEOF

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Para iniciar el clúster distribuido (2 máquinas):"
echo "  ./scripts/ray_start.sh all"
echo "══════════════════════════════════════════════════════════════"
