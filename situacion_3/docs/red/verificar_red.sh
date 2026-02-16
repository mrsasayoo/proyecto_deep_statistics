#!/bin/bash

# Script de verificación rápida del estado de red
# Ejecutar: ./verificar_red.sh

# Configuración de credenciales
PASSWORD="137919"
SSH_USER="mrsasayo_mesa"
SSH_HOST="nicolasmesa"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       VERIFICACIÓN RÁPIDA DE RED - Estado Actual              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 1. Hostname
echo "┌─ 1. ESTE DISPOSITIVO ────────────────────────────────────────┐"
echo "│ Hostname: $(hostname)"
echo "│ Usuario:  $USER"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# 2. IP Pública
echo "┌─ 2. IP PÚBLICA (WAN) ────────────────────────────────────────┐"
IP_PUBLICA=$(curl -s --max-time 3 ifconfig.me 2>/dev/null || echo "Error al obtener")
echo "│ IP Pública: $IP_PUBLICA"
echo "│ Duck DNS:   mrsasayomesa.duckdns.org"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# 3. IP Local
echo "┌─ 3. IP LOCAL (LAN) ──────────────────────────────────────────┐"
IP_LOCAL=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v 127.0.0.1 | grep "192.168" | head -n1)
GATEWAY=$(ip route | grep default | awk '{print $3}')
echo "│ IP Local:  $IP_LOCAL"
echo "│ Gateway:   $GATEWAY (Router)"
echo "│ Red:       192.168.1.0/24"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# 4. Tailscale
echo "┌─ 4. TAILSCALE VPN ───────────────────────────────────────────┐"
if command -v tailscale &> /dev/null; then
    TS_STATUS=$(tailscale status 2>&1)
    if [ $? -eq 0 ]; then
        echo "│ Estado: ✅ ACTIVO"
        TS_IP=$(tailscale ip -4 2>/dev/null)
        echo "│ Mi IP Tailscale: $TS_IP"
        echo "│"
        echo "│ Dispositivos conectados:"
        tailscale status | while IFS= read -r line; do
            echo "│   $line"
        done
    else
        echo "│ Estado: ❌ NO ACTIVO"
    fi
else
    echo "│ Estado: ⚠️  Tailscale no instalado"
fi
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# 5. Conectividad Servidor
echo "┌─ 5. CONECTIVIDAD SERVIDOR ───────────────────────────────────┐"
echo "│ Probando conexión a nicolasmesa (servidor)..."
echo "│"

# Ping por Tailscale (si existe)
if command -v tailscale &> /dev/null; then
    PING_TS=$(ping -c 1 -W 2 100.110.27.62 2>/dev/null | grep "time=" | awk -F'time=' '{print $2}' | awk '{print $1}')
    if [ ! -z "$PING_TS" ]; then
        echo "│ ✅ Tailscale (100.110.27.62): ${PING_TS}ms"
    else
        echo "│ ❌ Tailscale (100.110.27.62): Sin respuesta"
    fi
fi

# Ping por LAN
PING_LAN=$(ping -c 1 -W 2 192.168.1.15 2>/dev/null | grep "time=" | awk -F'time=' '{print $2}' | awk '{print $1}')
if [ ! -z "$PING_LAN" ]; then
    echo "│ ✅ LAN (192.168.1.15): ${PING_LAN}ms"
else
    echo "│ ❌ LAN (192.168.1.15): Sin respuesta"
fi

# SSH test con contraseña automática
SSH_TEST=$(timeout 3 sshpass -p "$PASSWORD" ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no $SSH_USER@$SSH_HOST echo "OK" 2>/dev/null)
if [ "$SSH_TEST" == "OK" ]; then
    echo "│ ✅ SSH: Conexión exitosa (automática)"
else
    echo "│ ⚠️  SSH: Sin respuesta"
fi

echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# 6. Interfaces Activas
echo "┌─ 6. INTERFACES DE RED ACTIVAS ───────────────────────────────┐"
ip -4 addr show | grep -E "^[0-9]+:|inet " | grep -v "127.0.0.1" | while IFS= read -r line; do
    if [[ $line =~ ^[0-9]+: ]]; then
        # Es nombre de interfaz
        IF_NAME=$(echo "$line" | awk '{print $2}' | tr -d ':')
        IF_STATE=$(echo "$line" | grep -o "state [A-Z]*" | awk '{print $2}')
        echo "│"
        echo "│ ┌─ $IF_NAME ($IF_STATE)"
    elif [[ $line =~ inet ]]; then
        # Es dirección IP
        IF_IP=$(echo "$line" | awk '{print $2}')
        echo "│ ├─ IP: $IF_IP"
    fi
done
echo "│"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# 7. Puertos Escuchando (top 5)
echo "┌─ 7. SERVICIOS ESCUCHANDO (Top 5) ───────────────────────────┐"
if command -v ss &> /dev/null; then
    echo "$PASSWORD" | sudo -S ss -tlpn 2>/dev/null | grep LISTEN | head -n 5 | awk '{print "│ " $4 " - " $6}' || echo "│ No se puede obtener información"
else
    echo "│ Comando 'ss' no disponible"
fi
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# 8. Resumen
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                          RESUMEN                               ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║ Para conectar al servidor:                                     ║"
echo "║   ssh mrsasayo_mesa@nicolasmesa        (vía Tailscale)         ║"
echo "║   ssh mrsasayo_mesa@100.110.27.62      (vía IP Tailscale)     ║"
echo "║   ssh mrsasayo_mesa@192.168.1.15       (vía LAN local)        ║"
echo "║                                                                 ║"
echo "║ Duck DNS: mrsasayomesa.duckdns.org → $IP_PUBLICA  ║"
echo "║                                                                 ║"
echo "║ Información completa: informacion_red_actual.md                ║"
echo "╚════════════════════════════════════════════════════════════════╝"

echo ""
echo "Generado: $(date '+%Y-%m-%d %H:%M:%S')"
