#!/bin/bash
# CHEAT SHEET - Comandos Esenciales de Red
# Ejecutar: ./comandos_red.sh [opcion]

# Cargar configuración (si existe)
if [ -f "$(dirname "$0")/.config_red.sh" ]; then
    source "$(dirname "$0")/.config_red.sh"
fi

case "$1" in
    "ips")
        echo "=== IPs Actuales ==="
        echo "Laptop:   192.168.1.17 (LAN) | 100.116.239.1 (Tailscale)"
        echo "Servidor: 192.168.1.15 (LAN) | 100.110.27.62 (Tailscale)"
        echo "Pública:  $(curl -s ifconfig.me)"
        ;;
    
    "ssh")
        echo "=== Conectar al Servidor ==="
        echo "# Método manual (pide contraseña):"
        echo "ssh mrsasayo_mesa@nicolasmesa         # Via Tailscale"
        echo "ssh mrsasayo_mesa@100.110.27.62       # Via IP Tailscale"
        echo "ssh mrsasayo_mesa@192.168.1.15        # Via LAN local"
        echo ""
        echo "# Método rápido (sin contraseña):"
        echo "./ssh_rapido.sh                        # Sesión interactiva"
        echo "./ssh_rapido.sh 'hostname'             # Ejecutar comando"
        echo ""
        echo "# Copiar archivos:"
        echo "./copiar_servidor.sh archivo.txt       # Subir archivo"
        echo "./descargar_servidor.sh ~/datos.csv    # Descargar archivo"
        ;;
    
    "rapido")
        if [ ! -f "$(dirname "$0")/ssh_rapido.sh" ]; then
            echo "❌ Error: ssh_rapido.sh no existe"
            exit 1
        fi
        shift
        ./ssh_rapido.sh "$@"
        ;;
    
    "ping")
        echo "=== Verificar Conectividad ==="
        ping -c 2 192.168.1.15       # LAN
        ping -c 2 100.110.27.62      # Tailscale
        ;;
    
    "status")
        echo "=== Estado Tailscale ==="
        tailscale status
        ;;
    
    "publica")
        echo "=== IP Pública Actual ==="
        curl -s ifconfig.me
        echo ""
        ;;
    
    "dns")
        echo "=== Verificar Duck DNS ==="
        echo "Dominio: mrsasayomesa.duckdns.org"
        echo "IP: $(dig +short mrsasayomesa.duckdns.org)"
        echo "Esperada: $(curl -s ifconfig.me)"
        ;;
    
    "full")
        ./verificar_red.sh
        ;;
    
    *)
        echo "╔═══════════════════════════════════════════════════════════╗"
        echo "║           CHEAT SHEET - Comandos de Red                  ║"
        echo "╚═══════════════════════════════════════════════════════════╝"
        echo ""
        echo "Uso: ./comandos_red.sh [opcion]"
        echo ""
        echo "Opciones disponibles:"
        echo "  ips      - Ver todas las IPs (LAN, Tailscale, Pública)"
        echo "  ssh      - Comandos para conectar al servidor"
        echo "  rapido   - SSH rápido sin contraseña (./ssh_rapido.sh)"
        echo "  ping     - Verificar conectividad con servidor"
        echo "  status   - Estado de Tailscale"
        echo "  publica  - Ver IP pública actual"
        echo "  dns      - Verificar Duck DNS"
        echo "  full     - Verificación completa (ejecuta verificar_red.sh)"
        echo ""
        echo "Ejemplos:"
        echo "  ./comandos_red.sh ips"
        echo "  ./comandos_red.sh ssh"
        echo "  ./comandos_red.sh rapido 'ls -la'           # Ejecutar comando"
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "ACCESO RÁPIDO (sin contraseña):"
        echo "  ./ssh_rapido.sh                             # SSH interactivo"
        echo "  ./ssh_rapido.sh 'hostname'                  # Ejecutar comando"
        echo "  ./copiar_servidor.sh archivo.txt            # Subir archivo"
        echo "  ./descargar_servidor.sh ~/datos.csv         # Descargar"
        echo "═══════════════════════════════════════════════════════════"
        echo "DATOS RÁPIDOS:"
        echo "  Servidor: mrsasayo_mesa@nicolasmesa (nicolasmesa = 100.110.27.62)"
        echo "  Password: 137919"
        echo "  IP Servidor LAN: 192.168.1.15"
        echo "  IP Pública: $(curl -s --max-time 2 ifconfig.me 2>/dev/null || echo 'Verificar manualmente')"
        echo "═══════════════════════════════════════════════════════════"
        ;;
esac
