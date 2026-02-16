#!/usr/bin/env python3
"""
Ejemplo de Paralelización con Tailscale
Ejecuta tareas en laptop y servidor simultáneamente
"""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

# Configuración de credenciales
SSH_PASSWORD = "137919"
SSH_USER = "mrsasayo_mesa"

# Configuración de tus máquinas (Tailscale)
HOSTS = {
    'laptop': 'localhost',  # O '100.116.239.1'
    'servidor': '100.110.27.62'  # nicolasmesa
}

def ejecutar_en_host(host_name, comando):
    """Ejecuta comando en un host específico"""
    host = HOSTS[host_name]
    
    if host == 'localhost':
        # Ejecutar localmente
        resultado = subprocess.run(
            comando,
            shell=True,
            capture_output=True,
            text=True
        )
    else:
        # Ejecutar por SSH con contraseña automática
        comando_escapado = comando.replace("'", "'\"'\"'")  # Escapar comillas simples
        ssh_comando = f"sshpass -p '{SSH_PASSWORD}' ssh -o StrictHostKeyChecking=no {SSH_USER}@{host} '{comando_escapado}'"
        resultado = subprocess.run(
            ssh_comando,
            shell=True,
            capture_output=True,
            text=True
        )
    
    return {
        'host': host_name,
        'comando': comando,
        'stdout': resultado.stdout,
        'stderr': resultado.stderr,
        'exitcode': resultado.returncode
    }

def tarea_ejemplo(host_name, numero):
    """Ejemplo de tarea computacional"""
    comando = f"python3 -c 'import time; print(\"Host: {host_name}, Tarea: {numero}\"); time.sleep(2); print(\"Completado: {numero}\")'"
    return ejecutar_en_host(host_name, comando)

def main():
    print("="*60)
    print("Ejemplo de Paralelización con Tailscale")
    print("="*60)
    
    # Verificar conectividad
    print("\n1. Verificando conectividad...")
    for host_name in HOSTS.keys():
        resultado = ejecutar_en_host(host_name, "hostname")
        if resultado['exitcode'] == 0:
            print(f"   ✅ {host_name}: {resultado['stdout'].strip()}")
        else:
            print(f"   ❌ {host_name}: Error de conexión")
            return
    
    # Ejemplo 1: Ejecutar comandos en paralelo
    print("\n2. Ejecutando tareas en paralelo...")
    inicio = time.time()
    
    tareas = [
        ('laptop', 1),
        ('laptop', 2),
        ('servidor', 3),
        ('servidor', 4),
    ]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futuros = [executor.submit(tarea_ejemplo, host, num) for host, num in tareas]
        resultados = [f.result() for f in futuros]
    
    fin = time.time()
    
    print(f"\n3. Resultados (tiempo total: {fin-inicio:.2f}s):")
    for r in resultados:
        print(f"\n   Host: {r['host']}")
        print(f"   Salida: {r['stdout'].strip()}")
    
    # Ejemplo 2: Obtener información del sistema
    print("\n4. Información de recursos:")
    info_comandos = {
        'CPU': "lscpu | grep 'CPU(s):' | head -n1",
        'Memoria': "free -h | grep Mem | awk '{print $2}'",
        'Disco': "df -h / | tail -n1 | awk '{print $4}'"
    }
    
    for host_name in HOSTS.keys():
        print(f"\n   {host_name.upper()}:")
        for nombre, comando in info_comandos.items():
            resultado = ejecutar_en_host(host_name, comando)
            if resultado['exitcode'] == 0:
                print(f"      {nombre}: {resultado['stdout'].strip()}")
    
    print("\n" + "="*60)
    print("✅ Ejemplo completado exitosamente")
    print("="*60)

if __name__ == "__main__":
    main()
