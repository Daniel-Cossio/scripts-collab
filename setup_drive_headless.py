# ==============================================================================
# MONTAJE HEADLESS DE GOOGLE DRIVE EN GOOGLE COLAB (RCLONE VFS CACHE)
# Monta /content/drive de forma 100% desatendida sin necesidad de navegador.
# ==============================================================================
import os
import subprocess
import time
from pathlib import Path

def ensure_drive_mounted() -> bool:
    """Verifica si /content/drive/MyDrive está montado; si no, lo monta usando Rclone."""
    drive_root = Path("/content/drive")
    my_drive = drive_root / "MyDrive"

    if my_drive.exists():
        print("✅ Google Drive ya está montado y listo en /content/drive/MyDrive.")
        return True

    rclone_conf_path = Path("/root/.config/rclone/rclone.conf")
    content_conf = Path("/content/rclone.conf")
    if not rclone_conf_path.exists() and content_conf.exists():
        rclone_conf_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(content_conf, rclone_conf_path)

    if not rclone_conf_path.exists():
        print("⚠️ No se encontró rclone.conf en /root/.config/rclone/ ni en /content/. Omitiendo montaje headless.")
        return False

    print("🚀 Montando Google Drive vía Rclone en modo headless...")

    # 1. Instalar binario de rclone si no existe en la VM
    if not Path("/usr/bin/rclone").exists() and not Path("/usr/local/bin/rclone").exists():
        print("   -> Instalando rclone en Colab VM...")
        subprocess.run("curl -s https://rclone.org/install.sh | bash", shell=True, check=True)

    # 2. Instalar y asegurar fuse3
    subprocess.run("apt-get update -qq && apt-get install -y -qq fuse3 >/dev/null 2>&1", shell=True)

    # 3. Crear punto de montaje
    my_drive.mkdir(parents=True, exist_ok=True)

    # 4. Lanzar rclone mount en segundo plano
    cmd = [
        "rclone", "mount",
        "gdrive:", str(my_drive),
        "--vfs-cache-mode", "full",
        "--vfs-cache-max-age", "24h",
        "--buffer-size", "64M",
        "--allow-other",
        "--daemon"
    ]
    
    subprocess.run(cmd)

    # 5. Esperar confirmación del punto de montaje
    for _ in range(15):
        time.sleep(1)
        if my_drive.exists() and any(my_drive.iterdir()):
            print("✅ Google Drive montado exitosamente en modo headless.")
            return True

    print("⚠️ Timeout esperando que Rclone monte Google Drive. Continuando en modo local.")
    return False

if __name__ == "__main__":
    ensure_drive_mounted()
