# Guía de Ejecución - Reconocimiento D435

### 1. Lanzamiento de la Cámara
Ejecuta el nodo de la cámara RealSense en una terminal:

```bash
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true rgb_camera.profile:=1280x720x30 depth_module.profile:=640x480x30
```

### 2. Configuración del Entorno Virtual
Crea y activa el entorno para gestionar las dependencias:

```bash
# Crear el entorno virtual
python3 -m venv venv

# Activar el entorno
source venv/bin/activate
```

### 3. Instalación de Dependencias
Instala Ultralytics dentro del entorno virtual activo:

```bash
pip install ultralytics
```

### 4. Ejecución del Algoritmo
Lanza el script de reconocimiento (asegúrate de usar la ruta correcta):

```bash
python /ruta_completa_al_archivo/reconocimiento.py
```

### 5. Visualización en RViz2
Para visualizar los resultados:
1. Abre una terminal y ejecuta `rviz2`.
2. Añade un visualizador de **Image**.
3. En **Image Topic**, selecciona: `reconocimiento imagen`.
4. En los ajustes de QoS, selecciona: **Best Effort**.
