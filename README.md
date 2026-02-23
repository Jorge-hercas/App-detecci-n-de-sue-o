# SleepGuard · Detector de Somnolencia

App Python que usa tu cámara para detectar cuando te quedas dormido
y reproduce un video de alarma en el navegador.

## Tecnologías
- **Flask** — servidor web
- **OpenCV** — captura de cámara y dibujo
- **MediaPipe Face Mesh** — landmarks faciales (478 puntos)
- **Eye Aspect Ratio (EAR)** — métrica para detectar ojos cerrados

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python app.py
```

Abrir en: **http://localhost:5050**

## Funcionamiento

El algoritmo EAR (Eye Aspect Ratio) mide la relación entre la altura
y el ancho de cada ojo usando landmarks de MediaPipe.

- EAR alto (~0.30+) = ojos abiertos
- EAR bajo (<0.22)  = ojos cerrados