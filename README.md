# Analítica de procesos para ciberseguridad · prácticas
Cuatro prácticas y una demostración corta en Google Colab. Todas trabajan con **incidentes de seguridad y de TI**, y cada una responde una pregunta distinta con datos reales.

| # | Práctica | Pregunta | Datos | Abrir en Colab |
|---|---|---|---|---|
| 1 | **Clasificación** | ¿Este incidente es una amenaza real, no hay amenaza o es actividad esperada? | Incidentes de SOC de Microsoft (GUIDE) | [Abrir](https://colab.research.google.com/github/lsandinop/analitica-procesos-ciberseguridad/blob/main/cuadernos/Practica_1_Clasificacion.ipynb) |
| 2 | **Minería de procesos** | ¿Cómo se atiende *realmente* un incidente y dónde se pierde el tiempo? | Registro de incidentes de una mesa de servicio (ServiceNow) | [Abrir](https://colab.research.google.com/github/lsandinop/analitica-procesos-ciberseguridad/blob/main/cuadernos/Practica_2_Mineria_de_procesos.ipynb) |
| 3 | **Segmentación** | ¿Qué tipos de incidentes hay según cómo se atienden? | El mismo registro de ServiceNow | [Abrir](https://colab.research.google.com/github/lsandinop/analitica-procesos-ciberseguridad/blob/main/cuadernos/Practica_3_Segmentacion.ipynb) |
| 4 | **Detección de anomalías** | ¿Qué incidentes se comportan de forma rara? | El mismo registro de ServiceNow | [Abrir](https://colab.research.google.com/github/lsandinop/analitica-procesos-ciberseguridad/blob/main/cuadernos/Practica_4_Anomalias.ipynb) |
| Demo 2 | **Tablero de KPIs** (unos 15 min) | ¿Cómo se ve el proceso en un tablero de control con indicadores? | El mismo registro de ServiceNow | [Abrir](https://colab.research.google.com/github/lsandinop/analitica-procesos-ciberseguridad/blob/main/cuadernos/Demo_2_Tablero_de_KPIs.ipynb) |

## Cómo usarlas
1. Abre una práctica con el enlace de la tabla e inicia sesión con tu cuenta de Google. Cada persona trabaja en **su propia copia temporal**: no puedes modificar este repositorio.
2. Si Colab avisa que el cuaderno no fue creado por Google, elige **Ejecutar de todos modos**.
3. Ejecuta las celdas **en orden** (`Entorno de ejecución → Ejecutar todo`, o `Shift + Enter` celda por celda). No hay que instalar nada a mano: la Práctica 2 instala sola la librería de minería de procesos (PM4Py).
4. Si la sesión se desconecta, vuelve a ejecutar desde la primera celda.

No necesitas GPU ni cuenta de pago. Cada práctica toma unos 45 minutos.

## Datos
Los datos están en `datos/` y cada práctica los descarga sola desde aquí. **Fuentes, licencias y cambios** (se tradujeron al español, se resumieron y se tomó una muestra) en [`datos/LEEME_datos.md`](datos/LEEME_datos.md):
- **GUIDE** (Microsoft): licencia CDLA-Permissive-2.0 (el texto del acuerdo está en `datos/`).
- **Registro de incidentes de ServiceNow** (Amaral et al., UCI): licencia CC BY 4.0.

Los scripts `scripts/preparar_guide.py` y `scripts/preparar_servicenow.py` reproducen los archivos a partir de los originales.
