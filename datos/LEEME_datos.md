# Datos de las prácticas

| Archivo | Práctica | Qué es |
|---|---|---|
| `guide_incidentes.csv` | 1 · Clasificación | 80.000 incidentes de SOC (una fila por incidente) con la etiqueta que les puso un analista |
| `incidentes_servicenow.csv` | 2, 3 y 4 | Registro de eventos de una mesa de servicio de TI (24.916 incidentes, 141.697 eventos) |
| `LICENCIA_GUIDE_CDLA-Permissive-2.0.txt` | | Texto del acuerdo bajo el cual se publica GUIDE (el acuerdo pide acompañar con él los datos que se comparten) |

## `guide_incidentes.csv`
**Fuente:** conjunto GUIDE, «Microsoft Security Incident Prediction» (Kaggle). Freitas, S., Kalajdjieski, J., Gharib, A. y McCann, R. (2025). *AI-Driven Guided Response for Security Operation Centers with Microsoft Copilot for Security.* Companion Proceedings of the ACM Web Conference 2025, pp. 191-200. https://doi.org/10.1145/3701716.3715209
**Licencia:** Community Data License Agreement – Permissive, Versión 2.0 (CDLA-Permissive-2.0); el texto está en este mismo directorio.

**Cambios respecto del original** (hechos con fines académicos; los reproduce `scripts/preparar_guide.py`):
1. El original trae una fila por cada evidencia de cada alerta; aquí se resume en **una fila por incidente** (cuántas alertas y pruebas tiene, su duración, la proporción de cada táctica MITRE ATT&CK y de cada tipo de entidad, y los veredictos automáticos).
2. Se tomó una **muestra aleatoria de 80.000 incidentes** (semilla 123456789) del archivo `GUIDE_Test.csv`.
3. Nombres de columnas traducidos al español. Las etiquetas de triaje reciben nombres propios, para no confundirlas con los errores de un modelo: **amenaza real** (TruePositive), **sin amenaza** (FalsePositive) y **actividad esperada** (BenignPositive). Del detector solo se conserva el identificador del más frecuente en el incidente (si está entre los 200 más comunes; si no, -1).

## `incidentes_servicenow.csv`
**Fuente:** Amaral, C. A. M., Fantinato, M., Reijers, H. A. y Peres, S. M. (2018). *Incident management process enriched event log.* UCI Machine Learning Repository. https://doi.org/10.24432/C57S4H
**Licencia:** Creative Commons Atribución 4.0 Internacional (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. Los datos personales ya vienen anonimizados por los autores.

**Cambios respecto del original** (hechos con fines académicos; los reproduce `scripts/preparar_servicenow.py`):
1. Se quitaron los 2 incidentes con el estado inválido «-100» (5 filas).
2. Se conservaron solo las columnas necesarias y se tradujeron al español sus nombres, los estados, las prioridades, los impactos, las urgencias y los tipos de contacto.
3. Las fechas pasaron de día/mes/año a AAAA-MM-DD hh:mm.
