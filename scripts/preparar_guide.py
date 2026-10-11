"""
Prepara la tabla de incidentes de SOC para la Práctica 1 (Clasificación) a partir de GUIDE.

Fuente: Freitas, S., Kalajdjieski, J., Gharib, A., McCann, R. (2025). "AI-Driven Guided Response for Security
Operation Centers with Microsoft Copilot for Security". Companion Proceedings of the ACM Web Conference 2025, pp. 191-200.
Conjunto «Microsoft Security Incident Prediction» (GUIDE), Kaggle. Licencia CDLA-Permissive-2.0.

Qué hace (cambios respecto del original, que la licencia pide conservar con su atribución):
  1. El original trae una fila por cada evidencia de cada alerta (4,1 millones de filas en GUIDE_Test.csv). Aquí se
     resume en UNA FILA POR INCIDENTE: cuántas alertas y evidencias tiene, cuánto dura, qué tácticas MITRE ATT&CK
     aparecen y en qué proporción, qué tipos de entidades involucra y qué veredictos automáticos recibió.
  2. Los nombres de columnas pasan a español y las etiquetas de triaje reciben nombres propios (amenaza real = TruePositive,
     sin amenaza = FalsePositive, actividad esperada = BenignPositive), para no confundirlas con los errores de un modelo.
  3. Se toma una muestra aleatoria de incidentes (semilla fija) para que el archivo se pueda descargar en Colab.
  4. Del detector que generó cada alerta solo se conserva el identificador del más frecuente en el incidente, y
     solo si está entre los 200 más comunes (si no, -1).

Uso (desde la carpeta del curso):
    python scripts/preparar_guide.py                                 # lee fuentes/GUIDE_Test.csv y escribe datos/guide_incidentes.csv
    python scripts/preparar_guide.py RUTA/GUIDE_Test.csv 80000      # otra ruta y número de incidentes de la muestra
    python scripts/preparar_guide.py RUTA/GUIDE_Test.csv 80000 SALIDA.csv
El archivo GUIDE_Test.csv (1,09 GB) se descarga de Kaggle; NO se publica en el repositorio (está en fuentes/, que el .gitignore excluye).
"""
import sys

import numpy as np
import pandas as pd

ORIGEN = sys.argv[1] if len(sys.argv) > 1 else "fuentes/GUIDE_Test.csv"
N_INCIDENTES = int(sys.argv[2]) if len(sys.argv) > 2 else 80_000
DESTINO = sys.argv[3] if len(sys.argv) > 3 else "datos/guide_incidentes.csv"
SEMILLA = 123456789

COLUMNAS = ["OrgId", "IncidentId", "AlertId", "Timestamp", "DetectorId", "Category", "MitreTechniques", "IncidentGrade",
            "EntityType", "EvidenceRole", "SuspicionLevel", "LastVerdict", "CountryCode"]
GRADO = {"TruePositive": "amenaza real", "FalsePositive": "sin amenaza", "BenignPositive": "actividad esperada"}
TACTICAS = {"InitialAccess": "acceso_inicial", "Exfiltration": "exfiltracion", "SuspiciousActivity": "actividad_sospechosa",
            "CommandAndControl": "comando_y_control", "Impact": "impacto", "CredentialAccess": "acceso_a_credenciales",
            "Execution": "ejecucion", "Malware": "malware", "Discovery": "descubrimiento", "Persistence": "persistencia",
            "DefenseEvasion": "evasion_de_defensas", "LateralMovement": "movimiento_lateral"}
ENTIDADES = {"Ip": "ip", "User": "usuario", "MailMessage": "mensaje_correo", "Machine": "equipo", "CloudLogonRequest": "inicio_sesion_nube",
             "Url": "url", "File": "archivo", "Mailbox": "buzon", "Process": "proceso", "MailCluster": "grupo_correos"}

d = pd.read_csv(ORIGEN, usecols=COLUMNAS, parse_dates=["Timestamp"], low_memory=False,
                dtype={"OrgId": "int32", "IncidentId": "int32", "AlertId": "int32", "DetectorId": "int32"})
d["inc"] = d.OrgId.astype("int64") * 10_000_000 + d.IncidentId           # un identificador único por organización + incidente
g = d.groupby("inc", observed=True)

t = pd.DataFrame({
    "organizacion": g.OrgId.first(),
    "grado": g.IncidentGrade.first().map(GRADO),                         # la etiqueta que puso el analista del cliente
    "n_alertas": g.AlertId.nunique(), "n_evidencias": g.size(), "n_detectores": g.DetectorId.nunique(),
    "n_tacticas": g.Category.nunique(),
    "minutos_duracion": (g.Timestamp.max() - g.Timestamp.min()).dt.total_seconds() / 60,
    "hora_inicio": g.Timestamp.min().dt.hour, "dia_semana": g.Timestamp.min().dt.dayofweek, "n_paises": g.CountryCode.nunique()})
t["prop_con_tecnica_mitre"] = d.MitreTechniques.notna().groupby(d.inc).mean()
t["prop_nivel_sospecha"] = d.SuspicionLevel.notna().groupby(d.inc).mean()
t["prop_veredicto_malicioso"] = (d.LastVerdict == "Malicious").groupby(d.inc).mean()
t["prop_veredicto_sin_amenazas"] = (d.LastVerdict == "NoThreatsFound").groupby(d.inc).mean()
t["prop_equipo_afectado"] = (d.EvidenceRole == "Impacted").groupby(d.inc).mean()
for original, nombre in TACTICAS.items():
    t["tactica_" + nombre] = (d.Category == original).groupby(d.inc).mean()
for original, nombre in ENTIDADES.items():
    t["entidad_" + nombre] = (d.EntityType == original).groupby(d.inc).mean()
principal = g.DetectorId.agg(lambda s: s.value_counts().index[0])       # el detector que más alertas generó en el incidente
frecuentes = principal.value_counts().head(200).index
t["detector_principal"] = principal.where(principal.isin(frecuentes), -1)

t = t.dropna(subset=["grado"]).sample(n=min(N_INCIDENTES, len(t)), random_state=SEMILLA).reset_index(drop=True)
t["minutos_duracion"] = t.minutos_duracion.round(1)
for c in t.columns[t.dtypes == "float64"]:
    t[c] = t[c].round(4)
t.to_csv(DESTINO, index=False, encoding="utf-8")
print(f"{DESTINO}: {len(t):,} incidentes · {t.organizacion.nunique():,} organizaciones · {t.shape[1]} columnas")
print(t.grado.value_counts(normalize=True).round(3).to_dict())
