"""
Prepara el registro de eventos de incidentes para las Prácticas 2, 3 y 4 (minería de procesos, segmentación y anomalías).

Fuente: Amaral, C. A. M., Fantinato, M., Reijers, H. A., Peres, S. M. (2018). "Incident management process enriched event
log". UCI Machine Learning Repository, https://doi.org/10.24432/C57S4H. Licencia CC BY 4.0. Datos de una instancia de
ServiceNow de una empresa de TI, con los datos personales ya anonimizados por los autores.

Qué hace (cambios respecto del original, que la licencia pide indicar):
  1. Se quedan los 2 incidentes con el estado «-100» (un valor inválido: 5 filas) fuera.
  2. Se conservan solo las columnas necesarias y sus nombres pasan a español.
  3. Los estados, prioridades, impactos, urgencias y tipos de contacto se traducen al español.
  4. Las fechas, que vienen como día/mes/año, pasan al formato AAAA-MM-DD hh:mm.
  5. Cada fila sigue siendo un EVENTO: un cambio del incidente (de estado, de grupo, de analista). No se resume nada.

Uso (desde la carpeta del curso):
    python scripts/preparar_servicenow.py          # lee fuentes/incident_event_log.csv y escribe datos/incidentes_servicenow.csv
"""
import pandas as pd

ORIGEN = "fuentes/incident_event_log.csv"
DESTINO = "datos/incidentes_servicenow.csv"

ESTADO = {"New": "Nuevo", "Active": "En curso", "Resolved": "Resuelto", "Closed": "Cerrado",
          "Awaiting User Info": "Esperando al usuario", "Awaiting Vendor": "Esperando al proveedor",
          "Awaiting Problem": "Esperando análisis del problema", "Awaiting Evidence": "Esperando evidencia"}
PRIORIDAD = {"1 - Critical": "1 - Crítica", "2 - High": "2 - Alta", "3 - Moderate": "3 - Moderada", "4 - Low": "4 - Baja"}
NIVEL = {"1 - High": "1 - Alto", "2 - Medium": "2 - Medio", "3 - Low": "3 - Bajo"}
CONTACTO = {"Phone": "Teléfono", "Self service": "Autoservicio", "Email": "Correo", "IVR": "Línea automática", "Direct opening": "Apertura directa"}

df = pd.read_csv(ORIGEN, na_values=["?"], low_memory=False)
invalidos = df.loc[df.incident_state == "-100", "number"].unique()
df = df[~df.number.isin(invalidos)].copy()

def fecha(columna):
    return pd.to_datetime(df[columna], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

out = pd.DataFrame({
    "incidente": df.number,
    "estado": df.incident_state.map(ESTADO),
    "fecha_evento": fecha("sys_updated_at"),
    "fecha_apertura": fecha("opened_at"),
    "fecha_resolucion": fecha("resolved_at"),
    "fecha_cierre": fecha("closed_at"),
    "prioridad": df.priority.map(PRIORIDAD),
    "impacto": df.impact.map(NIVEL),
    "urgencia": df.urgency.map(NIVEL),
    "categoria": df.category.str.replace("Category", "Categoría"),
    "grupo_asignado": df.assignment_group.str.replace("Group", "Grupo"),
    "tipo_contacto": df.contact_type.map(CONTACTO),
    "reasignaciones": df.reassignment_count,
    "reaperturas": df.reopen_count,
    "modificaciones": df.sys_mod_count,
    "uso_base_conocimiento": df.knowledge,
    "cumplio_sla": df.made_sla,
})
out = out.sort_values(["incidente", "fecha_evento"], kind="stable")
out.to_csv(DESTINO, index=False, encoding="utf-8")
print(f"{DESTINO}: {len(out):,} eventos · {out.incidente.nunique():,} incidentes · se quitaron {len(invalidos)} incidentes con estado inválido")
print(out.estado.value_counts().to_dict())
