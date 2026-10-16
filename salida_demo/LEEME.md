# Salida de ejemplo de la demostración con IA

Resultado de la demostración **«La IA como ayudante de analítica descriptiva»** de la Práctica 1: analítica descriptiva de la base completa `datos/guide_incidentes.csv` (80.000 incidentes de SOC, muestra del conjunto GUIDE de Microsoft) y contraste de cada variable con la etiqueta `grado`.

| Archivo | Qué es |
|---|---|
| `analisis.py` | El script que calcula todos los números del informe (semilla fija = 42). Lee el CSV en modo solo lectura. |
| `reporte.md` | El informe, con tablas y conclusiones. |
| `figuras/` | Las cinco figuras que usa el informe. |

**Cómo leerlo.** Es contenido **generado con IA**. Sirve para tener con qué comparar lo que produzca cada persona, no como respuesta correcta: la idea de la demostración es justamente revisar si una IA hizo bien el trabajo (los números salen de código que se puede leer, y dos métodos de impacto se comparan entre sí). Las cifras de las prácticas son de otro análisis (modelos con partición 70/30), así que no tienen por qué coincidir.

**Cómo reproducirlo.** Desde la raíz del repositorio: `python salida_demo/analisis.py`. Tarda unos 8 minutos y necesita pandas, scikit-learn, scipy y matplotlib. Vuelve a escribir `reporte.md` y las figuras.

Fuente de los datos y su licencia: [`datos/LEEME_datos.md`](../datos/LEEME_datos.md).
