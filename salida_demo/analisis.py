# -*- coding: utf-8 -*-
"""
Analítica descriptiva de los incidentes GUIDE (80.000 incidentes de SOC) frente a la etiqueta `grado`.

Uso (desde cualquier carpeta):
    python salida_demo/analisis.py

Lee   datos/guide_incidentes.csv   (solo lectura, nunca se modifica)
Crea  salida_demo/reporte.md  y  salida_demo/figuras/*.png

Todos los números del reporte se calculan aquí; el texto del reporte los inserta con f-strings.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, mutual_info_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------------------------
SEED = 42
AQUI = Path(__file__).resolve().parent
CSV = AQUI.parent / "datos" / "guide_incidentes.csv"
FIG = AQUI / "figuras"
FIG.mkdir(exist_ok=True)

ORDEN = ["amenaza real", "sin amenaza", "actividad esperada"]
ABREV = {"amenaza real": "AR", "sin amenaza": "SA", "actividad esperada": "AE"}
COLOR = {"amenaza real": "#D55E00", "sin amenaza": "#0072B2", "actividad esperada": "#8C8C8C"}
IDS = ["organizacion", "detector_principal"]  # variables con muchísimos niveles: se reportan aparte
OBJ = "grado"

BLOQUES = {
    "tamaño": ["n_alertas", "n_evidencias", "n_detectores", "n_tacticas", "n_paises"],
    "tiempo": ["minutos_duracion", "hora_inicio", "dia_semana"],
    "tácticas": None,  # tactica_*
    "entidades": None,  # entidad_*
    "señales automáticas": [
        "prop_veredicto_malicioso", "prop_veredicto_sin_amenazas", "prop_nivel_sospecha",
        "prop_con_tecnica_mitre", "prop_equipo_afectado",
    ],
}

# Significado de cada variable en lenguaje de analista de seguridad (sin números: estos salen del código)
SIGNIFICADO = {
    "n_alertas": "cuántas alertas se agruparon en el incidente (su 'tamaño' de investigación)",
    "n_evidencias": "cuántas pruebas (IP, usuarios, equipos, archivos…) cuelgan del incidente",
    "n_detectores": "cuántos detectores distintos dispararon alertas en el mismo incidente",
    "n_tacticas": "cuántas tácticas MITRE ATT&CK distintas aparecen (cuántas etapas del ataque se ven)",
    "n_paises": "en cuántos países aparecen las evidencias geolocalizadas",
    "minutos_duracion": "tiempo entre la primera y la última alerta del incidente",
    "hora_inicio": "hora del día en que arranca el incidente",
    "dia_semana": "día de la semana en que arranca el incidente (0 = lunes)",
    "prop_con_tecnica_mitre": "parte de las alertas que el detector ya asocia a una técnica MITRE concreta",
    "prop_nivel_sospecha": "parte de las alertas marcadas con nivel de sospecha por el producto",
    "prop_veredicto_malicioso": "parte de las evidencias que el sistema ya calificó como maliciosas",
    "prop_veredicto_sin_amenazas": "parte de las evidencias que el sistema ya calificó como sin amenaza",
    "prop_equipo_afectado": "parte de las evidencias cuyo equipo figura como afectado/impactado",
    "tactica_acceso_inicial": "peso de alertas de acceso inicial (el atacante intenta entrar)",
    "tactica_exfiltracion": "peso de alertas de exfiltración (salida de datos)",
    "tactica_actividad_sospechosa": "peso de alertas de actividad sospechosa genérica, sin táctica precisa",
    "tactica_comando_y_control": "peso de alertas de comando y control (comunicación con infraestructura atacante)",
    "tactica_impacto": "peso de alertas de impacto (cifrado, destrucción, interrupción)",
    "tactica_acceso_a_credenciales": "peso de alertas de robo de credenciales",
    "tactica_ejecucion": "peso de alertas de ejecución de código",
    "tactica_malware": "peso de alertas de malware",
    "tactica_descubrimiento": "peso de alertas de descubrimiento/reconocimiento interno",
    "tactica_persistencia": "peso de alertas de persistencia (el atacante intenta quedarse)",
    "tactica_evasion_de_defensas": "peso de alertas de evasión de defensas",
    "tactica_movimiento_lateral": "peso de alertas de movimiento lateral",
    "entidad_ip": "parte de las evidencias que son direcciones IP",
    "entidad_usuario": "parte de las evidencias que son cuentas de usuario",
    "entidad_mensaje_correo": "parte de las evidencias que son mensajes de correo",
    "entidad_equipo": "parte de las evidencias que son equipos/hosts",
    "entidad_inicio_sesion_nube": "parte de las evidencias que son inicios de sesión en la nube",
    "entidad_url": "parte de las evidencias que son URL",
    "entidad_archivo": "parte de las evidencias que son archivos",
    "entidad_buzon": "parte de las evidencias que son buzones de correo",
    "entidad_proceso": "parte de las evidencias que son procesos",
    "entidad_grupo_correos": "parte de las evidencias que son grupos de correos",
}

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------------------------
def md(df, index=True):
    """DataFrame (ya con texto formateado) -> tabla markdown."""
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]
    filas = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, f in d.iterrows():
        filas.append("| " + " | ".join(str(v) for v in f.tolist()) + " |")
    return "\n".join(filas)


def dec(x, d=3):
    """Número con coma decimal (convención en español)."""
    return f"{x:.{d}f}".replace(".", ",")


def num(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", ".")
    ax = abs(x)
    if ax != 0 and ax < 10 ** (-d):
        return f"{x:.1e}".replace(".", ",")
    if ax >= 1000:
        return f"{x:,.0f}".replace(",", ".")
    return dec(x, d)


def fnum(x):
    """Entero si el valor es entero (conteos), si no 3 decimales."""
    x = float(x)
    return num(int(x)) if x.is_integer() else num(x)


def pct(x, d=1):
    return f"{dec(100 * x, d)} %"


def pval(p):
    return "<1e-300" if p == 0 else f"{p:.1e}".replace(".", ",")


def cramer_v(tabla, corregido=False):
    """V de Cramér (y versión con corrección de sesgo de Bergsma) de una tabla de contingencia."""
    chi2 = stats.chi2_contingency(tabla, correction=False)[0]
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    if not corregido:
        return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    kc = k - (k - 1) ** 2 / (n - 1)
    rc = r - (r - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(kc - 1, rc - 1))))


def discretizar(s, max_niv=30, bins=10, masa=0.05):
    """Para información mutua. Pocos valores: tal cual. Muchos: cada valor con >= 5 % de los datos (p. ej. el 0,
    o n_alertas = 1) es su propio nivel y el resto se reparte en deciles."""
    if s.nunique() <= max_niv:
        return pd.factorize(s)[0]
    vc = s.value_counts(normalize=True)
    pesados = vc[vc >= masa].index
    cod = np.full(len(s), -1)
    for i, v in enumerate(pesados):
        cod[(s == v).to_numpy()] = i
    resto = ~s.isin(pesados)
    if resto.any():
        q = pd.qcut(s[resto], q=bins, duplicates="drop", labels=False)
        cod[resto.to_numpy()] = len(pesados) + q.to_numpy().astype(int)
    return pd.factorize(cod)[0]


def entropia(y):
    p = np.bincount(y) / len(y)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def nuevo_modelo():
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=300, max_leaf_nodes=31, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=15, class_weight="balanced", random_state=SEED,
    )


def f1m(y, p):
    return f1_score(y, p, average="macro")


def particion(X, y):
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)


def ajustar_y_permutar(X, y, repeticiones=10):
    """Gradient boosting + importancia por permutación sobre el 30 % de prueba (F1 macro)."""
    Xtr, Xte, ytr, yte = particion(X, y)
    m = nuevo_modelo().fit(Xtr, ytr)
    base = f1m(yte, m.predict(Xte))
    pi = permutation_importance(m, Xte, yte, scoring="f1_macro", n_repeats=repeticiones,
                                random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"media": pi.importances_mean, "sd": pi.importances_std}, index=X.columns)
    return m, base, f1m(ytr, m.predict(Xtr)), imp, (Xtr, Xte, ytr, yte)


def permutar_bloque(m, Xte, yte, cols, repeticiones=10):
    rng = np.random.default_rng(SEED)
    base = f1m(yte, m.predict(Xte))
    caidas = []
    for _ in range(repeticiones):
        Xp = Xte.copy()
        Xp[cols] = Xte[cols].to_numpy()[rng.permutation(len(Xte))]  # permuta el bloque completo junto
        caidas.append(base - f1m(yte, m.predict(Xp)))
    return float(np.mean(caidas)), float(np.std(caidas))


def firma_filas(X):
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


# ----------------------------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------------------------
df = pd.read_csv(CSV)
df[OBJ] = df[OBJ].astype(str)
N = len(df)
TAC = [c for c in df.columns if c.startswith("tactica_")]
ENT = [c for c in df.columns if c.startswith("entidad_")]
BLOQUES["tácticas"] = TAC
BLOQUES["entidades"] = ENT
NUM = [c for c in df.columns if c not in IDS + [OBJ]]  # 35 variables numéricas "normales"
CAT_BAJA = ["dia_semana", "hora_inicio"]  # discretas con pocos niveles: también como categóricas
PROP = [c for c in df.columns if c.startswith(("prop_", "tactica_", "entidad_"))]
y = df[OBJ]
log(f"Leídas {N} filas y {df.shape[1]} columnas de {CSV.name}")

# ==============================================================================================
# 1. Calidad y estructura
# ==============================================================================================
Q = {}
Q["tipos"] = df.dtypes.astype(str).value_counts().to_dict()
Q["nulos"] = int(df.isna().sum().sum())
Q["dup_total"] = int(df.duplicated().sum())
Q["dup_sin_grado"] = int(df.drop(columns=OBJ).duplicated().sum())
X_all = df.drop(columns=OBJ)
grp = df.groupby(list(X_all.columns))[OBJ].nunique()
Q["grupos_repetidos"] = int((df.groupby(list(X_all.columns)).size() > 1).sum())
Q["grupos_conflicto"] = int((grp > 1).sum())
tam_g = df.groupby(list(X_all.columns))[OBJ].transform("size")
nun_g = df.groupby(list(X_all.columns))[OBJ].transform("nunique")
Q["filas_en_conflicto"] = int(((nun_g > 1)).sum())
Q["dup_share_1alerta"] = float(df.loc[df.n_alertas == 1].duplicated().mean())
Q["dup_share_masalertas"] = float(df.loc[df.n_alertas > 1].duplicated().mean())
Q["max_copias"] = int(tam_g.max())
Q["prop_grado"] = y.value_counts(normalize=True).reindex(ORDEN)
Q["n_grado"] = y.value_counts().reindex(ORDEN)

suma_tac = df[TAC].sum(axis=1)
suma_ent = df[ENT].sum(axis=1)
reglas = {
    "Proporciones fuera de [0, 1]": int(((df[PROP] < 0) | (df[PROP] > 1)).any(axis=1).sum()),
    "hora_inicio fuera de 0–23": int((~df.hora_inicio.between(0, 23)).sum()),
    "dia_semana fuera de 0–6": int((~df.dia_semana.between(0, 6)).sum()),
    "minutos_duracion negativos": int((df.minutos_duracion < 0).sum()),
    "n_alertas < 1": int((df.n_alertas < 1).sum()),
    "n_detectores > n_alertas": int((df.n_detectores > df.n_alertas).sum()),
    "n_evidencias < n_alertas": int((df.n_evidencias < df.n_alertas).sum()),
    "n_tacticas > n_alertas": int((df.n_tacticas > df.n_alertas).sum()),
    "Suma de tactica_* > 1": int((suma_tac > 1 + 1e-3).sum()),
    "Suma de tactica_* = 0 (ninguna táctica registrada)": int((suma_tac < 1e-9).sum()),
    "Suma de entidad_* > 1": int((suma_ent > 1 + 1e-3).sum()),
    "Suma de entidad_* < 0,99 (tipos de entidad no incluidos)": int((suma_ent < 0.99).sum()),
    "n_alertas = 1 pero minutos_duracion > 0": int(((df.n_alertas == 1) & (df.minutos_duracion > 0)).sum()),
    "detector_principal = -1 (detector poco común)": int((df.detector_principal == -1).sum()),
}
Q["reglas"] = reglas
log("1. Calidad lista")

# ==============================================================================================
# 2. Descriptivo de la base completa
# ==============================================================================================
D = pd.DataFrame(index=NUM)
D["media"] = df[NUM].mean()
D["mediana"] = df[NUM].median()
D["p90"] = df[NUM].quantile(0.90)
D["p99"] = df[NUM].quantile(0.99)
D["max"] = df[NUM].max()
D["pct_ceros"] = (df[NUM] == 0).mean()
D["asimetria"] = df[NUM].skew()
D["max_sobre_p99"] = np.where(D.p99 > 0, D["max"] / D.p99.replace(0, np.nan), np.nan)
D["asimetrica"] = D.asimetria.abs() > 2
D["extremos"] = (D.max_sobre_p99 >= 5) | ((D.p99 == 0) & (D["max"] > 0) & (~D.index.isin(PROP)))

CATD = []
for c in IDS + CAT_BAJA + [OBJ]:
    vc = df[c].value_counts()
    top = ", ".join(f"{k} ({num(int(v))}; {pct(v / N)})" for k, v in vc.head(4).items())
    CATD.append({"variable": c, "niveles": num(int(vc.size)), "más frecuentes (n; %)": top})
CATD = pd.DataFrame(CATD).set_index("variable")
log("2. Descriptivo listo")

# ==============================================================================================
# 3. Contraste con grado
# ==============================================================================================
MEDIA = df.groupby(OBJ)[NUM].mean().reindex(ORDEN).T
MEDIANA = df.groupby(OBJ)[NUM].median().reindex(ORDEN).T
PCT_NOCERO = df.groupby(OBJ)[NUM].agg(lambda s: (s != 0).mean()).reindex(ORDEN).T

# ==============================================================================================
# 4a. Asociación una a una: KW + epsilon², información mutua; V de Cramér
# ==============================================================================================
y_cod = pd.factorize(y)[0]
H_y = entropia(y_cod)
rng = np.random.default_rng(SEED)
N_SHUF = 20
shuf = [rng.permutation(y_cod) for _ in range(N_SHUF)]

filas = []
for c in NUM:
    grupos = [df.loc[y == g, c].to_numpy() for g in ORDEN]
    try:
        H, p = stats.kruskal(*grupos)
    except ValueError:  # variable constante
        H, p = np.nan, np.nan
    eps2 = H / (N - 1) if H == H else np.nan
    x = discretizar(df[c])
    mi = mutual_info_score(x, y_cod)
    mi_ruido = float(np.mean([mutual_info_score(x, s) for s in shuf]))
    filas.append({"variable": c, "H": H, "p": p, "eps2": eps2, "MI": mi, "MI_ruido": mi_ruido,
                  "niveles_MI": len(np.unique(x))})
A = pd.DataFrame(filas).set_index("variable")
A["MI_corr"] = (A.MI - A.MI_ruido).clip(lower=0)
A["MI_pct"] = A.MI_corr / H_y  # fracción de la incertidumbre de `grado` que reduce


def tamano_efecto(e):
    return "insignificante" if e < 0.01 else "pequeño" if e < 0.06 else "mediano" if e < 0.14 else "grande"


A["efecto"] = A.eps2.map(tamano_efecto)

# V de Cramér: dia_semana y hora_inicio como categóricas
V_CAT = {}
for c in CAT_BAJA + IDS:
    tabla = pd.crosstab(df[c], y)
    v = cramer_v(tabla)
    vc = cramer_v(tabla, corregido=True)
    vr = np.mean([cramer_v(pd.crosstab(df[c], pd.Series(s, index=df.index))) for s in shuf[:10]])
    vcr = np.mean([cramer_v(pd.crosstab(df[c], pd.Series(s, index=df.index)), True) for s in shuf[:10]])
    x = pd.factorize(df[c])[0]
    mi = mutual_info_score(x, y_cod)
    mi_r = float(np.mean([mutual_info_score(x, s) for s in shuf]))
    V_CAT[c] = {"niveles": tabla.shape[0], "V": v, "V_corr": vc, "V_azar": vr, "V_corr_azar": vcr,
                "MI_pct": mi / H_y, "MI_azar_pct": mi_r / H_y,
                "chi2_p": stats.chi2_contingency(tabla, correction=False)[1]}
V_CAT = pd.DataFrame(V_CAT).T
log("4a. Asociación una a una lista")

# ==============================================================================================
# 4b. Gradient boosting + importancia por permutación (partición 70/30 estratificada, semilla fija)
# ==============================================================================================
X = df[NUM].copy()
modelo, F1_TEST, F1_TRAIN, PI, (Xtr, Xte, ytr, yte) = ajustar_y_permutar(X, y, repeticiones=10)
F1_AZAR = f1m(yte, np.repeat(ytr.value_counts().idxmax(), len(yte)))
F1_CLASE = pd.Series(f1_score(yte, modelo.predict(Xte), average=None, labels=ORDEN), index=ORDEN)
ACC_TEST = float((modelo.predict(Xte) == yte).mean())
log(f"4b. Modelo A (sin IDs): F1 macro prueba = {F1_TEST:.3f}")

# Bloques de variables: importancia por permutación del bloque entero + exclusión del bloque
BL = []
for nombre, cols in BLOQUES.items():
    caida, sd = permutar_bloque(modelo, Xte, yte, cols)
    m2 = nuevo_modelo().fit(Xtr.drop(columns=cols), ytr)
    f1_sin = f1m(yte, m2.predict(Xte.drop(columns=cols)))
    BL.append({"bloque": nombre, "n_vars": len(cols), "caida_perm": caida, "sd": sd, "F1_sin_bloque": f1_sin,
               "caida_exclusion": F1_TEST - f1_sin})
BL = pd.DataFrame(BL).set_index("bloque")
# Solo con el bloque (para ver cuánto aporta cada bloque por sí solo)
solo = {}
for nombre, cols in BLOQUES.items():
    m3 = nuevo_modelo().fit(Xtr[cols], ytr)
    solo[nombre] = f1m(yte, m3.predict(Xte[cols]))
BL["F1_solo_bloque"] = pd.Series(solo)
log("4b. Bloques listos")

# Sensibilidad a duplicados exactos: los mismos pasos sobre la base sin filas repetidas
df_u = df.drop_duplicates().reset_index(drop=True)
mu, F1_TEST_U, F1_TRAIN_U, PI_U, (Xtr_u, Xte_u, ytr_u, yte_u) = ajustar_y_permutar(df_u[NUM], df_u[OBJ], 10)
firmas_tr = set(firma_filas(Xtr.assign(_o=df.loc[Xtr.index, "organizacion"].values)))
gem_prueba = float(np.isin(firma_filas(Xte.assign(_o=df.loc[Xte.index, "organizacion"].values)),
                           list(firmas_tr)).mean())
# Porcentaje de filas de la partición de prueba que tienen un gemelo exacto (mismas 36 columnas) en entrenamiento
log(f"4b. Sin duplicados: {len(df_u)} filas; F1 macro prueba = {F1_TEST_U:.3f}")

# ==============================================================================================
# 5. Variables con muchos niveles: organizacion y detector_principal
# ==============================================================================================
ORG = {}
org_n = df.organizacion.value_counts()
ORG["niveles"] = int(org_n.size)
ORG["mediana_inc"] = float(org_n.median())
ORG["top10_share"] = float(org_n.head(10).sum() / N)
ORG["con_menos_de_10"] = float((org_n < 10).mean())
ORG["inc_en_orgs_menos_10"] = float(org_n[org_n < 10].sum() / N)
ORG["con_unico_grado"] = float((pd.crosstab(df.organizacion, y).gt(0).sum(axis=1) == 1).mean())
ct = pd.crosstab(df.organizacion, y)
pureza = float(ct.max(axis=1).sum() / N)
pureza_azar = float(np.mean([pd.crosstab(df.organizacion, pd.Series(s, index=df.index)).max(axis=1).sum() / N
                             for s in shuf[:10]]))
ORG["pureza"], ORG["pureza_azar"] = pureza, pureza_azar
grande = ct[ct.sum(axis=1) >= 100]
pr_ar = (grande["amenaza real"] / grande.sum(axis=1))
ORG["n_grandes"] = int(len(grande))
ORG["ar_q"] = pr_ar.quantile([0, .1, .5, .9, 1]).to_dict()
top_org = ct.loc[org_n.head(8).index].copy()
top_org_pct = top_org.div(top_org.sum(axis=1), axis=0).reindex(columns=ORDEN)
top_org_pct.insert(0, "incidentes", top_org.sum(axis=1))

det_n = df.detector_principal.value_counts()
ctd = pd.crosstab(df.detector_principal, y)
top_det = ctd.loc[det_n.head(8).index].copy()
top_det_pct = top_det.div(top_det.sum(axis=1), axis=0).reindex(columns=ORDEN)
top_det_pct.insert(0, "incidentes", top_det.sum(axis=1))
DET = {"niveles": int(det_n.size), "menos_comun": int((df.detector_principal == -1).sum()),
       "pureza": float(ctd.max(axis=1).sum() / N),
       "pureza_azar": float(np.mean([pd.crosstab(df.detector_principal, pd.Series(s, index=df.index)).max(axis=1).sum() / N
                                     for s in shuf[:10]]))}


def mayoria_por_grupo(col, tr_idx, te_idx):
    """Predice en `te_idx` la etiqueta más frecuente del grupo en `tr_idx` (si el grupo es nuevo: la global)."""
    tabla = df.loc[tr_idx].groupby(col)[OBJ].agg(lambda s: s.value_counts().idxmax())
    glob = df.loc[tr_idx, OBJ].value_counts().idxmax()
    return df.loc[te_idx, col].map(tabla).fillna(glob)


# (i) misma partición 70/30 de antes
F1_ORG_MAY = f1m(yte, mayoria_por_grupo("organizacion", Xtr.index, Xte.index))
F1_DET_MAY = f1m(yte, mayoria_por_grupo("detector_principal", Xtr.index, Xte.index))
# (ii) modelo con las dos variables-ID añadidas (organizacion como número; detector como categórica)
X_id = df[NUM + IDS].copy()
cat_mask = [c == "detector_principal" for c in X_id.columns]
X_id["detector_principal"] = pd.factorize(X_id["detector_principal"], sort=True)[0]
Xtr_i, Xte_i, ytr_i, yte_i = particion(X_id, y)
m_id = HistGradientBoostingClassifier(
    learning_rate=0.1, max_iter=300, max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=15, class_weight="balanced", random_state=SEED, categorical_features=cat_mask,
).fit(Xtr_i, ytr_i)
F1_ID = f1m(yte_i, m_id.predict(Xte_i))
pi_id = permutation_importance(m_id, Xte_i, yte_i, scoring="f1_macro", n_repeats=10, random_state=SEED, n_jobs=1)
PI_ID = pd.DataFrame({"media": pi_id.importances_mean, "sd": pi_id.importances_std}, index=X_id.columns)
# (iii) organizaciones nuevas: el modelo sin IDs entrenado con el 70 % de las organizaciones y probado en el 30 % restante
#       (5 particiones distintas, porque con organizaciones de tamaños tan desiguales una sola puede engañar)
gss = GroupShuffleSplit(n_splits=5, test_size=0.30, random_state=SEED)
F1_NUEVAS, F1_AZAR_NUEVAS_L, N_NUEVAS = [], [], []
for itr, ite in gss.split(X, y, groups=df.organizacion):
    m_g = nuevo_modelo().fit(X.iloc[itr], y.iloc[itr])
    F1_NUEVAS.append(f1m(y.iloc[ite], m_g.predict(X.iloc[ite])))
    F1_AZAR_NUEVAS_L.append(f1m(y.iloc[ite], np.repeat(y.iloc[itr].value_counts().idxmax(), len(ite))))
    N_NUEVAS.append(len(ite))
F1_ORG_NUEVAS = float(np.mean(F1_NUEVAS))
F1_AZAR_NUEVAS = float(np.mean(F1_AZAR_NUEVAS_L))
log(f"5. IDs: F1 org-mayoría {F1_ORG_MAY:.3f}; modelo con IDs {F1_ID:.3f}; org nuevas {F1_ORG_NUEVAS:.3f} {np.round(F1_NUEVAS,3)}")

# Señales automáticas: ¿qué etiqueta tienen los incidentes donde el sistema ya había emitido un veredicto/sospecha?
VER = {}
for c, txt in [("prop_veredicto_malicioso", "con algún veredicto «malicioso»"),
               ("prop_veredicto_sin_amenazas", "con algún veredicto «sin amenazas»"),
               ("prop_nivel_sospecha", "con alertas de nivel de sospecha")]:
    m = df[c] > 0
    VER[txt] = {"incidentes": int(m.sum()), **{ABREV[g]: float((y[m] == g).mean()) for g in ORDEN}}
VER["(para comparar) todos los incidentes"] = {"incidentes": N, **{ABREV[g]: float((y == g).mean()) for g in ORDEN}}
VER = pd.DataFrame(VER).T

# Valores "con huella": n_evidencias exacto vs grado, y qué detector está detrás del caso n_evidencias = 4
top_ne = df.n_evidencias.value_counts().head(8).index
NE = pd.crosstab(df.loc[df.n_evidencias.isin(top_ne), "n_evidencias"], y, normalize="index").reindex(columns=ORDEN)
NE.insert(0, "incidentes", df.n_evidencias.value_counts()[NE.index])
NE = NE.sort_index()
s4 = df[df.n_evidencias == 4]
NE4_det0 = float((s4.detector_principal == 0).mean())
NE4_orgs = int(s4.organizacion.nunique())
NE4_1alerta = float((s4.n_alertas == 1).mean())
# n_paises vs inicios de sesión en la nube
CORR_PAIS_NUBE = float(df[["n_paises", "entidad_inicio_sesion_nube"]].corr(method="spearman").iloc[0, 1])
PAIS = pd.crosstab(df.n_paises.clip(upper=4), y, normalize="index").reindex(columns=ORDEN)
PAIS.insert(0, "incidentes", df.n_paises.clip(upper=4).value_counts().sort_index())
PAIS_MASDE1_Y_NUBE = float(((df.n_paises > 1) & (df.entidad_inicio_sesion_nube > 0)).sum() / (df.n_paises > 1).sum())

# ==============================================================================================
# 6. Ranking final
# ==============================================================================================
R = A[["eps2", "efecto", "MI_pct"]].copy()
R["PI"] = PI["media"]
R["PI_sd"] = PI["sd"]
R["PI_sin_dup"] = PI_U["media"]
R["r_eps"] = R.eps2.rank(ascending=False)
R["r_MI"] = R.MI_pct.rank(ascending=False)
R["r_asoc"] = (R.r_eps + R.r_MI) / 2
R["r_PI"] = R.PI.rank(ascending=False)
R["r_PI_sin_dup"] = R.PI_sin_dup.rank(ascending=False)
R["r_medio"] = (R.r_asoc + R.r_PI) / 2
R = R.sort_values(["r_medio", "r_PI"])
R["puesto"] = np.arange(1, len(R) + 1)
TOP = R.head(10).copy()
rho_ranks = stats.spearmanr(R.r_asoc, R.r_PI)[0]
rho_eps_mi = stats.spearmanr(R.r_eps, R.r_MI)[0]
rho_pi_dup = stats.spearmanr(R.r_PI, R.r_PI_sin_dup)[0]
top10_asoc = set(R.sort_values("r_asoc").head(10).index)
top10_pi = set(R.sort_values("r_PI").head(10).index)
top10_pi_u = set(R.sort_values("r_PI_sin_dup").head(10).index)
top10_union_overlap = top10_asoc & top10_pi
R["dif_rango"] = (R.r_asoc - R.r_PI)

corr = df[NUM].corr(method="spearman")
pares = []
for i, a in enumerate(NUM):
    for b in NUM[i + 1:]:
        if abs(corr.loc[a, b]) >= 0.7:
            pares.append((a, b, corr.loc[a, b]))
pares.sort(key=lambda t: -abs(t[2]))
log("6. Ranking listo")

# ==============================================================================================
# Figuras (5)
# ==============================================================================================
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "axes.titlesize": 11, "axes.titleweight": "bold"})

# Fig 1: grado y asimetría
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
b = ax[0].bar([ABREV[g] for g in ORDEN], [Q["prop_grado"][g] * 100 for g in ORDEN], color=[COLOR[g] for g in ORDEN])
for r, g in zip(b, ORDEN):
    ax[0].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.8, f"{pct(Q['prop_grado'][g])}\n({num(int(Q['n_grado'][g]))})",
               ha="center", va="bottom", fontsize=9)
ax[0].set_ylim(0, 62)
ax[0].set_ylabel("% de incidentes")
ax[0].set_title("Proporción de cada grado")
ax[0].set_xlabel("AR = amenaza real · SA = sin amenaza · AE = actividad esperada", fontsize=8)
sk = D.asimetria.sort_values(ascending=False).head(12)[::-1]
ax[1].barh(sk.index, sk.values, color="#555555")
for i, (k, v) in enumerate(sk.items()):
    ax[1].text(v + 1, i, f"{v:.0f}", va="center", fontsize=8)
ax[1].set_title("Variables más asimétricas (coef. de asimetría)")
ax[1].set_xlabel("asimetría (0 = simétrica; > 2 = muy sesgada a la derecha)")
fig.tight_layout()
fig.savefig(FIG / "fig1_grado_y_asimetria.png")
plt.close(fig)

# Fig 2: contraste de las 6 más llamativas (mayor tamaño de efecto)
top6 = A.sort_values("eps2", ascending=False).head(6).index.tolist()
fig, axs = plt.subplots(2, 3, figsize=(12, 6.2))
for a, c in zip(axs.ravel(), top6):
    vals = [MEDIA.loc[c, g] for g in ORDEN]
    bb = a.bar([ABREV[g] for g in ORDEN], vals, color=[COLOR[g] for g in ORDEN])
    top_y = max(vals) if max(vals) > 0 else 1
    for r, v in zip(bb, vals):
        a.text(r.get_x() + r.get_width() / 2, v + top_y * 0.02, dec(v, 2) if v < 100 else f"{v:.0f}", ha="center", fontsize=8)
    a.set_ylim(0, top_y * 1.18)
    a.set_title(f"{c}\n(ε² = {dec(A.loc[c, 'eps2'])})", fontsize=9)
fig.suptitle("Media por grado de las 6 variables con mayor tamaño de efecto (AR = amenaza real · SA = sin amenaza · AE = actividad esperada)",
             fontsize=9.5, y=0.995)
fig.tight_layout()
fig.savefig(FIG / "fig2_contraste_por_grado.png")
plt.close(fig)

# Fig 3: asociación una a una (ε² y MI)
orden3 = A.sort_values("eps2")
fig, ax = plt.subplots(1, 2, figsize=(12, 8.2), sharey=True)
ax[0].barh(orden3.index, orden3.eps2, color="#444444")
ax[0].axvline(0.01, color="#999", ls=":", lw=1)
ax[0].axvline(0.06, color="#999", ls="--", lw=1)
ax[0].set_title("Kruskal-Wallis: tamaño del efecto ε²\n(líneas: 0,01 pequeño · 0,06 mediano)")
ax[0].tick_params(axis="y", labelsize=8)
ax[1].barh(orden3.index, orden3.MI_pct * 100, color="#0072B2")
ax[1].set_title("Información mutua con grado\n(% de la incertidumbre de grado, sin ruido)")
fig.tight_layout()
fig.savefig(FIG / "fig3_asociacion_una_a_una.png")
plt.close(fig)

# Fig 4: permutación + comparación de rangos
orden4 = PI.sort_values("media").tail(15)
fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.35})
ax[0].barh(orden4.index, orden4.media, xerr=orden4.sd, color="#D55E00", ecolor="#333", capsize=2)
ax[0].set_title("Importancia por permutación (15 mayores)\ncaída del F1 macro (media ± sd, 10 repeticiones)", fontsize=10)
ax[0].tick_params(axis="y", labelsize=8)
ax[1].scatter(R.r_asoc, R.r_PI, c="#0072B2", s=28)
lim = len(R) + 1
ax[1].plot([0, lim], [0, lim], color="#999", ls=":")
for v in TOP.index:
    ax[1].annotate(v, (R.loc[v, "r_asoc"], R.loc[v, "r_PI"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[1].set_xlim(0, lim)
ax[1].set_ylim(0, lim)
ax[1].invert_xaxis()
ax[1].invert_yaxis()
ax[1].set_xlabel("puesto por asociación una a una (1 = más asociada)")
ax[1].set_ylabel("puesto por permutación (1 = más importante)")
ax[1].set_title(f"Los dos métodos comparados\n(Spearman = {dec(rho_ranks, 2)})", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig4_permutacion_y_comparacion.png")
plt.close(fig)

# Fig 5: variables con muchos niveles
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
etq = ["dia_semana\n(7)", "hora_inicio\n(24)", "detector_principal\n(" + str(DET["niveles"]) + ")",
       "organizacion\n(" + num(ORG["niveles"]) + ")"]
claves = ["dia_semana", "hora_inicio", "detector_principal", "organizacion"]
xx = np.arange(len(claves))
ax[0].bar(xx - 0.2, V_CAT.loc[claves, "V"].astype(float), 0.4, label="V de Cramér observada", color="#0072B2")
ax[0].bar(xx + 0.2, V_CAT.loc[claves, "V_azar"].astype(float), 0.4, label="V con grado barajado al azar", color="#BBBBBB")
ax[0].set_xticks(xx)
ax[0].set_xticklabels(etq, fontsize=8)
ax[0].set_title("La V de Cramér se infla con los niveles\n(la barra gris es lo que sale SIN relación real)")
ax[0].legend(fontsize=8, frameon=False)
nombres = ["Siempre la clase más frecuente", "Mayoría por detector", "Mayoría por organización",
           "Modelo sin IDs (mismas organizaciones)", "Modelo sin IDs (organizaciones NUEVAS)", "Modelo con IDs"]
vals = [F1_AZAR, F1_DET_MAY, F1_ORG_MAY, F1_TEST, F1_ORG_NUEVAS, F1_ID]
cols = ["#BBBBBB", "#0072B2", "#0072B2", "#444444", "#D55E00", "#444444"]
bb = ax[1].barh(range(6), vals, color=cols)
for r, v in zip(bb, vals):
    ax[1].text(v + 0.01, r.get_y() + r.get_height() / 2, dec(v, 2), va="center", fontsize=9)
ax[1].set_yticks(range(6))
ax[1].set_yticklabels(nombres, fontsize=8.5)
ax[1].invert_yaxis()
ax[1].set_xlim(0, max(vals) * 1.12)
ax[1].set_xlabel("F1 promedio (macro) en datos de prueba")
ax[1].set_title("¿Cuánto predice la etiqueta cada fuente?")
fig.tight_layout()
fig.savefig(FIG / "fig5_variables_con_muchos_niveles.png")
plt.close(fig)
log("Figuras listas")


# ==============================================================================================
# Reporte (reporte.md). Todo número sale de las variables calculadas arriba.
# ==============================================================================================
def M(v, g):  # media de la variable v en el grado g (texto)
    return dec(MEDIA.loc[v, g], 3)


def tres(v, f=None):  # "AR a · SA b · AE c"
    f = f or (lambda x: dec(x, 3))
    return " · ".join(f"{ABREV[g]} {f(MEDIA.loc[v, g])}" for g in ORDEN)


def grado_max(v):
    return MEDIA.loc[v].idxmax()


def grado_min(v):
    return MEDIA.loc[v].idxmin()


LECTURA = {
    "entidad_inicio_sesion_nube": lambda: f"Los incidentes con inicios de sesión en la nube se concentran en «sin amenaza»; casi no aparecen en «actividad esperada» (media por grado: {tres('entidad_inicio_sesion_nube')}).",
    "prop_equipo_afectado": lambda: f"«Actividad esperada» tiene la media más alta y «sin amenaza» la más baja ({tres('prop_equipo_afectado')}); además hay composiciones exactas que concentran una sola etiqueta (relación no monótona).",
    "entidad_ip": lambda: f"Las IP pesan más en «amenaza real» y «sin amenaza» que en «actividad esperada» ({tres('entidad_ip')}).",
    "entidad_url": lambda: f"Las URL pesan menos en «sin amenaza» que en las otras dos etiquetas ({tres('entidad_url')}).",
    "tactica_comando_y_control": lambda: f"Contraintuitivo: el comando y control pesa más en «actividad esperada» que en «amenaza real» ({tres('tactica_comando_y_control')}); hipótesis por verificar: detectores ruidosos o actividad autorizada, no que C2 sea inocuo.",
    "entidad_equipo": lambda: f"Los equipos como evidencia pesan más en «actividad esperada» que en las otras dos ({tres('entidad_equipo')}).",
    "entidad_usuario": lambda: f"El promedio casi no cambia entre etiquetas ({tres('entidad_usuario')}), pero la proporción exacta sí: hay composiciones típicas de cada etiqueta (por eso pesa la información mutua y no Kruskal-Wallis).",
    "n_evidencias": lambda: f"«Amenaza real» tiene la media más alta por colas largas ({tres('n_evidencias', lambda x: dec(x, 1))}), pero las medianas van al revés ({MEDIANA.loc['n_evidencias', 'amenaza real']:.0f}/{MEDIANA.loc['n_evidencias', 'sin amenaza']:.0f}/{MEDIANA.loc['n_evidencias', 'actividad esperada']:.0f}); y ciertos tamaños exactos (p. ej. 4 pruebas) los domina una etiqueta: hipótesis: son huellas de plantillas de alerta, no gravedad.",
    "prop_nivel_sospecha": lambda: f"Más frecuente en «actividad esperada» y menos en «sin amenaza» ({tres('prop_nivel_sospecha')}): el nivel de sospecha del producto no pone a «amenaza real» por encima de las otras dos.",
    "tactica_acceso_inicial": lambda: f"Pesa más en «sin amenaza» y menos en «actividad esperada» ({tres('tactica_acceso_inicial')}).",
    "n_paises": lambda: f"Casi equivale a «hay inicios de sesión en la nube» (Spearman {dec(CORR_PAIS_NUBE, 2)}); por eso su importancia por permutación se diluye.",
    "entidad_buzon": lambda: f"Buzones: pesan más en «amenaza real» ({tres('entidad_buzon')}).",
    "entidad_archivo": lambda: f"Archivos: pesan menos en «amenaza real» ({tres('entidad_archivo')}).",
}


def frase(v):
    base = SIGNIFICADO[v][0].upper() + SIGNIFICADO[v][1:]
    lec = LECTURA[v]() if v in LECTURA else f"Media por grado: {tres(v)}."
    return f"{base} — {lec}"


def senal(v):
    t = []
    if D.loc[v, "asimetrica"]:
        t.append("asimétrica")
    if D.loc[v, "extremos"]:
        t.append("valores extremos")
    if D.loc[v, "pct_ceros"] >= 0.8:
        t.append("casi todo ceros")
    return ", ".join(t) if t else "—"


L = []
w = L.append
w("# Qué variables se relacionan con la etiqueta `grado` en 80.000 incidentes de SOC (GUIDE)\n")
w(f"**Datos:** `datos/guide_incidentes.csv` ({num(N)} incidentes, muestra del conjunto GUIDE de Microsoft; el CSV no se modifica). "
  f"**Alcance:** analítica descriptiva de la base completa, no un modelo para producción. **Reproducir:** `python salida_demo/analisis.py` "
  f"(semilla fija = {SEED}; tarda unos 8 minutos por los gradient boosting). Todas las cifras de este reporte las produce ese script.\n")
w("**Convenciones.** AR = amenaza real · SA = sin amenaza · AE = actividad esperada. ε² = tamaño del efecto de Kruskal-Wallis "
  "(H/(n−1); convención habitual: <0,01 insignificante, <0,06 pequeño, <0,14 mediano, ≥0,14 grande). "
  "MI % = información mutua con `grado` descontando el ruido que da el azar, expresada como % de la incertidumbre de `grado`. "
  "PI = importancia por permutación (caída del F1 macro en el 30 % de prueba al barajar la variable).\n")

# ---- 1 ----
w("## 1. Calidad y estructura\n")
tab = pd.DataFrame({"valor": [
    num(N), num(df.shape[1]),
    ", ".join(f"{k}: {v}" for k, v in Q["tipos"].items()),
    num(Q["nulos"]),
    f"{num(Q['dup_total'])} ({pct(Q['dup_total'] / N)})",
    f"{num(Q['dup_sin_grado'])}",
    f"{num(Q['grupos_conflicto'])} grupos / {num(Q['filas_en_conflicto'])} filas",
]}, index=["Filas", "Columnas", "Tipos de columna", "Valores nulos (total)", "Filas duplicadas exactas (las 38 columnas, contando cada copia sobrante)",
           "Filas repetidas ignorando `grado`", "Mismas características pero etiquetas distintas"])
w(md(tab.rename_axis("Indicador")))
w("\n**Proporción de cada grado**\n")
tg = pd.DataFrame({"incidentes": [num(int(Q["n_grado"][g])) for g in ORDEN], "%": [pct(Q["prop_grado"][g]) for g in ORDEN]}, index=ORDEN)
w(md(tg.rename_axis("grado")))
w(f"\nLa clase mayoritaria es «actividad esperada». Un clasificador que siempre diga eso obtiene F1 macro = {dec(F1_AZAR)} en la partición de prueba (el piso para comparar).\n")
w("\n**Reglas de coherencia (cuántas filas las violan o cumplen)**\n")
tr = pd.DataFrame({"filas": [num(v) for v in Q["reglas"].values()]}, index=list(Q["reglas"].keys()))
w(md(tr.rename_axis("Regla")))
w(f"""
Lectura:

- **No hay nulos ni valores imposibles** (proporciones fuera de [0, 1], horas o días fuera de rango, duraciones negativas): las primeras ocho reglas dan 0.
- **Duplicados.** {num(Q['dup_total'])} filas ({pct(Q['dup_total'] / N)}) son copias exactas de otra. Todas son incidentes de **una sola alerta** (entre los incidentes de 1 alerta, {pct(Q['dup_share_1alerta'])} están repetidos; entre los de más de una alerta, {pct(Q['dup_share_masalertas'])}), y la copia más repetida aparece {num(Q['max_copias'])} veces. No las eliminé en el análisis principal (el pedido es la base completa), pero **sí importan**: (a) en la partición 70/30, {pct(gem_prueba)} de las filas de prueba tienen un gemelo exacto en entrenamiento, lo que infla el F1; (b) hay {num(Q['grupos_conflicto'])} grupos de filas idénticas con etiquetas distintas ({num(Q['filas_en_conflicto'])} filas): con estas variables dos incidentes indistinguibles recibieron etiquetas distintas: o difieren en algo que el CSV no recoge, o el criterio del analista no fue consistente. En ambos casos hay un techo a lo que estas variables pueden explicar. En la sección 4b repito el análisis sin duplicados.
- **Valores extremos reales, no errores:** hay incidentes con hasta {num(int(D.loc['n_alertas', 'max']))} alertas, {num(int(D.loc['n_evidencias', 'max']))} evidencias y {num(int(D.loc['n_paises', 'max']))} países (detalle en la sección 2). No puedo saber con estos datos si son campañas masivas o artefactos de cómo se agrupan las alertas; los dejé tal cual porque las medidas que uso (rangos, deciles, árboles) no se distorsionan por ellos.
- **`tactica_*` suma 0 en {num(Q['reglas']['Suma de tactica_* = 0 (ninguna táctica registrada)'])} incidentes** (ninguna alerta traía táctica MITRE). **`entidad_*` suma menos de 0,99 en {num(Q['reglas']['Suma de entidad_* < 0,99 (tipos de entidad no incluidos)'])} incidentes**: el CSV solo conserva 10 tipos de entidad, el resto no está.
- `detector_principal = -1` ({num(Q['reglas']['detector_principal = -1 (detector poco común)'])} incidentes) no es «faltante», sino «detector fuera de los 200 más comunes».
""")

# ---- 2 ----
w("## 2. Descriptivo de la base completa\n")
w("**Variables numéricas** (35). `p99` y `máx/p99` sirven para ver extremos; asimétrica = |coef. de asimetría| > 2; valores extremos = máx ≥ 5 × p99 "
  "(umbrales míos, no una norma). En las variables de proporción (0–1) «asimétrica» solo significa «casi todo ceros».\n")
td = pd.DataFrame({
    "media": D["media"].map(fnum), "mediana": D["mediana"].map(fnum),
    "p90": D["p90"].map(fnum), "p99": D["p99"].map(fnum), "máx": D["max"].map(fnum), "% ceros": D["pct_ceros"].map(pct),
    "asimetría": D["asimetria"].map(lambda x: dec(float(x), 1)), "señal": [senal(v) for v in D.index],
})
w(md(td.rename_axis("variable")))
tamano_ext = [v for v in D.index if D.loc[v, "extremos"]]
w(f"""
Lectura:

- **Valores extremos:** {', '.join(f'`{v}`' for v in tamano_ext)}. Son las de tamaño/duración: la mediana es {fnum(D.loc['n_alertas','mediana'])} alerta y {fnum(D.loc['n_evidencias','mediana'])} evidencias, pero el máximo llega a {num(int(D.loc['n_alertas','max']))} y {num(int(D.loc['n_evidencias','max']))}. La media engaña: `n_evidencias` tiene media {dec(D.loc['n_evidencias','media'],1)} y p90 {fnum(D.loc['n_evidencias','p90'])}.
- **`minutos_duracion` es casi siempre 0** ({pct(D.loc['minutos_duracion','pct_ceros'])}): la mayoría de incidentes son de una sola alerta, sin duración medible. Su p90 es {dec(D.loc['minutos_duracion','p90'],1)} minutos y el máximo {fnum(D.loc['minutos_duracion','max'])} (≈ {dec(D.loc['minutos_duracion','max']/1440,0)} días).
- **Casi todas las `tactica_*` son raras:** las de persistencia, evasión de defensas y movimiento lateral valen 0 en más del {dec(100*min(D.loc[['tactica_persistencia','tactica_evasion_de_defensas','tactica_movimiento_lateral'],'pct_ceros']),1)} % de los incidentes. Con tan pocos casos, cualquier conclusión sobre ellas es frágil.
- **Casi simétricas:** `hora_inicio` y `dia_semana` (asimetría {dec(D.loc['hora_inicio','asimetria'],2)} y {dec(D.loc['dia_semana','asimetria'],2)}); el día {int(df.dia_semana.value_counts().idxmax())} es el más frecuente.
""")
w("**Variables categóricas / de identificación**\n")
w(md(CATD))
w("\n![Proporción de cada grado y variables más asimétricas](figuras/fig1_grado_y_asimetria.png)\n")

# ---- 3 ----
w("## 3. Contraste de cada variable con `grado`\n")
w("Media y mediana por etiqueta, ordenadas por tamaño del efecto (ε², sección 4). Con tantos ceros la mediana casi siempre es 0 (no discrimina); "
  "por eso muestro también la media. Los gráficos solo cubren las 6 variables más llamativas.\n")
cont = pd.DataFrame(index=A.sort_values("eps2", ascending=False).index)
for g in ORDEN:
    cont[f"media {ABREV[g]}"] = MEDIA.loc[cont.index, g].map(lambda x: num(float(x)))
for g in ORDEN:
    cont[f"mediana {ABREV[g]}"] = MEDIANA.loc[cont.index, g].map(fnum)
cont["ε²"] = A.loc[cont.index, "eps2"].map(lambda x: dec(x, 3))
w(md(cont.rename_axis("variable")))
n_med_dist = int(((MEDIANA.nunique(axis=1)) > 1).sum())
w(f"""
Lectura:

- Solo {n_med_dist} de las 35 variables tienen medianas distintas entre las tres etiquetas; en las demás la mediana es la misma (casi siempre 0) y la diferencia está en las colas o en el porcentaje de incidentes con valor distinto de cero.
- Las seis más llamativas por tamaño del efecto: {', '.join(f'`{v}`' for v in top6)} (figura abajo).
- **`n_paises` y `entidad_inicio_sesion_nube`** son las dos de mayor efecto y están casi calcadas: Spearman {dec(CORR_PAIS_NUBE, 2)}; de los incidentes con más de un país, {pct(PAIS_MASDE1_Y_NUBE)} tienen inicios de sesión en la nube (hipótesis: el país solo se registra en las evidencias de inicio de sesión). Por número de países, según la tabla:
""")
tp = PAIS.copy()
tp["incidentes"] = tp["incidentes"].map(lambda x: num(int(x)))
for g in ORDEN:
    tp[g] = tp[g].map(pct)
tp.index = [f"{i}" if i < 4 else "4 o más" for i in tp.index]
w(md(tp.rename_axis("n_paises")))
w("\n![Media por grado de las 6 variables con mayor efecto](figuras/fig2_contraste_por_grado.png)\n")

# ---- 4 ----
w("## 4. Impacto sobre la etiqueta con dos métodos distintos\n")
w("### 4a. Asociación una a una (sin controlar por las demás)\n")
w(f"Kruskal-Wallis compara las tres etiquetas con rangos (sin suponer normalidad; con muchos empates se aplica la corrección de empates). "
  f"**Con n = {num(N)} el valor p es ≈ 0 en casi todo; no sirve para ordenar:** por eso uso el tamaño del efecto ε². "
  f"La información mutua (MI) detecta también relaciones **no monótonas**; discretizo las variables (valores frecuentes = su propio nivel, el resto en deciles) y le resto el ruido medio de {N_SHUF} barajados al azar de `grado`.\n")
ta = A.sort_values("eps2", ascending=False)
tabla_a = pd.DataFrame({
    "H": ta.H.map(lambda x: num(float(x), 1)), "valor p": ta.p.map(pval), "ε²": ta.eps2.map(lambda x: dec(x, 4)),
    "efecto": ta.efecto, "MI % de H(grado)": (100 * ta.MI_pct).map(lambda x: dec(x, 2)),
    "ruido MI (azar) %": (100 * ta.MI_ruido / H_y).map(lambda x: dec(x, 3)),
})
w(md(tabla_a.rename_axis("variable")))
w(f"\nLa incertidumbre total de `grado` es H = {dec(H_y, 3)} nats; «MI %» es qué parte de ella reduce conocer la variable.\n")
w("\n**Categóricas con pocos niveles (V de Cramér).** `dia_semana` y `hora_inicio` también se analizaron como categorías (además de como números arriba). "
  "`organizacion` y `detector_principal` están en la sección 5.\n")
tv = V_CAT.loc[CAT_BAJA, ["niveles", "V", "V_corr", "V_azar", "MI_pct"]].astype(float)
tv_f = pd.DataFrame({"niveles": tv.niveles.map(lambda x: num(int(x))), "V de Cramér": tv.V.map(lambda x: dec(x, 3)),
                     "V corregida por sesgo": tv.V_corr.map(lambda x: dec(x, 3)),
                     "V si `grado` fuera azar": tv.V_azar.map(lambda x: dec(x, 3)),
                     "MI %": (100 * tv.MI_pct).map(lambda x: dec(x, 2))})
w(md(tv_f.rename_axis("variable")))
w(f"\nAmbas son débiles (V ≈ {dec(V_CAT.loc['dia_semana','V'],2)} y {dec(V_CAT.loc['hora_inicio','V'],2)}): cuándo empieza un incidente dice muy poco de cómo lo etiquetan.\n")
w("\n![Asociación una a una](figuras/fig3_asociacion_una_a_una.png)\n")
w(f"""
**Por qué ε² y MI a veces discrepan.** Kruskal-Wallis solo ve si una etiqueta tiende a tener valores *mayores* que otra; la MI ve cualquier dependencia. Ejemplos: `n_evidencias` ({dec(100*A.loc['n_evidencias','MI_pct'],1)} % de MI pero ε² {dec(A.loc['n_evidencias','eps2'],3)}) y `entidad_usuario` ({dec(100*A.loc['entidad_usuario','MI_pct'],1)} % y ε² {dec(A.loc['entidad_usuario','eps2'],3)}). No es «más pruebas = más grave»; son **valores exactos** que identifican a una etiqueta:
""")
ne = NE.copy()
ne["incidentes"] = ne["incidentes"].map(lambda x: num(int(x)))
for g in ORDEN:
    ne[g] = ne[g].map(pct)
w(md(ne.rename_axis("n_evidencias")))
w(f"""
Con exactamente 4 evidencias, {pct(NE.loc[4, 'sin amenaza'])} son «sin amenaza»; de esos incidentes, {pct(NE4_det0)} los genera el detector 0, {pct(NE4_1alerta)} tienen una sola alerta y vienen de {num(NE4_orgs)} organizaciones distintas (no es cosa de un solo cliente). **Hipótesis por verificar:** es la huella de una plantilla de alerta ruidosa del producto, que los analistas descartan casi siempre. Sería información útil para el triaje, pero describiría a la plantilla, no al ataque.
""")

w("### 4b. Importancia por permutación de un gradient boosting\n")
w(f"""Modelo: `HistGradientBoostingClassifier` (scikit-learn, {len([c for c in NUM])} variables numéricas; **sin** `organizacion` ni `detector_principal`, que van aparte), `class_weight='balanced'`, sin ajuste fino de hiperparámetros (no es el objetivo). Partición 70/30 **estratificada** por `grado` con semilla {SEED}. Métrica: F1 promedio (macro). La importancia se mide en el 30 % de prueba: cuánto baja el F1 macro al barajar una columna (10 repeticiones; la «sd» es solo la variación del barajado, no la de otra partición).

- F1 macro en prueba = **{dec(F1_TEST, 3)}** (entrenamiento {dec(F1_TRAIN, 3)}; piso «siempre AE» = {dec(F1_AZAR, 3)}). Por clase: AR {dec(F1_CLASE['amenaza real'], 3)}, SA {dec(F1_CLASE['sin amenaza'], 3)}, AE {dec(F1_CLASE['actividad esperada'], 3)} — la amenaza real, la clase que más interesa, es la que peor se distingue.
- Con la base **sin filas duplicadas** ({num(len(df_u))} filas) el F1 macro de prueba baja a **{dec(F1_TEST_U, 3)}** (entrenamiento {dec(F1_TRAIN_U, 3)}): es coherente con que parte de lo «explicado» en la base completa sea memorizar gemelos (las dos pruebas no son idénticas: la base sin duplicados tiene otras filas de prueba).
""")
tpi = PI.sort_values("media", ascending=False).head(15)
tpi_f = pd.DataFrame({
    "PI (caída F1 macro)": tpi["media"].map(lambda x: dec(x, 4)), "± sd": tpi["sd"].map(lambda x: dec(x, 4)),
    "puesto": [int(R.loc[v, "r_PI"]) for v in tpi.index],
    "PI sin duplicados": [dec(R.loc[v, "PI_sin_dup"], 4) for v in tpi.index],
    "puesto sin duplicados": [int(R.loc[v, "r_PI_sin_dup"]) for v in tpi.index],
})
w("**Las 15 variables de mayor importancia por permutación (base completa)**\n")
w(md(tpi_f.rename_axis("variable")))
w(f"""
Un valor de 0,0000 o negativo = barajar la variable no cambia (o mejora por azar) el F1: no aporta al modelo *dado el resto*. La correlación de puestos entre la base completa y la base sin duplicados es Spearman = {dec(rho_pi_dup, 2)}: el ranking es estable en lo grueso, pero no en el detalle.

**Importancia por bloques** (barajando todas las columnas de un bloque a la vez, y reentrenando sin el bloque / solo con el bloque). Esto corrige un problema de la permutación individual: cuando varias variables dicen lo mismo, barajar una sola apenas duele porque las otras la suplen.
""")
tb = pd.DataFrame({
    "variables": BL.n_vars.map(lambda x: num(int(x))),
    "caída F1 al barajar el bloque (± sd)": [f"{dec(r.caida_perm, 3)} ± {dec(r.sd, 3)}" for r in BL.itertuples()],
    "F1 sin el bloque": BL.F1_sin_bloque.map(lambda x: dec(x, 3)),
    "caída al excluirlo": BL.caida_exclusion.map(lambda x: dec(x, 3)),
    "F1 solo con el bloque": BL.F1_solo_bloque.map(lambda x: dec(x, 3)),
})
w(md(tb.rename_axis("bloque")))
b_max = BL.caida_perm.idxmax()
w(f"""
Lectura: el bloque **{b_max}** es el que más duele al barajarlo ({dec(BL.loc[b_max,'caida_perm'],3)}) y el único que casi solo reproduce el F1 completo ({dec(BL.loc[b_max,'F1_solo_bloque'],3)} frente a {dec(F1_TEST,3)}). Pero excluir cualquier bloque cuesta como máximo {dec(BL.caida_exclusion.max(),3)} de F1: la información está **repartida y redundante** entre bloques. Las «señales automáticas» (veredictos, sospecha, técnica MITRE), que uno esperaría decisivas, aportan poco ({dec(BL.loc['señales automáticas','caida_exclusion'],3)} al excluirlas).
""")

w("### Comparación de los dos métodos\n")
sobre = R[(R.dif_rango <= -8)].sort_values("dif_rango").index.tolist()
bajo = R[(R.dif_rango >= 8)].sort_values("dif_rango", ascending=False).index.tolist()
w(f"""
- **Acuerdo global:** correlación de Spearman entre el puesto por asociación (promedio de los puestos por ε² y por MI) y el puesto por permutación = **{dec(rho_ranks, 2)}**. Entre ε² y MI: {dec(rho_eps_mi, 2)}.
- **Coinciden en el top 10** ({len(top10_asoc & top10_pi)} de 10): {', '.join(f'`{v}`' for v in sorted(top10_asoc & top10_pi))}.
- **Asociación alta, permutación baja** (puesto por asociación ≥ 8 lugares mejor): {', '.join(f'`{v}`' for v in sobre) if sobre else 'ninguna'}. Suelen ser variables **redundantes** con otras (p. ej. `n_paises` ≈ `entidad_inicio_sesion_nube`, Spearman {dec(CORR_PAIS_NUBE, 2)}): solas se asocian mucho, pero el modelo ya las tiene cubiertas.
- **Permutación alta, asociación baja** (≥ 8 lugares): {', '.join(f'`{v}`' for v in bajo) if bajo else 'ninguna'}. Pueden ser efectos **combinados** con otras variables, no monótonos, o memorización de filas repetidas: p. ej. `hora_inicio` baja de {dec(R.loc['hora_inicio','PI'],4)} a {dec(R.loc['hora_inicio','PI_sin_dup'],4)} de PI al quitar los duplicados (`dia_semana`: de {dec(R.loc['dia_semana','PI'],4)} a {dec(R.loc['dia_semana','PI_sin_dup'],4)}).
- Pares de variables muy correlacionadas (|Spearman| ≥ 0,7) que reparten o duplican importancia: {'; '.join(f'`{a}`–`{b}` ({dec(r, 2)})' for a, b, r in pares)}.
""")
w("\n![Importancia por permutación y comparación de rangos](figuras/fig4_permutacion_y_comparacion.png)\n")

# ---- 5 ----
w("## 5. Variables con muchos niveles (reportadas aparte)\n")
w(f"`organizacion` ({num(ORG['niveles'])} niveles) y `detector_principal` ({DET['niveles']} niveles, uno de ellos «-1» = detector poco común) **no están en el ranking de la sección 6**.\n")
w("### Por qué las medidas de asociación se inflan con muchos niveles\n")
w(f"""Con una tabla de contingencia de miles de filas y solo 3 columnas, muchas celdas tienen pocos incidentes y, **por puro azar**, cada categoría queda con una mezcla de etiquetas distinta de la global; la chi-cuadrada y la V de Cramér lo leen como asociación. Lo comprobé barajando `grado` al azar (se destruye toda relación real) y recalculando: 
""")
tn = V_CAT.loc[["dia_semana", "hora_inicio", "detector_principal", "organizacion"],
               ["niveles", "V", "V_azar", "V_corr", "V_corr_azar", "MI_pct", "MI_azar_pct"]].astype(float)
tn_f = pd.DataFrame({
    "niveles": tn.niveles.map(lambda x: num(int(x))),
    "V observada": tn.V.map(lambda x: dec(x, 3)), "V con `grado` al azar": tn.V_azar.map(lambda x: dec(x, 3)),
    "V corregida": tn.V_corr.map(lambda x: dec(x, 3)), "V corregida al azar": tn.V_corr_azar.map(lambda x: dec(x, 3)),
    "MI %": (100 * tn.MI_pct).map(lambda x: dec(x, 1)), "MI % con azar": (100 * tn.MI_azar_pct).map(lambda x: dec(x, 1)),
})
w(md(tn_f.rename_axis("variable")))
w(f"""
Con `organizacion`, **aun sin ninguna relación real la V de Cramér sale {dec(V_CAT.loc['organizacion','V_azar'], 2)} y la MI {dec(100*V_CAT.loc['organizacion','MI_azar_pct'],1)} %**; con 7 niveles (`dia_semana`) sale {dec(V_CAT.loc['dia_semana','V_azar'], 3)}. Es el mismo defecto que ve un árbol de decisión o la ganancia de información: premian a las variables con muchas categorías. Por eso hay que mirar la versión **corregida por sesgo** y compararla con el azar: la de `organizacion` ({dec(V_CAT.loc['organizacion','V_corr'],2)}) y la de `detector_principal` ({dec(V_CAT.loc['detector_principal','V_corr'],2)}) están muy por encima del azar ({dec(V_CAT.loc['organizacion','V_corr_azar'],3)} y {dec(V_CAT.loc['detector_principal','V_corr_azar'],3)}): **la relación existe de verdad**, solo que el número crudo la exagera, y no es comparable con la V de variables de 7 niveles. (La V corregida de `organizacion` es una aproximación: {pct(ORG['con_menos_de_10'])} de las organizaciones tienen menos de 10 incidentes y la chi-cuadrada pierde fiabilidad con celdas tan pequeñas.)
""")
w("### Organización\n")
w(f"""
- Tamaños muy desiguales: mediana de {dec(ORG['mediana_inc'], 0)} incidentes por organización; las 10 mayores suman {pct(ORG['top10_share'])} de la base; {pct(ORG['con_menos_de_10'])} de las organizaciones tienen menos de 10 incidentes (juntas, {pct(ORG['inc_en_orgs_menos_10'])} de la base).
- **Pureza:** si a cada incidente se le asigna la etiqueta más común de *su* organización, se acierta en {pct(ORG['pureza'])} de los casos; con `grado` barajado al azar se acertaría en {pct(ORG['pureza_azar'])} (parte de eso es sobreajuste a organizaciones diminutas). {pct(ORG['con_unico_grado'])} de las organizaciones solo tienen incidentes de una etiqueta (esperable cuando la mediana es de {dec(ORG['mediana_inc'], 0)} incidentes por organización).
- Entre las {ORG['n_grandes']} organizaciones con ≥ 100 incidentes, la proporción de «amenaza real» va de {pct(ORG['ar_q'][0.0])} a {pct(ORG['ar_q'][1.0])}; mediana {pct(ORG['ar_q'][0.5])} y percentil 90 = {pct(ORG['ar_q'][0.9])}.
""")
to = top_org_pct.copy()
to["incidentes"] = to["incidentes"].map(lambda x: num(int(x)))
for g in ORDEN:
    to[g] = to[g].map(pct)
w("**Las 8 organizaciones más grandes**\n")
w(md(to.rename_axis("organizacion")))
w("\n### Detector principal\n")
td2 = top_det_pct.copy()
td2["incidentes"] = td2["incidentes"].map(lambda x: num(int(x)))
for g in ORDEN:
    td2[g] = td2[g].map(pct)
w(f"Pureza por detector: {pct(DET['pureza'])} (azar: {pct(DET['pureza_azar'])}). Los 8 detectores más frecuentes:\n")
w(md(td2.rename_axis("detector_principal")))
w("\n### ¿Cuánto predice la etiqueta saber el cliente o el detector?\n")
w(f"""Predicción con F1 macro en la misma partición 70/30 (y, para «orgs nuevas», el promedio de 5 particiones por organización, con rango {dec(min(F1_NUEVAS),3)}–{dec(max(F1_NUEVAS),3)}):
""")
tf = pd.DataFrame({"F1 macro": [dec(v, 3) for v in [F1_AZAR, F1_DET_MAY, F1_ORG_MAY, F1_TEST, F1_ID, F1_ORG_NUEVAS]]},
                  index=["Siempre la clase más frecuente (piso)",
                         "Etiqueta más común del detector (aprendida en entrenamiento)",
                         "Etiqueta más común de la organización (aprendida en entrenamiento)",
                         "Gradient boosting con las 35 variables, sin IDs",
                         "Gradient boosting con las 35 variables + `organizacion` + `detector_principal`",
                         "Gradient boosting sin IDs, probado en organizaciones que no vio nunca"])
w(md(tf.rename_axis("Predictor")))
w(f"""
Importancia por permutación en el modelo con IDs: `detector_principal` = {dec(PI_ID.loc['detector_principal','media'],3)} y `organizacion` = {dec(PI_ID.loc['organizacion','media'],3)} (caída de F1 macro), muy por encima de cualquiera de las 35 variables (la mayor: {dec(PI_ID.drop(IDS)['media'].max(),3)}). (En ese modelo `organizacion` entra como número —su identificador anónimo—, que es una forma tosca de representarla; el predictor «mayoría por organización» de la tabla es la medida más honesta de cuánto sabe la organización.)

![Variables con muchos niveles](figuras/fig5_variables_con_muchos_niveles.png)

**Qué significaría que la organización explique la etiqueta.**

1. **El «grado» refleja en parte a quién pertenece el incidente, no solo qué pasó.** Saber solo el cliente da un F1 macro de {dec(F1_ORG_MAY,2)} (clientes presentes en entrenamiento y prueba), más que las 35 variables del incidente juntas ({dec(F1_TEST,2)}); incluso saber solo el detector principal ({dec(F1_DET_MAY,2)}) iguala a esas 35 variables. Hay organizaciones donde casi todo es una sola etiqueta (tabla de arriba).
2. **Pueden mezclarse tres cosas que los datos no separan:** (a) el cliente de verdad recibe más o menos ataques reales; (b) cada cliente tiene su propia política y criterio al etiquetar (qué considera «actividad esperada»: pruebas de intrusión, escáneres internos, herramientas autorizadas); (c) los detectores activos y las plantillas de alerta que se generan difieren entre clientes (el detector principal por sí solo ya se asocia con la etiqueta; ver también 4a).
3. **No viaja:** el mismo modelo, sin IDs, cae de {dec(F1_TEST,2)} a {dec(F1_ORG_NUEVAS,2)} (promedio de 5 particiones; rango {dec(min(F1_NUEVAS),2)}–{dec(max(F1_NUEVAS),2)}) cuando se prueba en organizaciones que no vio (piso: {dec(F1_AZAR_NUEVAS,2)}); son pruebas distintas, así que la comparación es orientativa. Lo aprendido sirve sobre todo para clientes ya conocidos.
4. Si se usa la organización como predictor, el modelo aprende «quién», no «qué». Para saber si una variable del incidente importa de verdad hay que mirarla **dentro de cada organización** (análisis que este reporte no hace, ver límites).
""")

# ---- 6 ----
w("## 6. Ranking final: las 10 variables de más impacto\n")
w("Criterio: puesto medio entre (i) el puesto por asociación una a una (promedio de los puestos por ε² y por MI) y (ii) el puesto por importancia por permutación; "
  "desempata la permutación. No incluye `organizacion` ni `detector_principal` (sección 5).\n")
rows = []
for v in TOP.index:
    rows.append({
        "#": int(TOP.loc[v, "puesto"]), "variable": f"`{v}`",
        "ε²": dec(TOP.loc[v, "eps2"], 3), "MI %": dec(100 * TOP.loc[v, "MI_pct"], 1),
        "PI (± sd)": f"{dec(TOP.loc[v, 'PI'], 4)} ± {dec(TOP.loc[v, 'PI_sd'], 4)}",
        "puesto asociación": f"{TOP.loc[v, 'r_asoc']:.1f}".replace(".", ","), "puesto permutación": int(TOP.loc[v, "r_PI"]),
        "en palabras de analista": frase(v),
    })
w(md(pd.DataFrame(rows), index=False))
w(f"""
Notas:

- Composición del top 10: {sum(v.startswith('entidad_') for v in TOP.index)} `entidad_*`, {sum(v.startswith('tactica_') for v in TOP.index)} `tactica_*`, {sum(v.startswith('prop_') for v in TOP.index)} `prop_*` (señales automáticas), {sum(v.startswith('n_') for v in TOP.index)} de tamaño (`n_*`) y {sum(v in ('minutos_duracion','hora_inicio','dia_semana') for v in TOP.index)} de tiempo. Las entidades dominan: el bloque `entidades` solo ya recupera F1 macro = {dec(BL.loc['entidades','F1_solo_bloque'],2)} (completo: {dec(F1_TEST,2)}).
- Efectos «grandes» (ε² ≥ 0,14): {', '.join(f'`{v}`' for v in A.index[A.efecto == 'grande'])}, y son casi la misma señal (Spearman {dec(CORR_PAIS_NUBE, 2)}). `n_paises` tiene el mayor ε² de todos ({dec(A.loc['n_paises','eps2'],3)}) pero queda en el puesto {int(R.loc['n_paises','puesto'])} porque la permutación la ve como copia de `entidad_inicio_sesion_nube`.
- Variables que **no** aparecen y podrían esperarse: `n_alertas`, `n_detectores`, `minutos_duracion`, `hora_inicio` y `dia_semana` (≈ nulas en ambos métodos) y los veredictos `prop_veredicto_*` (puestos {int(R.loc['prop_veredicto_malicioso','puesto'])} y {int(R.loc['prop_veredicto_sin_amenazas','puesto'])} de 35).
""")
w("**Ranking completo (35 variables)**\n")
rc = pd.DataFrame({"ε²": R.eps2.map(lambda x: dec(x, 4)), "MI %": (100 * R.MI_pct).map(lambda x: dec(x, 2)),
                   "PI": R.PI.map(lambda x: dec(x, 4)), "puesto asoc.": R.r_asoc.map(lambda x: f"{x:.1f}".replace(".", ",")),
                   "puesto PI": R.r_PI.astype(int), "puesto final": R.puesto})
w(md(rc.rename_axis("variable")))

# ---- 7 ----
w("\n## 7. Límites: qué no se puede concluir\n")
w("### Asociación no es causalidad\n")
w("Que `entidad_inicio_sesion_nube` se asocie con «sin amenaza» no significa que los inicios de sesión en la nube *hagan* inocuo un incidente: puede ser que los detectores de identidad en la nube sean ruidosos, que ciertos clientes (con otros hábitos de etiquetado) los usen más, o que ambas cosas dependan de un tercer factor. Los datos son observacionales y no hay manera de manipular nada.\n")
w("### ¿Alguna variable podría contener la respuesta o información posterior a la decisión?\n")
w(f"""Con lo que trae el archivo **no puedo confirmarlo ni descartarlo**; esto es lo que sí se puede decir:

- **`prop_veredicto_malicioso`, `prop_veredicto_sin_amenazas`, `prop_nivel_sospecha`** son veredictos y niveles automáticos. Si el sistema los fue actualizando *después* de que el analista cerró el incidente, serían información posterior a la decisión. El diccionario no dice cuándo se asignan. Evidencia a favor de que no «regalan» la respuesta: en la tabla siguiente, aun cuando existe veredicto «malicioso», solo una parte son «amenaza real», y excluir el bloque de señales automáticas cuesta {dec(BL.loc['señales automáticas','caida_exclusion'],3)} de F1. Esto **no prueba** que sean anteriores.
- **`n_alertas`, `n_evidencias`, `n_detectores`, `minutos_duracion`, `n_paises`, `entidad_*`, `tactica_*`:** miden el incidente tal como está en el archivo. Si un incidente sigue acumulando alertas mientras se investiga (o si los incidentes confirmados se enriquecen con más pruebas), el tamaño y la duración podrían reflejar la investigación y no la situación al llegar al analista. No sé en qué momento se «congelaron».
- **`organizacion` y `detector_principal`** son identificadores; no «contienen» la respuesta pero la predicen por la política del cliente (sección 5).
- **Gemelos entre entrenamiento y prueba:** {pct(gem_prueba)} de las filas de prueba tienen una copia exacta en entrenamiento; eso sí infla el F1 (baja de {dec(F1_TEST,2)} a {dec(F1_TEST_U,2)} sin duplicados).
""")
tv2 = VER.copy()
tv2["incidentes"] = tv2["incidentes"].map(lambda x: num(int(x)))
for g in ORDEN:
    tv2[ABREV[g]] = tv2[ABREV[g]].map(pct)
w(md(tv2[["incidentes"] + [ABREV[g] for g in ORDEN]].rename_axis("subconjunto")))
w(f"""
### Lo que NO se puede concluir con estos datos

- **Causas, ni efecto «manteniendo lo demás igual»:** la permutación mide cuánto usa el modelo una columna, no su efecto causal; con variables correlacionadas el crédito se reparte arbitrariamente (ver pares correlacionados en 4b).
- **Que valga para otros clientes o para producción:** con organizaciones nuevas el F1 macro cae a {dec(F1_ORG_NUEVAS,2)}. El modelo es solo un instrumento de medición; su F1 ({dec(F1_TEST,2)}) no lo hace desplegable (la clase «amenaza real» llega a F1 = {dec(F1_CLASE['amenaza real'],2)}).
- **La verdadera prevalencia de amenazas:** es una muestra aleatoria de {num(N)} incidentes del archivo de prueba de GUIDE; las proporciones ({pct(Q['prop_grado']['amenaza real'])} AR) no son las de ningún SOC.
- **Que la etiqueta sea «la verdad»:** es el criterio de un analista del cliente. Hay {num(Q['grupos_conflicto'])} grupos de incidentes idénticos con etiquetas distintas, así que hay ruido y subjetividad.
- **Nada sobre el orden en el tiempo ni sobre flujos de ataque:** no hay fecha, solo hora y día de la semana (y no sé en qué zona horaria están); tampoco la secuencia de tácticas, solo su proporción.
- **Significación estadística:** con n = {num(N)}, p ≈ 0 incluso para efectos «insignificantes» (`n_alertas`: ε² = {dec(A.loc['n_alertas','eps2'], 4)}, p = {pval(float(A.loc['n_alertas', 'p']))}); tomar solo el tamaño del efecto. Los umbrales de ε² son una convención, y las «sd» de la permutación no incluyen la variabilidad entre particiones (una sola partición con semilla fija).
- **Variables raras:** `tactica_persistencia`, `tactica_evasion_de_defensas` y `tactica_movimiento_lateral` casi nunca son distintas de cero; su baja importancia puede ser falta de casos, no falta de relevancia.
- **Hiperparámetros sin ajustar** y una sola familia de modelo: otro modelo podría reordenar el medio de la tabla.

## 8. Resumen y qué revisar antes de creerlo
""")
RESUMEN = [
    f"1. Base: {num(N)} incidentes, 0 nulos, sin valores imposibles; {pct(Q['prop_grado']['actividad esperada'])} actividad esperada, {pct(Q['prop_grado']['sin amenaza'])} sin amenaza, {pct(Q['prop_grado']['amenaza real'])} amenaza real.",
    f"2. {num(Q['dup_total'])} filas ({pct(Q['dup_total'] / N)}) son copias exactas (todas de 1 alerta) y {num(Q['grupos_conflicto'])} grupos idénticos tienen etiquetas distintas; sin duplicados el F1 del modelo baja de {dec(F1_TEST,2)} a {dec(F1_TEST_U,2)}.",
    f"3. Las variables de tamaño/duración están muy sesgadas (hasta {num(int(D.loc['n_alertas','max']))} alertas; `minutos_duracion` = 0 en {pct(D.loc['minutos_duracion','pct_ceros'],0)}), así que la media engaña y casi todas las medianas por etiqueta coinciden.",
    f"4. Ninguna variable del incidente explica bien la etiqueta por sí sola: el mayor ε² es {dec(A.eps2.max(),2)} (`{A.eps2.idxmax()}`) y el resto son pequeños; el modelo completo llega a F1 macro = {dec(F1_TEST,2)}, y {dec(F1_CLASE['amenaza real'],2)} en «amenaza real».",
    f"5. Top 3 combinando ambos métodos: " + ", ".join(f"`{v}`" for v in TOP.index[:3]) + "; dominan las entidades (qué tipo de evidencia aparece), no las tácticas ni los veredictos automáticos.",
    f"6. Los dos métodos coinciden en lo grueso (Spearman = {dec(rho_ranks,2)}; {len(top10_asoc & top10_pi)}/10 en común) y discrepan por redundancia (`n_paises` ≈ inicios de sesión en la nube) y por relaciones no monótonas (`n_evidencias`, `entidad_usuario`).",
    f"7. Las señales automáticas (veredictos, sospecha, técnica MITRE) aportan poco: excluirlas cuesta {dec(BL.loc['señales automáticas','caida_exclusion'],3)} de F1.",
    f"8. `organizacion` ({num(ORG['niveles'])} niveles) y `detector_principal` ({DET['niveles']}) tienen V de Cramér inflada por el número de niveles (con `grado` al azar la V de organización sale {dec(V_CAT.loc['organizacion','V_azar'],2)}), pero corregida sigue muy por encima del azar ({dec(V_CAT.loc['organizacion','V_corr'],2)}).",
    f"9. Solo con la organización se logra F1 macro = {dec(F1_ORG_MAY,2)} (más que las 35 variables, {dec(F1_TEST,2)}); en organizaciones nuevas el modelo cae a {dec(F1_ORG_NUEVAS,2)}: la etiqueta depende del cliente (política de etiquetado y plantillas de alerta), no solo del incidente.",
    f"10. Todo es asociación en una muestra de un solo conjunto; no hay causalidad ni evidencia de que sirva en producción.",
]
REVISAR = [
    "Preguntar a quien produjo GUIDE **cuándo** se asignan los veredictos automáticos y se congelan los conteos de alertas/evidencias (posible información posterior a la decisión).",
    "Repetir el análisis **dentro de cada organización grande** (o con efectos aleatorios) para separar el efecto del incidente del efecto del cliente.",
    "Rehacer la partición **agrupando por organización y por huella de fila** (sin gemelos) y con varias semillas, para ver cuánto del ranking sobrevive.",
    "Auditar una muestra de incidentes idénticos con etiquetas distintas, y las plantillas de alerta tipo `n_evidencias` = 4, con un analista del SOC.",
    "Confirmar con un analista que el sentido de patrones contraintuitivos (p. ej. más comando y control en «actividad esperada» que en «amenaza real») tiene explicación operativa.",
]
for linea in RESUMEN:
    w(linea)
w("\n**Qué revisaría antes de creerlo**\n")
for r in REVISAR:
    w(f"- {r}")
import inspect
_src = Path(__file__).read_text(encoding="utf-8")
_kw = _src[_src.index("filas = []\nfor c in NUM:"):_src.index("A = pd.DataFrame(filas)")]
w("\n## Apéndice: código clave (el completo está en `salida_demo/analisis.py`)\n")
w("**Kruskal-Wallis, ε² e información mutua (una a una)**\n\n```python\n" + _kw.rstrip() + "\n```\n")
w("**Discretización para la MI y V de Cramér (con corrección de sesgo)**\n\n```python\n" + inspect.getsource(discretizar) + "\n" + inspect.getsource(cramer_v) + "```\n")
w("**Modelo, partición 70/30 estratificada e importancia por permutación**\n\n```python\n" + inspect.getsource(nuevo_modelo) + "\n" + inspect.getsource(particion) + "\n" + inspect.getsource(ajustar_y_permutar) + "```\n")
w("\n---\n*Generado por `salida_demo/analisis.py`.*\n")

(AQUI / "reporte.md").write_text("\n".join(L), encoding="utf-8")
log(f"reporte.md escrito ({len(L)} bloques)")
print("\n".join(RESUMEN))
