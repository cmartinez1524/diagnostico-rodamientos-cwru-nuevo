"""Incertidumbre de la comparación con/sin frecuencia esperada.

Lee comparacion_133.csv (no necesita los .mat) y agrega lo que el resumen
actual no informa: intervalos de confianza, pruebas pareadas y un bootstrap
por condición de adquisición (tipo x tamaño x sensor), porque las cargas de
una misma condición no son experimentos independientes.

Uso:  python 05_Diagnostico_Ciego/incertidumbre_resultados.py
Salida: comparacion_con_sin_esperada_todos/incertidumbre.{json,md}
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

AQUI = Path(__file__).resolve().parent
ORDEN = {"ND": 0, "PD": 1, "D": 2}
SEMILLA, REPETICIONES = 2026, 10000


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def condicion(archivo: str) -> str:
    """12k_DE/OR007@6_2.mat -> 12k_DE/OR007@6 (une las cargas 0-3)."""
    return re.sub(r"_\d+\.mat$", "", archivo)


def cambio(grado: pd.Series, paper: pd.Series, tipo_ok: pd.Series) -> pd.Series:
    """+1 mejora, 0 igual, -1 empeora, -2 tipo incorrecto."""
    d = np.sign(grado.map(ORDEN) - paper.map(ORDEN)).astype(int)
    return d.where(tipo_ok, -2)


def resumen(c: pd.Series, grupos: pd.Series, rng: np.random.Generator) -> dict:
    n = len(c)
    mej, emp, mal = int((c == 1).sum()), int((c == -1).sum()), int((c == -2).sum())
    ok = int((c >= 0).sum())
    # bootstrap por condición: se remuestrean condiciones completas
    ids = grupos.unique()
    idx = {g: np.flatnonzero(grupos.values == g) for g in ids}
    v = c.values
    tasas, netos = [], []
    for _ in range(REPETICIONES):
        sel = np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])
        s = v[sel]
        tasas.append((s >= 0).mean())
        netos.append(((s == 1).sum() - (s < 0).sum()) / len(s) * n)
    return {
        "n": n, "condiciones": int(len(ids)),
        "mejoran": mej, "empeoran": emp, "tipo_incorrecto": mal,
        "sin_retroceso": ok, "tasa": ok / n,
        "ic95_wilson": wilson(ok, n),
        "ic95_bootstrap_condicion": [float(x) for x in np.percentile(tasas, [2.5, 97.5])],
        # ¿hay más mejoras que retrocesos? (signos; los iguales no informan)
        "p_signos_mejora_vs_empeora": float(binomtest(mej, mej + emp, 0.5).pvalue),
        "p_signos_mejora_vs_empeora_o_tipo_mal": float(binomtest(mej, mej + emp + mal, 0.5).pvalue),
        "neto_estricto": mej - emp - mal,
        "neto_estricto_ic95_bootstrap": [float(x) for x in np.percentile(netos, [2.5, 97.5])],
    }


def main(raiz: Path = AQUI) -> dict:
    carpeta = raiz / "comparacion_con_sin_esperada_todos"
    d = pd.read_csv(carpeta / "comparacion_133.csv")
    a = d[d.paper.notna()].reset_index(drop=True)
    g = a.archivo.map(condicion)
    rng = np.random.default_rng(SEMILLA)
    ciego = cambio(a.grado_sin_esperada, a.paper, a.tipo_correcto)
    oraculo = cambio(a.grado_con_esperada, a.paper, pd.Series(True, index=a.index))
    sal = {"semilla": SEMILLA, "repeticiones": REPETICIONES,
           "sin_esperada": resumen(ciego, g, rng), "oraculo": resumen(oraculo, g, rng)}
    # McNemar exacto: registros donde solo uno de los dos escenarios retrocede
    b = int(((oraculo >= 0) & (ciego < 0)).sum()); c = int(((oraculo < 0) & (ciego >= 0)).sum())
    sal["oraculo_vs_ciego"] = {"solo_oraculo_ok": b, "solo_ciego_ok": c,
                               "p_mcnemar_exacto": float(binomtest(b, b + c, 0.5).pvalue)}
    # Filas donde retroceder es posible (paper PD o D): quita los éxitos automáticos de ND
    m = a.paper != "ND"
    for nombre, s in (("sin_esperada", ciego), ("oraculo", oraculo)):
        k, n = int((s[m] >= 0).sum()), int(m.sum())
        sal[nombre]["sin_nd_del_paper"] = {"k": k, "n": n, "tasa": k / n, "ic95_wilson": wilson(k, n)}
    # Tipo correcto según protocolo y resto de señales
    for o, s in a.groupby("origen"):
        k, n = int(s.tipo_correcto.sum()), len(s)
        sal.setdefault("tipo_12k_por_protocolo", {})[o] = {"k": k, "n": n, "ic95_wilson": wilson(k, n)}
    f = d[(d.fs == 48000) & (d.real != "N")]
    k = int(f.tipo_correcto.sum()); sal["tipo_48k"] = {"k": k, "n": len(f), "ic95_wilson": wilson(k, len(f))}
    s = d[d.real == "N"]; k = int(s.tipo_correcto.sum())
    sal["normales"] = {"k": k, "n": len(s), "ic95_wilson": wilson(k, len(s))}
    (carpeta / "incertidumbre.json").write_text(json.dumps(sal, indent=2, ensure_ascii=False), encoding="utf-8")
    L = ["# Incertidumbre de la comparación (105 registros de 12 kHz)", "",
         "Bootstrap por condición de adquisición; pruebas exactas. No es validación externa.", "",
         "| Escenario | Sin retroceso | IC95 Wilson | IC95 por condición | Neto estricto | IC95 neto | p signos |",
         "| --- | ---: | --- | --- | ---: | --- | ---: |"]
    for nombre, t in (("Oráculo", sal["oraculo"]), ("Sin frecuencia esperada", sal["sin_esperada"])):
        L.append(f"| {nombre} | {t['sin_retroceso']}/{t['n']} | {t['ic95_wilson'][0]:.1%}–{t['ic95_wilson'][1]:.1%} | "
                 f"{t['ic95_bootstrap_condicion'][0]:.1%}–{t['ic95_bootstrap_condicion'][1]:.1%} | {t['neto_estricto']:+d} | "
                 f"{t['neto_estricto_ic95_bootstrap'][0]:+.0f} a {t['neto_estricto_ic95_bootstrap'][1]:+.0f} | "
                 f"{t['p_signos_mejora_vs_empeora_o_tipo_mal']:.3f} |")
    L += ["", "Neto estricto = mejoran − empeoran − tipo incorrecto. p signos: mejoras frente a retrocesos de cualquier clase.",
          f"Oráculo frente a ciego (McNemar exacto): p = {sal['oraculo_vs_ciego']['p_mcnemar_exacto']:.3f}."]
    (carpeta / "incertidumbre.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return sal


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI), indent=1, ensure_ascii=False))
