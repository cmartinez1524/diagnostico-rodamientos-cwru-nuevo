"""Evalúa el clasificador de tipo nuevo contra la receta vigente, con validación estricta.

Protocolos (105 registros de 12 kHz con referencia del paper):
  repo       OOF por condición en la mitad de desarrollo + modelo congelado en el resto
             (el protocolo de `05_Diagnostico_Ciego`; debe reproducir los tipos publicados).
  loco       se excluye la condición completa (tipo × tamaño × posición × extremo, todas sus
             cargas) de cada registro evaluado, sobre los 105. Ningún registro se evalúa con un
             modelo que vio su condición.
  anidado    (--anidado) igual que loco, pero la receta (arbol, peine o ambos) se elige dentro
             de cada pliegue con las condiciones restantes. Estima el sesgo de haber elegido receta.

La banda, la regla de grado ND/PD/D y el detector sano/falla son los del proyecto, sin cambios.
Métrica: sin retroceso = tipo correcto y grado >= grado del paper.

Uso:  python 06_Tipo_Ciego_Peine/evaluar.py [--anidado]
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import binomtest

AQUI = Path(__file__).resolve().parent
C5 = AQUI.parent/'05_Diagnostico_Ciego'
sys.path.insert(0, str(C5))
from modelo_ciego import F  # noqa: E402
from modelo_por_dominio import calidad, dominio_catalogo  # noqa: E402
from ventanas_verificadas import rasgos as rasgos_repo  # noqa: E402
from modelo_jerarquico import aplicar  # noqa: E402
from clasificador import Ensamble, vistas, nuevo_lda, puntajes_exportados, TIPOS  # noqa: E402

DATOS = AQUI.parent/'data2/CWRU'
CACHE = AQUI/'cache'
SALIDA = AQUI/'resultados'
ORDEN = {'ND': 0, 'PD': 1, 'D': 2}
SEMILLA, REPETICIONES = 2026, 10000


def condicion(archivo):
    """12k_FE/OR007@3_2.mat -> 12k_FE/OR007@3: une las cuatro cargas de una misma falla."""
    return re.sub(r'_\d+\.mat$', '', archivo)


def preparar():
    """Todo lo que no depende de etiquetas se calcula una vez por señal y se guarda en cache/."""
    CACHE.mkdir(exist_ok=True)
    idx = F.leer_indice(str(DATOS))
    assert len(idx) == 133 and idx.archivo.is_unique
    filas, datos = [], {}
    for i, r in enumerate(idx.itertuples()):
        mat = loadmat(DATOS/r.archivo)
        x = np.ravel(mat[F.clave_canal(mat, r.sensor_falla, int(r.registro))]).astype(float)
        rpm = F.rpm_registro(mat, int(r.registro), nominal=float(r.rpm_nominal))
        c = {k: float(v*rpm/60) for k, v in F.MULTIPLICADORES[r.sensor_falla].items()}
        huella = hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()
        fila = dict(archivo=r.archivo, real=r.zona, fs=int(r.fs), sensor=r.sensor_falla, sha256_senal=huella,
                    suficiente=bool(len(x)//2 >= 2.5*r.fs))
        ruta = CACHE/(huella[:24]+'.npz')
        if fila['suficiente']:
            if not ruta.exists():
                v = vistas(x, int(r.fs), c)
                base = rasgos_repo(x, int(r.fs), c)
                grados = {t: calidad(x, int(r.fs), c, t)['grado'] for t in TIPOS}
                puerta = aplicar(base.mean(axis=0), C5/'resultados_por_dominio'/f'modelo_{dominio_catalogo(c)}.npz')
                np.savez_compressed(ruta, arbol=v['arbol'], peine=v['peine'], base=base,
                                    grados=np.array([grados[t] for t in TIPOS]), puerta=np.array(puerta))
                print(f'{i+1}/133 {r.archivo}', flush=True)
            with np.load(ruta, allow_pickle=False) as z:
                datos[r.archivo] = dict(arbol=z['arbol'], peine=z['peine'], base=z['base'],
                                        grados=dict(zip(TIPOS, map(str, z['grados']))), puerta=str(z['puerta']))
        filas.append(fila)
    return pd.DataFrame(filas), datos


def receta_repo(entreno, etiquetas, prueba):
    """LDA del proyecto sobre las cuatro ventanas del prefijo de 2,5 s."""
    clf = nuevo_lda().fit(np.vstack([r['base'] for r in entreno]), np.concatenate([[e]*len(r['base']) for r, e in zip(entreno, etiquetas)]))
    return [str(clf.classes_[clf.decision_function(r['base']).mean(axis=0).argmax()]) for r in prueba]


def receta_nueva(entreno, etiquetas, prueba, nombres=('arbol', 'peine')):
    m = Ensamble(nombres).ajustar(entreno, etiquetas)
    return [m.predecir(r) for r in prueba]


def por_grupos(receta, R, y, g, indices):
    """Predicción de cada registro con un modelo que no vio ningún registro de su condición."""
    pred = np.empty(len(y), dtype=object)
    for grupo in sorted(set(g[indices])):
        te = [i for i in indices if g[i] == grupo]
        tr = [i for i in indices if g[i] != grupo]
        assert grupo not in set(g[tr])
        pred[te] = receta([R[i] for i in tr], y[tr], [R[i] for i in te])
    return pred


def protocolo_repo(receta, R, y, g, dev):
    pred = por_grupos(receta, R, y, g, np.flatnonzero(dev))
    tr, te = np.flatnonzero(dev), np.flatnonzero(~dev)
    pred[te] = receta([R[i] for i in tr], y[tr], [R[i] for i in te])
    return pred


def anidado(R, y, g):
    """La receta se elige en cada pliegue externo con un LOCO interno; empates: la más simple."""
    candidatas = [('arbol',), ('peine',), ('arbol', 'peine')]
    pred = np.empty(len(y), dtype=object)
    elegidas = {}
    for grupo in sorted(set(g)):
        te = np.flatnonzero(g == grupo)
        tr = np.flatnonzero(g != grupo)
        # Un solo ajuste por pliegue interno: los puntajes de cada vista se combinan después.
        internos = {}
        for gi in sorted(set(g[tr])):
            tri = tr[g[tr] != gi]
            m = Ensamble().ajustar([R[i] for i in tri], y[tri])
            for i in tr[g[tr] == gi]:
                internos[i] = m.puntajes_por_vista(R[i])
        aciertos = [int(sum(TIPOS[int(sum(internos[i][v] for v in nombres).argmax())] == y[i] for i in tr)) for nombres in candidatas]
        mejor = candidatas[int(np.argmax(aciertos))]
        pred[te] = receta_nueva([R[i] for i in tr], y[tr], [R[i] for i in te], mejor)
        elegidas[grupo] = dict(receta='+'.join(mejor), aciertos_internos=dict(zip(['+'.join(c) for c in candidatas], aciertos)), de=len(tr))
        print(f'anidado {grupo}: {elegidas[grupo]["receta"]}', flush=True)
    return pred, elegidas


def puntuar(pred, y, grados, paper):
    """Métrica del proyecto: una predicción de tipo equivocada nunca cuenta, sea cual sea su grado."""
    gr = np.array([grados[i][p] for i, p in enumerate(pred)])
    d = np.array([ORDEN[a]-ORDEN[b] for a, b in zip(gr, paper)])
    ok = pred == y
    estado = np.where(~ok, 'tipo_incorrecto', np.where(d > 0, 'mejora', np.where(d == 0, 'igual', 'empeora')))
    return gr, estado, dict(tipo_correcto=int(ok.sum()), mejoran=int((estado == 'mejora').sum()), iguales=int((estado == 'igual').sum()),
                            empeoran=int((estado == 'empeora').sum()), tipo_incorrecto=int((~ok).sum()),
                            sin_retroceso=int(np.isin(estado, ['mejora', 'igual']).sum()), total=len(y))


def intervalo(exitos, g, rng):
    """Bootstrap por condición: se remuestrean condiciones completas, no registros sueltos."""
    ids = np.unique(g)
    pos = {k: np.flatnonzero(g == k) for k in ids}
    t = [exitos[np.concatenate([pos[k] for k in rng.choice(ids, len(ids))])].mean() for _ in range(REPETICIONES)]
    return [float(v) for v in np.percentile(t, [2.5, 97.5])]


def pareada(a, b):
    """McNemar exacto entre dos vectores de éxito sobre los mismos registros."""
    solo_a, solo_b = int((a & ~b).sum()), int((~a & b).sum())
    return dict(solo_referencia=solo_a, solo_nuevo=solo_b,
                p_mcnemar_exacto=float(binomtest(solo_b, solo_a+solo_b, .5).pvalue) if solo_a+solo_b else 1.)


def figura(resumen, ruta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    claves = [('repo · receta vigente', 'repo_vigente'), ('repo · receta nueva', 'repo_nuevo'),
              ('estricto · receta vigente', 'loco_vigente'), ('estricto · receta nueva', 'loco_nuevo')]
    partes = [('mejoran', '#174A7C'), ('iguales', '#B9D0E8'), ('empeoran', '#F0B37E'), ('tipo_incorrecto', '#3A3F47')]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, (titulo, k) in enumerate(claves):
        izq = 0
        for nombre, color in partes:
            v = resumen[k][nombre]
            ax.barh(i, v, left=izq, color=color, edgecolor='white')
            ax.text(izq+v/2, i, str(v), ha='center', va='center', fontsize=9, color='white' if color in ('#174A7C', '#3A3F47') else 'black')
            izq += v
        ax.text(106, i, f'{resumen[k]["sin_retroceso"]}/105', va='center', fontsize=10, fontweight='bold')
    ax.axvline(90, color='black', linestyle='--', linewidth=1)
    ax.text(90, -.72, 'meta 90', ha='center', fontsize=9)
    ax.set_yticks(range(len(claves)), [t for t, _ in claves])
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel('registros de 12 kHz')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in partes], ['mejoran', 'iguales', 'empeoran', 'tipo incorrecto'],
              ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(.45, 1.2), fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--anidado', action='store_true', help='añade el LOCO anidado (lento)')
    args = ap.parse_args()
    SALIDA.mkdir(exist_ok=True)
    meta, datos = preparar()
    ref = pd.read_csv(C5/'comparacion_con_sin_esperada_todos/comparacion_133.csv').set_index('archivo')
    assert (meta.set_index('archivo').sha256_senal == ref.sha256_senal.reindex(meta.archivo)).all(), 'Las señales no son las del proyecto'
    part = pd.read_csv(C5/'particion_desarrollo.csv').set_index('archivo').particion

    m12 = meta[(meta.fs == 12000) & (meta.real != 'N')].reset_index(drop=True)
    assert len(m12) == 105 and m12.suficiente.all()
    R = [datos[a] for a in m12.archivo]
    y = m12.real.to_numpy()
    g = m12.archivo.map(condicion).to_numpy()
    dev = m12.archivo.map(part).eq('desarrollo').to_numpy()
    paper = ref.paper.reindex(m12.archivo).to_numpy()
    grados = [r['grados'] for r in R]
    assert dev.sum() == 52 and len(set(g)) == 28 and all(r['puerta'] != 'N' for r in R)

    pred = dict(repo_vigente=protocolo_repo(receta_repo, R, y, g, dev), repo_nuevo=protocolo_repo(receta_nueva, R, y, g, dev),
                loco_vigente=por_grupos(receta_repo, R, y, g, np.arange(105)), loco_nuevo=por_grupos(receta_nueva, R, y, g, np.arange(105)))
    # Control: el protocolo y la receta del proyecto deben reproducir sus tipos y grados publicados.
    assert (pred['repo_vigente'] == ref.tipo_sin_esperada.reindex(m12.archivo).to_numpy()).all(), 'No se reproducen los tipos publicados'
    elegidas = None
    if args.anidado:
        pred['loco_anidado'], elegidas = anidado(R, y, g)

    rng = np.random.default_rng(SEMILLA)
    tabla = m12[['archivo', 'real', 'sensor']].assign(condicion=g, particion=np.where(dev, 'desarrollo', 'reserva'), paper=paper)
    resumen, exito = {}, {}
    for k, p in pred.items():
        gr, estado, r = puntuar(p, y, grados, paper)
        exito[k] = np.isin(estado, ['mejora', 'igual'])
        r['ic95_sin_retroceso_por_condicion'] = intervalo(exito[k], g, rng)
        r['ic95_tipo_por_condicion'] = intervalo(p == y, g, rng)
        resumen[k] = r
        tabla[f'tipo_{k}'], tabla[f'grado_{k}'], tabla[f'estado_{k}'] = p, gr, estado
    assert resumen['repo_vigente']['sin_retroceso'] == 88 and (tabla.grado_repo_vigente.to_numpy() == ref.grado_sin_esperada.reindex(m12.archivo).to_numpy()).all()
    techo = puntuar(y, y, grados, paper)[2]
    resumen['techo_tipo_perfecto'] = techo
    resumen['pareadas'] = dict(
        loco_sin_retroceso=pareada(exito['loco_vigente'], exito['loco_nuevo']),
        loco_tipo=pareada(pred['loco_vigente'] == y, pred['loco_nuevo'] == y),
        repo_sin_retroceso=pareada(exito['repo_vigente'], exito['repo_nuevo']),
        repo_tipo=pareada(pred['repo_vigente'] == y, pred['repo_nuevo'] == y))
    if elegidas:
        resumen['anidado_recetas'] = elegidas
    tabla.to_csv(SALIDA/'predicciones_105.csv', index=False)
    pd.crosstab(pd.Series(paper, name='paper'), pd.Series(np.where(pred['loco_nuevo'] == y, tabla.grado_loco_nuevo, 'Tipo incorrecto'), name='modelo')
                ).reindex(index=['ND', 'PD', 'D'], columns=['ND', 'PD', 'D', 'Tipo incorrecto'], fill_value=0).to_csv(SALIDA/'matriz_loco_nuevo.csv')
    pd.crosstab(pd.Series(y, name='real'), pd.Series(pred['loco_nuevo'], name='modelo')).to_csv(SALIDA/'confusion_tipo_loco_nuevo.csv')

    # Señales sin referencia del paper: 48 kHz con falla y normales. Modelo ajustado con los 105 de 12 kHz,
    # excluyendo la misma falla física (tipo y tamaño en el extremo motor) cuando existe a 12 kHz.
    otras = []
    for r in meta[(meta.fs == 48000)].itertuples():
        if not r.suficiente:
            otras.append(dict(archivo=r.archivo, real=r.real, tipo='sin_diagnostico', motivo='menos_de_5_segundos'))
            continue
        d = datos[r.archivo]
        if d['puerta'] == 'N':
            tipo = 'N'
        else:
            fuera = condicion(r.archivo).replace('48k_DE', '12k_DE')
            tr = np.flatnonzero(g != fuera)
            tipo = receta_nueva([R[i] for i in tr], y[tr], [d])[0]
        otras.append(dict(archivo=r.archivo, real=r.real, tipo=tipo, motivo='detector_sano_falla' if d['puerta'] == 'N' else 'clasificador'))
    otras = pd.DataFrame(otras)
    otras['correcto'] = otras.tipo == otras.real
    otras.to_csv(SALIDA/'otras_senales.csv', index=False)
    f48, nor = otras[otras.real != 'N'], otras[otras.real == 'N']
    resumen['48k_con_falla'] = dict(tipo_correcto=int(f48.correcto.sum()), total=len(f48), sin_diagnostico=int((f48.tipo == 'sin_diagnostico').sum()))
    resumen['normales'] = dict(correctos=int(nor.correcto.sum()), total=len(nor), nota='detector sano/falla del proyecto, sin cambios')

    # Modelo final: ajustado con los 105; se exporta sin pickle y se comprueba la inferencia con NumPy.
    final = Ensamble().ajustar(R, y)
    firmas = {p: hashlib.sha256((AQUI/p).read_bytes()).hexdigest() for p in ('peine.py', 'clasificador.py')}
    final.exportar(SALIDA/'modelo_tipo.npz', **firmas, entrenamiento=np.array(m12.sha256_senal))
    for r in R[::7]:
        s, tipos = puntajes_exportados(SALIDA/'modelo_tipo.npz', r)
        assert np.allclose(s, final.puntajes(r)) and tipos == list(TIPOS)
    resumen['protocolo'] = dict(semilla=SEMILLA, repeticiones=REPETICIONES, condiciones=28, firmas=firmas,
        receta_nueva='LDA(arbol, 9 ventanas de la primera mitad) + LDA(peine, 93 rasgos); puntajes tipificados y sumados',
        sin_cambios='banda (bandas_estables), regla de grado (criterio_usuario), detector sano/falla (resultados_por_dominio)',
        validacion_externa=False)
    (SALIDA/'resumen.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
    figura(resumen, SALIDA/'comparacion.png')

    def linea(titulo, k):
        r = resumen[k]
        a, b = r['ic95_sin_retroceso_por_condicion']
        return (f'| {titulo} | {r["tipo_correcto"]}/105 | {r["mejoran"]} | {r["iguales"]} | {r["empeoran"]} | {r["tipo_incorrecto"]} | '
                f'**{r["sin_retroceso"]}/105** | {100*a:.0f}–{100*b:.0f} % |')
    L = ['# Clasificador de tipo con rasgos de peine: resultados', '',
         '105 registros de 12 kHz con referencia del paper. Sin retroceso = tipo correcto y grado ≥ paper.',
         'Banda, regla de grado y detector sano/falla son los del proyecto, sin cambios. No es validación externa.', '',
         '| Protocolo y receta | Tipo correcto | Mejoran | Iguales | Empeoran | Tipo incorrecto | Sin retroceso | IC 95 % |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
         linea('Repo (desarrollo OOF + congelado) · vigente', 'repo_vigente'), linea('Repo (desarrollo OOF + congelado) · nueva', 'repo_nuevo'),
         linea('Estricto (condición excluida, 105) · vigente', 'loco_vigente'), linea('Estricto (condición excluida, 105) · nueva', 'loco_nuevo')]
    if 'loco_anidado' in resumen:
        L.append(linea('Estricto anidado (receta elegida por pliegue)', 'loco_anidado'))
    p = resumen['pareadas']
    L += ['', f'Techo con tipo perfecto y la regla de grado actual: {techo["sin_retroceso"]}/105.',
          'IC 95 %: bootstrap que remuestrea las 28 condiciones completas.', '',
          f'Comparación pareada, protocolo estricto, sin retroceso: {p["loco_sin_retroceso"]["solo_nuevo"]} registros solo aciertan con la receta nueva y '
          f'{p["loco_sin_retroceso"]["solo_referencia"]} solo con la vigente (McNemar exacto, p = {p["loco_sin_retroceso"]["p_mcnemar_exacto"]:.3f}).',
          f'Tipo: {p["loco_tipo"]["solo_nuevo"]} frente a {p["loco_tipo"]["solo_referencia"]} (p = {p["loco_tipo"]["p_mcnemar_exacto"]:.3f}).', '',
          f'48 kHz con falla: {resumen["48k_con_falla"]["tipo_correcto"]}/{resumen["48k_con_falla"]["total"]} tipos correctos '
          f'({resumen["48k_con_falla"]["sin_diagnostico"]} sin duración suficiente). Normales: {resumen["normales"]["correctos"]}/{resumen["normales"]["total"]}.']
    (SALIDA/'resumen.md').write_text('\n'.join(L)+'\n', encoding='utf-8')
    print('\n'.join(L))
