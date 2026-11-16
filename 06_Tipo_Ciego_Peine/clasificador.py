"""Clasificador de tipo a ciegas: dos discriminantes lineales cuyos puntajes se suman.

1. `arbol`: los 1.953 rasgos del árbol del proyecto (`modelo_espectral.extraer`),
   en ventanas de 1 s con salto de 0,5 s sobre TODA la primera mitad de la señal.
2. `peine`: los 93 rasgos de `peine.rasgos_peine` sobre la misma primera mitad.

Cada discriminante es el LDA regularizado ya usado en el proyecto. Sus puntajes
se tipifican con la media y la desviación del propio entrenamiento antes de
sumarlos, para que ninguno domine por escala. La segunda mitad de la señal no
interviene: queda reservada para verificar la banda, como en `modelo_por_dominio`.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.signal import resample_poly
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent/'05_Diagnostico_Ciego'))
from modelo_espectral import extraer  # noqa: E402
from peine import rasgos_peine  # noqa: E402

TIPOS = ('B', 'IR', 'OR')
VISTAS = ('arbol', 'peine')


def primera_mitad(x, fs):
    if fs not in (12000, 48000) or len(x)//2 < 2.5*fs:
        raise ValueError('Se necesita una primera mitad de al menos 2,5 s a 12 o 48 kHz')
    return np.asarray(x[:len(x)//2], dtype=float)


def vistas(x, fs, catalogo):
    """Rasgos de las dos vistas; solo usa la primera mitad de la señal."""
    z = primera_mitad(x, fs)
    w = resample_poly(z, 1, 4, padtype='line') if fs == 48000 else z
    arbol = np.asarray([extraer(w[i:i+12000], 12000, catalogo, L=4)['arbol'] for i in range(0, len(w)-12000+1, 6000)])
    return dict(arbol=arbol, peine=rasgos_peine(z, fs, catalogo)[None])


def nuevo_lda():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))


class Ensamble:
    def __init__(self, nombres=VISTAS):
        self.nombres = tuple(nombres)

    def ajustar(self, registros, etiquetas):
        """registros: lista de dict vista -> matriz (ventanas × rasgos); etiquetas: tipo por registro."""
        self.partes = {}
        with threadpool_limits(limits=2):
            for v in self.nombres:
                X = np.vstack([r[v] for r in registros])
                y = np.concatenate([[e]*len(r[v]) for r, e in zip(registros, etiquetas)])
                clf = nuevo_lda().fit(X, y)
                s = clf.decision_function(X)
                assert tuple(clf.classes_) == TIPOS
                self.partes[v] = (clf, s.mean(axis=0), s.std(axis=0))
        return self

    def puntajes_por_vista(self, registro):
        return {v: ((clf.decision_function(registro[v])-mu)/sd).mean(axis=0) for v, (clf, mu, sd) in self.partes.items()}

    def puntajes(self, registro):
        return sum(self.puntajes_por_vista(registro).values())

    def predecir(self, registro):
        return TIPOS[int(self.puntajes(registro).argmax())]

    def exportar(self, ruta, **meta):
        datos = dict(nombres=np.array(self.nombres), tipos=np.array(TIPOS), **{k: np.array(v) for k, v in meta.items()})
        for v, (clf, mu, sd) in self.partes.items():
            sc, lda = clf.steps[0][1], clf.steps[1][1]
            datos.update({f'{v}_media': sc.mean_, f'{v}_escala': sc.scale_, f'{v}_coef': lda.coef_,
                          f'{v}_intercepto': lda.intercept_, f'{v}_mu': mu, f'{v}_sd': sd})
        np.savez_compressed(ruta, **datos)


def puntajes_exportados(ruta, registro):
    """Inferencia con NumPy desde los pesos exportados (sin pickle ni scikit-learn)."""
    with np.load(ruta, allow_pickle=False) as m:
        total = np.zeros(len(m['tipos']))
        for v in m['nombres']:
            v = str(v)
            s = ((registro[v]-m[f'{v}_media'])/m[f'{v}_escala'])@m[f'{v}_coef'].T+m[f'{v}_intercepto']
            total += ((s-m[f'{v}_mu'])/m[f'{v}_sd']).mean(axis=0)
        return total, [str(t) for t in m['tipos']]
