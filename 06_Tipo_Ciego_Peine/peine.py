"""Rasgos de peine con deslizamiento para identificar el tipo de falla a ciegas.

Entradas: señal, frecuencia de muestreo y catálogo completo del rodamiento en Hz
(BPFI, BPFO, BSF, FTF). No recibe el tipo real, la frecuencia esperada, RPM,
etiquetas ni el nombre del archivo. La velocidad de giro se deduce del propio
catálogo y de la geometría configurada, igual que `dominio_catalogo`.

Idea. En cada una de las 31 bandas del árbol (L = 4) se blanquea el espectro de
la envolvente y, para cada familia del catálogo, se busca el deslizamiento
s ∈ [−2 %, +2 %] que mejor alinea un peine de hasta seis armónicas h·f·(1+s)
con tolerancia estrecha (±0,3 %). Una falla real desplaza todas sus armónicas
en la misma proporción; una coincidencia con una armónica del eje no. Por eso
el peine se mide dos veces: tal cual y con las armónicas del eje k·fr anuladas.
Se añaden bandas laterales a ±fr y ±FTF y el peine del propio eje.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.fft import rfft, rfftfreq, next_fast_len
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'05_Diagnostico_Ciego'))
from modelo_ciego import F, validar_catalogo  # noqa: E402
from bandas_estables import envolvente  # noqa: E402
from modelo_por_dominio import dominio_catalogo  # noqa: E402

NODOS = [(l, m) for l in range(5) for m in range(2**l)]
FAMILIAS = ('BPFI', 'BPFO', 'BSF', '2BSF', 'FTF')
DESLIZAMIENTOS = np.linspace(-.02, .02, 41)
POR_FAMILIA = ('peine', 'peine_sin_eje', 'deslizamiento_pct', 'laterales_fr', 'laterales_ftf', 'fundamental_sin_eje')
COLUMNAS = [f'{f}:{c}' for f in FAMILIAS for c in POR_FAMILIA]+['eje:peine']
NOMBRES = [f'{a}:{c}' for a in ('max', 'top3', 'media') for c in COLUMNAS]
TECHO_DB = 30.


def giro(catalogo):
    """Frecuencia de giro deducida del catálogo completo; no usa RPM ni etiquetas."""
    m = F.MULTIPLICADORES[dominio_catalogo(catalogo)]
    return float(np.median([catalogo[k]/m[k] for k in m]))


def blanquear(a, fs, ancho_hz=12.):
    """Espectro Hann de la envolvente, en dB sobre la mediana local."""
    a = a-a.mean()
    n = next_fast_len(len(a))
    s = abs(rfft(a*np.hanning(len(a)), n=n))**2
    f = rfftfreq(n, 1/fs)
    df = f[1]-f[0]
    fondo = median_filter(s, size=int(round(ancho_hz/df)) | 1, mode='nearest')
    return 10*np.log10(np.maximum(s, 1e-300)/np.maximum(fondo, 1e-300)), df


def picos(db, df, frecuencias, tolerancia=.003):
    salida = np.zeros(len(frecuencias))
    for i, fq in enumerate(frecuencias):
        d = max(tolerancia*fq, 1.5*df)
        lo, hi = int(np.floor((fq-d)/df)), int(np.ceil((fq+d)/df))+1
        if lo >= 1 and hi <= len(db):
            salida[i] = db[lo:hi].max()
    return np.clip(salida, 0, TECHO_DB)


def por_nodo(x, fs, catalogo, armonicas=6, fmax=700.):
    """Matriz 31 × 31: una fila por banda, columnas en `COLUMNAS`."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or fs not in (12000, 48000) or len(x) < 2.5*fs:
        raise ValueError('Se requiere señal finita de al menos 2,5 s a 12 o 48 kHz')
    validar_catalogo(catalogo, fs)
    if fs == 48000:
        x, fs = resample_poly(x, 1, 4, padtype='line'), 12000
    fr = giro(catalogo)
    bases = {k: (catalogo[k] if k != '2BSF' else 2*catalogo['BSF']) for k in FAMILIAS}
    filas = []
    for nodo in NODOS:
        db, df = blanquear(envolvente(x, fs, nodo), fs)
        limite = min(fmax, fs/2**(nodo[0]+1))
        sin_eje = db.copy()
        for k in range(1, int(limite/fr)+2):
            d = max(.004*k*fr, 2*df)
            sin_eje[max(int((k*fr-d)/df), 0):int((k*fr+d)/df)+2] = 0.
        fila = []
        for nombre in FAMILIAS:
            base = bases[nombre]
            hs = np.arange(1, armonicas+1)
            hs = hs[hs*base*1.02 <= limite]
            if not len(hs):
                fila += [0.]*len(POR_FAMILIA)
                continue
            crudo = [picos(db, df, hs*base*(1+s)).mean() for s in DESLIZAMIENTOS]
            limpio = [picos(sin_eje, df, hs*base*(1+s)).mean() for s in DESLIZAMIENTOS]
            empatados = np.flatnonzero(np.asarray(limpio) >= max(limpio)-1e-9)
            j = int(empatados[len(empatados)//2])      # centro de la meseta: la tolerancia hace equivalentes varios s
            b = base*(1+DESLIZAMIENTOS[j])
            h3 = hs[:3]
            fila += [max(crudo), max(limpio), 100*DESLIZAMIENTOS[j],
                     picos(db, df, np.r_[h3*b-fr, h3*b+fr]).mean(),
                     picos(db, df, np.r_[h3*b-catalogo['FTF'], h3*b+catalogo['FTF']]).mean(),
                     picos(sin_eje, df, [b])[0]]
        ks = np.arange(1, 11)
        ks = ks[ks*fr <= limite]
        fila.append(picos(db, df, ks*fr, .002).mean() if len(ks) else 0.)
        filas.append(fila)
    return np.asarray(filas)


def rasgos_peine(x, fs, catalogo):
    """93 valores: máximo, media de las tres mejores bandas y media, por columna."""
    a = por_nodo(x, fs, catalogo)
    v = np.r_[a.max(axis=0), np.sort(a, axis=0)[-3:].mean(axis=0), a.mean(axis=0)]
    assert v.shape == (len(NOMBRES),) and np.isfinite(v).all()
    return v
