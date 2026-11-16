"""Pruebas sin datos CWRU: señales sintéticas con verdad conocida."""
import inspect
import numpy as np
import peine
from peine import rasgos_peine, por_nodo, giro, COLUMNAS, NOMBRES

FS = 12000
FR = 29.95
MULT = dict(BPFI=5.4152, BPFO=3.5848, FTF=0.39828, BSF=2.3568)   # geometría DE configurada en el proyecto
CAT = {k: v*FR for k, v in MULT.items()}


def senal(frecuencia, portadora=3200., segundos=5, semilla=3, ruido=.5, armonicas=(1, 2, 3)):
    rng = np.random.default_rng(semilla)
    t = np.arange(int(segundos*FS))/FS
    m = sum(np.cos(2*np.pi*h*frecuencia*t)/h for h in armonicas)
    return (1+.5*m)*np.cos(2*np.pi*portadora*t)+ruido*rng.normal(size=len(t))


col = COLUMNAS.index
assert set(inspect.signature(rasgos_peine).parameters) == {'x', 'fs', 'catalogo'}, 'La interfaz no admite etiquetas ni frecuencia esperada'
assert abs(giro(CAT)-FR) < 1e-9
assert len(NOMBRES) == 93 and len(set(NOMBRES)) == 93

# 1. Una falla de pista interna con 1 % de deslizamiento: gana BPFI y se recupera el deslizamiento.
a = por_nodo(senal(CAT['BPFI']*1.01), FS, CAT)
mejor = {f: a[:, col(f+':peine_sin_eje')].max() for f in ('BPFI', 'BPFO', 'BSF')}
assert max(mejor, key=mejor.get) == 'BPFI' and mejor['BPFI'] > mejor['BPFO']+5, mejor
banda = int(a[:, col('BPFI:peine_sin_eje')].argmax())
assert abs(a[banda, col('BPFI:deslizamiento_pct')]-1.) <= .1+1e-9

# 2. Modulación al giro: 3, 6 y 9·fr caen a 1,7 % de las armónicas de BPFO del rodamiento FE; el peine sin eje las descarta.
mult_fe = dict(BPFI=4.9469, BPFO=3.0530, FTF=0.3817, BSF=1.9940)
cat_fe = {k: v*FR for k, v in mult_fe.items()}
b = por_nodo(senal(FR, armonicas=range(1, 10)), FS, cat_fe)
assert b[:, col('eje:peine')].max() > 15
assert b[:, col('BPFO:peine')].max() > b[:, col('BPFO:peine_sin_eje')].max()+3

# 3. Ruido blanco: ningún peine destaca y la salida es determinista.
r = np.random.default_rng(0).normal(size=5*FS)
c = por_nodo(r, FS, CAT)
assert c[:, col('BPFI:peine_sin_eje')].max() < mejor['BPFI']-5
assert np.array_equal(rasgos_peine(r, FS, CAT), rasgos_peine(r.copy(), FS, CAT))

# 4. 48 kHz se remuestrea a la misma geometría física; entradas inválidas se rechazan.
assert rasgos_peine(np.repeat(senal(CAT['BPFO']), 4)[:5*48000], 48000, CAT).shape == (93,)
for malo in (r[:1000], np.r_[r[:-1], np.nan]):
    try:
        rasgos_peine(malo, FS, CAT)
        raise AssertionError('Debió rechazar la entrada')
    except ValueError:
        pass
print('OK: BPFI con deslizamiento recuperado, armónica del eje descartada, ruido sin peine, 48 kHz y validaciones')
