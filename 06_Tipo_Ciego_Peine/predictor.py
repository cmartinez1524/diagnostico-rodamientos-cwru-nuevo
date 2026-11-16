"""Diagnóstico a ciegas con el clasificador de tipo nuevo y pesos congelados.

    from predictor import predecir
    resultado = predecir(x, fs, catalogo_completo)

Recibe señal, muestreo y catálogo completo en Hz. No recibe etiquetas ni frecuencia esperada.
El detector sano/falla, la banda y el grado son los de `05_Diagnostico_Ciego`.
"""
from pathlib import Path
import hashlib
import sys
import numpy as np

AQUI = Path(__file__).resolve().parent
C5 = AQUI.parent/'05_Diagnostico_Ciego'
sys.path.insert(0, str(C5))
from modelo_por_dominio import calidad, dominio_catalogo  # noqa: E402
from ventanas_verificadas import rasgos  # noqa: E402
from modelo_jerarquico import aplicar  # noqa: E402
from clasificador import vistas, puntajes_exportados  # noqa: E402

MODELO = AQUI/'resultados/modelo_tipo.npz'


def predecir(x, fs, catalogo, *, modelo=MODELO):
    with np.load(modelo, allow_pickle=False) as m:
        for archivo in ('peine.py', 'clasificador.py'):
            if str(m[archivo]) != hashlib.sha256((AQUI/archivo).read_bytes()).hexdigest():
                raise ValueError(f'{archivo} cambió desde el entrenamiento; vuelva a ejecutar evaluar.py')
    dominio = dominio_catalogo(catalogo)
    puerta = aplicar(rasgos(x, fs, catalogo).mean(axis=0), C5/'resultados_por_dominio'/f'modelo_{dominio}.npz')
    if puerta == 'N':
        tipo, puntajes = 'N', None
    else:
        s, tipos = puntajes_exportados(modelo, vistas(x, fs, catalogo))
        tipo, puntajes = tipos[int(s.argmax())], dict(zip(tipos, map(float, s)))
    return dict(dominio=dominio, decision_detector=puerta, puntajes_tipo=puntajes, **calidad(x, fs, catalogo, tipo))
