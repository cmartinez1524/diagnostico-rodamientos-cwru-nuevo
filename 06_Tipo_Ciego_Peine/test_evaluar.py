"""Pruebas del evaluador que no necesitan los .mat: agrupación, métrica y exclusión por condición."""
import numpy as np
from evaluar import condicion, puntuar, por_grupos, pareada

assert condicion('12k_FE/OR007@3_2.mat') == '12k_FE/OR007@3' and condicion('12k_DE/B028_0.mat') == '12k_DE/B028'

# Métrica: un tipo equivocado nunca cuenta, aunque su grado iguale o supere al paper.
y = np.array(['B', 'IR', 'OR', 'OR'], dtype=object)
grados = [dict(B='D', IR='D', OR='D'), dict(B='D', IR='PD', OR='D'), dict(B='D', IR='D', OR='ND'), dict(B='ND', IR='ND', OR='PD')]
paper = np.array(['PD', 'PD', 'PD', 'ND'])
_, estado, r = puntuar(np.array(['B', 'IR', 'OR', 'B'], dtype=object), y, grados, paper)
assert list(estado) == ['mejora', 'igual', 'empeora', 'tipo_incorrecto']
assert (r['sin_retroceso'], r['tipo_correcto'], r['tipo_incorrecto']) == (2, 3, 1)

# Exclusión: la receta nunca recibe en entrenamiento un registro de la condición evaluada.
g = np.array(['a', 'a', 'b', 'b', 'c', 'c']); vistos = []


def espia(entreno, etiquetas, prueba):
    vistos.append((set(r['g'] for r in entreno), set(r['g'] for r in prueba)))
    return ['B']*len(prueba)


R = [dict(g=k) for k in g]
p = por_grupos(espia, R, np.array(['B']*6, dtype=object), g, np.arange(6))
assert len(vistos) == 3 and all(not (tr & te) and len(te) == 1 for tr, te in vistos) and list(p) == ['B']*6

c = pareada(np.array([True, False, False, True]), np.array([True, True, True, False]))
assert (c['solo_referencia'], c['solo_nuevo']) == (1, 2) and 0 < c['p_mcnemar_exacto'] <= 1
print('OK: condición, métrica sin aciertos por tipo equivocado, exclusión por condición y comparación pareada')
