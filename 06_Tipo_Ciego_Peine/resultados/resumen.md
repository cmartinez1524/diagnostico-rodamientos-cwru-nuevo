# Clasificador de tipo con rasgos de peine: resultados

105 registros de 12 kHz con referencia del paper. Sin retroceso = tipo correcto y grado ≥ paper.
Banda, regla de grado y detector sano/falla son los del proyecto, sin cambios. No es validación externa.

| Protocolo y receta | Tipo correcto | Mejoran | Iguales | Empeoran | Tipo incorrecto | Sin retroceso | IC 95 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Repo (desarrollo OOF + congelado) · vigente | 95/105 | 15 | 73 | 7 | 10 | **88/105** | 76–91 % |
| Repo (desarrollo OOF + congelado) · nueva | 96/105 | 15 | 73 | 8 | 9 | **88/105** | 75–92 % |
| Estricto (condición excluida, 105) · vigente | 87/105 | 10 | 69 | 8 | 18 | **79/105** | 62–87 % |
| Estricto (condición excluida, 105) · nueva | 92/105 | 10 | 73 | 9 | 13 | **83/105** | 66–90 % |
| Estricto anidado (receta elegida por pliegue) | 92/105 | 10 | 73 | 9 | 13 | **83/105** | 66–90 % |

Techo con tipo perfecto y la regla de grado actual: 95/105.
IC 95 %: bootstrap que remuestrea las 28 condiciones completas.

Comparación pareada, protocolo estricto, sin retroceso: 8 registros solo aciertan con la receta nueva y 4 solo con la vigente (McNemar exacto, p = 0.388).
Tipo: 9 frente a 4 (p = 0.267).

48 kHz con falla: 21/24 tipos correctos (1 sin duración suficiente). Normales: 4/4.
