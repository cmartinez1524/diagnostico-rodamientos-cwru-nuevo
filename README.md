# Diagnóstico de rodamientos con CWRU

Proyecto de detección de anomalías de PUCV, segundo semestre de 2026.
Estudia MFD, kurtogram, entropía, estabilidad temporal y clasificación de fallas
de rodamientos. Incluye código, modelos numéricos, figuras, resultados y auditorías.

## Estado y resultados actuales

La última comparación cubre **133 registros**. Las matrices de cambios frente al
paper incluyen los **105 registros de 12 kHz con referencia transcrita**:

| Escenario | Mejoran | Mantienen | Empeoran | Tipo incorrecto | Sin retroceso |
| --- | ---: | ---: | ---: | ---: | ---: |
| Con familia/frecuencia esperada: oráculo | 21 | 74 | 10 | 0 | 95/105 (90,48 %) |
| Sin frecuencia esperada | 15 | 73 | 7 | 10 | 88/105 (83,81 %) |

![Comparación con y sin frecuencia esperada](05_Diagnostico_Ciego/comparacion_con_sin_esperada_todos/matrices_con_sin_esperada_105.png)

Las otras señales se informan separadamente: **21/24 tipos correctos a 48 kHz**
y **4/4 normales identificados** sin frecuencia esperada. Una señal de 48 kHz
no alcanza la duración mínima de cinco segundos y queda como sin diagnóstico;
no se elimina del denominador.

El oráculo recibe deliberadamente la familia verdadera antes de elegir la banda.
El método ciego recibe señal, frecuencia de muestreo y catálogo completo del
rodamiento, sin indicar cuál es la falla real. La banda se elige en la primera
mitad y se comprueba en la segunda.

**Estos números no son validación externa.** Se conservan predicciones fuera de
grupo en desarrollo y se aplica un modelo congelado al resto, con condiciones
compartidas. Los grados ND/PD/D siguen reglas propias y no están homologados
con la rúbrica del paper. La meta de al menos 90/105 sin retroceso **sin oráculo**
todavía no se alcanza.

Resultados por registro: [comparacion_133.csv](05_Diagnostico_Ciego/comparacion_con_sin_esperada_todos/comparacion_133.csv).

### Validación estricta y clasificador de tipo con rasgos de peine

El 88/105 anterior mezcla dos protocolos: 52 registros se predicen sin haber visto
su condición y 53 con un modelo que ya vio otras cargas de la misma falla. Al excluir
siempre la condición completa en los 105 registros, la misma receta baja a 79/105.
[`06_Tipo_Ciego_Peine`](06_Tipo_Ciego_Peine/README.md) agrega rasgos de peine con
deslizamiento al clasificador de tipo, sin cambiar la banda ni la regla de grado:

| Protocolo | Receta | Tipo correcto | Sin retroceso |
| --- | --- | ---: | ---: |
| Del proyecto (desarrollo fuera de grupo + congelado) | vigente | 95/105 | 88/105 |
| | con peine | 96/105 | 88/105 |
| Estricto (condición excluida en los 105) | vigente | 87/105 | 79/105 |
| | con peine | 92/105 | 83/105 |

La mejora en validación estricta no es estadísticamente concluyente (McNemar exacto,
p = 0,39) y la meta de 90/105 sigue sin alcanzarse. Con el tipo perfecto y la regla
de grado actual el techo es 95/105. Los intervalos de la comparación con y sin
frecuencia esperada están en
[incertidumbre.md](05_Diagnostico_Ciego/comparacion_con_sin_esperada_todos/incertidumbre.md).

## Instalación

Python 3.11 o posterior; el entorno original usa Python 3.13.

```shell
python -m venv .venv
python -m pip install -r requirements.txt
```

Active el entorno virtual antes de instalar o ejecutar. Jupyter es opcional
para los notebooks históricos: `python -m pip install jupyter`.

## Datos originales

Los archivos MATLAB del dataset, aproximadamente 536 MB incluyendo copias de
verificación, no se suben a Git. Sí se incluyen índices y registros de procedencia.
Descargue los datos del [Bearing Data Center de CWRU](https://engineering.case.edu/bearingdatacenter/download-data-file)
y ubíquelos en estas rutas, respetando los nombres de `indice_canal_verificado.csv`:

```text
data2/CWRU/
  12k_DE/    # 60 archivos
  12k_FE/    # 45 archivos
  48k_DE/    # 24 archivos
  Normal/    # 4 archivos
```

Fuentes: [12 kHz DE](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data),
[12 kHz FE](https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data),
[48 kHz DE](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data),
[normales](https://engineering.case.edu/bearingdatacenter/normal-baseline-data).

Algunos MAT contienen varios registros. **No seleccione la primera variable
que termine en `_DE_time` o `_FE_time`.** El lector corregido selecciona el
registro explícito y sus RPM correspondientes. El índice verificado conserva
133 señales distintas. Los identificadores internos pueden repetirse entre
archivos: la ruta también identifica la adquisición.

## Reproducir la comparación actual

Desde la raíz, después de disponer de los datos:

```shell
python 05_Diagnostico_Ciego/evaluar_comparacion_todos.py
python 05_Diagnostico_Ciego/verificar_comparacion_todos.py
```

No se reentrena ni se elige otro modelo mirando el total. Se emplean el modelo
de tipo en `resultados_jerarquico/` y los detectores DE/FE en
`resultados_por_dominio/`, dentro de `05_Diagnostico_Ciego/`.

Prueba del lector contra las copias descargadas de la fuente oficial:

```shell
python 05_Diagnostico_Ciego/recuperar_fuentes_cwru.py
python 05_Diagnostico_Ciego/test_selector_mat.py
```

`.gitattributes` conserva los bytes de los archivos porque los modelos verifican
hashes de código y artefactos. No normalice sus saltos de línea sin regenerar
los modelos y los manifiestos correspondientes.

## Organización e historial

| Carpeta | Contenido |
| --- | --- |
| `01_MFD_Kurtogram_Tarea1` | MFD, kurtogram y envolvente espectral |
| `02_Sinergia_Fraterna` | Métrica de estructura padre/hijos del árbol |
| `03_Comparacion_Kurtogram_vs_Sinergia` | Comparación histórica de métodos |
| `04_Comparacion_Kurtogram_vs_Entropia` | Entropía, filtros, doble banda y estabilidad |
| `05_Diagnostico_Ciego` | Inferencia sin frecuencia esperada, experimentos y auditorías |
| `06_Tipo_Ciego_Peine` | Clasificador de tipo con rasgos de peine y validación estricta por condición |

Se conservan resultados y pesos históricos para trazabilidad; no todos son
entradas vigentes. Algunos pesos anteriores están invalidados por las correcciones
de lectura y no deben usarse como resultados actuales. Las rutas de los primeros
experimentos pueden diferir: consulte sus instrucciones específicas y
[README_historico.md](README_historico.md).

Auditorías principales:

- [Selección de registros y RPM en MAT](05_Diagnostico_Ciego/auditoria_selector_mat/INFORME.md).
- [Auditoría de los modelos y criterios diagnósticos](05_Diagnostico_Ciego/auditoria_modelos_actuales/INFORME.md).
- [Protocolo de mejoras reales](05_Diagnostico_Ciego/PROTOCOLO_MEJORAS_REALES.md).
