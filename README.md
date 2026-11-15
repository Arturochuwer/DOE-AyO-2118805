
URL del repositorio: https://github.com/Arturochuwer/DOE-AyO-2118805.git

## Estructura

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── tsp.py                       # algoritmos TSP: MST, 2-opt, NN, Simulated Annealing
│   ├── common.py                    # rutas de salida
│   ├── ex1_speedup.py
│   ├── ex2_knapsack.py
│   ├── ex3_two_opt.py
│   ├── ex4_simulated_annealing.py
│   ├── ex5_approx_ratio.py
│   ├── tarea3_piloto.py             # Tarea 3: estudio piloto (rango de T e iteraciones)
│   ├── tarea3_doe.py                # Tarea 3: DOE de la temperatura del SA (opciones A y B)
│   └── run_all.py                   # corre los 5 ejercicios
├── results/
│   └── tarea3/                      # Tarea 3: figuras, datos (.csv), logs y tablas resumen
└── report/
    └── 2118805_REPORTE_TAREA3.pdf                  # análisis y respuestas
```

## Cómo ejecutar (Python 3.10+)

```bash
pip install -r requirements.txt
python src/run_all.py                  # los 5 ejercicios (~1 min)
# o uno por uno, siempre desde la raíz del repositorio:
python src/ex1_speedup.py
python src/ex2_knapsack.py
python src/ex3_two_opt.py
python src/ex4_simulated_annealing.py
python src/ex5_approx_ratio.py
```

## Qué viene del material y qué se escribió

| Algoritmo | Fuente |
|---|---|
| `estimate_pi`, DP del knapsack, `two_opt_move`, `mst_prim`, `preorder`, `tour_length`, fuerza bruta | PDF / `p2.py` (copiados) |
| Pseudocódigo de Simulated Annealing | PDF (solo pseudocódigo); la implementación es propia |
| Voraz por menor peso, Nearest Neighbour, 2-opt completo, esquemas de enfriamiento | Escritos para esta tarea |

## Tarea 3 (DOE de la temperatura del Recocido Simulado)

```bash
pip install -r requirements.txt
python src/tarea3_piloto.py                          # estudio piloto (justifica el rango de T)
python src/tarea3_doe.py --matricula 2118805    # Opcion A y B (usa --opcion A o B para una sola)
```
Todo se guarda en `results/tarea3/`. Reporte: `report/2118805_REPORTE_TAREA3.pdf`.
