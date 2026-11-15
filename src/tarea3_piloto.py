"""Estudio PILOTO de la Tarea 3 (McGeoch: 'pilot study' antes del 'workhorse').
Sirve para justificar el rango [T_min, T_max] y el numero de iteraciones del SA:
  - con presupuestos grandes el SA llega al optimo con cualquier T0 (efecto piso/techo) y no hay nada que comparar;
  - con presupuesto corto, la temperatura inicial si cambia la calidad.
Uso:  python src/tarea3_piloto.py --matricula TU_MATRICULA
Salida: results/tarea3/tarea3_piloto.png y results/tarea3/log_piloto.txt
"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tarea3_doe import COL, OUT, TAMANOS, Reporte, correr_sa, held_karp, ic_media
from tsp import random_points

TEMPS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 5, 8, 12]
PRESUPUESTOS = (100, 1000)
REPS = 20

if __name__ == "__main__":
    P = Reporte(OUT / "log_piloto.txt")
    ap = argparse.ArgumentParser()
    ap.add_argument("--matricula", type=int, default=1234567)
    args = ap.parse_args()
    inst = {n: random_points(n, random.Random(args.matricula + n)) for n in TAMANOS}
    opt = {n: held_karp(p) for n, p in inst.items()}
    fig, ax = plt.subplots(1, len(PRESUPUESTOS), figsize=(6 * len(PRESUPUESTOS), 4.4), sharey=False)
    for a, N in zip(ax, PRESUPUESTOS):
        P(f"\nPILOTO con {N} iteraciones por corrida ({REPS} replicas por punto) - brecha % sobre el optimo")
        P(f"{'T0':>6} " + " ".join(f"{'n=' + str(n):>10}" for n in TAMANOS))
        res = {n: [] for n in TAMANOS}
        for T in TEMPS:
            fila = []
            for n in TAMANOS:
                c = [correr_sa(inst[n], T, 9_000_000 + i, N) for i in range(REPS)]
                m, e = ic_media([100 * (x / opt[n] - 1) for x in c])
                res[n].append((m, e))
                fila.append(f"{m:9.1f}%")
            P(f"{T:>6} " + " ".join(fila))
        for n in TAMANOS:
            a.errorbar(TEMPS, [m for m, _ in res[n]], yerr=[e for _, e in res[n]], fmt="o-", capsize=3, color=COL[n], label=f"n = {n}")
        a.set_xscale("log")
        a.set_xlabel("Temperatura inicial T0 (escala log)")
        a.set_ylabel("Brecha media sobre el optimo (%)")
        a.set_title(f"Piloto: {N} iteraciones")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "tarea3_piloto.png", dpi=150)
    P.guardar()
    print("Figura: results/tarea3/tarea3_piloto.png")
