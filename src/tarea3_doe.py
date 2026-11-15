"""Tarea 3: ajuste de la temperatura del Recocido Simulado (2-opt, TSP) con Diseno de Experimentos.

Opcion A: factorial completo 3 niveles x 3 tamanos, 10 replicas, ANOVA de dos factores + bootstrap.
Opcion B: RSM (diseno central compuesto de 1 factor), 5 niveles (5/5/10/5/5 replicas), regresion cuadratica.

Uso (desde la raiz del repositorio):
    python src/tarea3_doe.py --matricula 1234567                 # corre A y B
    python src/tarea3_doe.py --matricula 1234567 --opcion A      # solo A
    python src/tarea3_doe.py --matricula 1234567 --opcion B      # solo B

DEL PDF/Tarea 2 (reutilizado): simulated_annealing, schedule exponencial, random_tour, tour_length (src/tsp.py).
ESCRITO PARA LA TAREA: niveles con r, diseno, ANOVA, bootstrap, regresion cuadratica, falta de ajuste, graficas.
"""
import argparse
import math
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from common import ROOT
from tsp import SCHEDULES, random_points, random_tour, simulated_annealing, tour_length

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "tarea3"
OUT.mkdir(parents=True, exist_ok=True)

TAMANOS = (5, 10, 15)          # bloque: tamano del problema (una instancia por tamano)
TF = 0.01                      # temperatura final fija; el factor es la temperatura INICIAL T0
ORDEN_A = ["bajo", "medio", "alto"]
ORDEN_B = ["axial bajo", "bajo", "centro", "alto", "axial alto"]
COL = {5: "#2ca02c", 10: "#1f77b4", 15: "#d62728"}


class Reporte:
    """Imprime y guarda todo lo que sale en pantalla (para pegarlo en el reporte)."""
    def __init__(self, archivo):
        self.archivo, self.lineas = archivo, []

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.lineas.append(s)

    def guardar(self):
        self.archivo.write_text("\n".join(self.lineas), encoding="utf-8")


# ------------------------------- utilidades -------------------------------
def held_karp(pts):
    """Optimo exacto (programacion dinamica Held-Karp) solo como referencia de calidad."""
    n = len(pts)
    d = [[math.dist(a, b) for b in pts] for a in pts]
    INF, full = float("inf"), 1 << (n - 1)
    dp = [[INF] * (n - 1) for _ in range(full)]
    for j in range(n - 1):
        dp[1 << j][j] = d[0][j + 1]
    for m in range(1, full):
        for j in range(n - 1):
            if not (m >> j) & 1 or dp[m][j] == INF:
                continue
            for k in range(n - 1):
                if (m >> k) & 1:
                    continue
                v = dp[m][j] + d[j + 1][k + 1]
                if v < dp[m | 1 << k][k]:
                    dp[m | 1 << k][k] = v
    return min(dp[full - 1][j] + d[j + 1][0] for j in range(n - 1))


def fracciones(r, opcion):
    if opcion == "A":
        return {"bajo": 0.3 * r, "medio": 0.3 * r + 0.3, "alto": 0.3 * r + 0.6}
    return {"axial bajo": 0.2 * r, "bajo": 0.2 * r + 0.2, "centro": 0.2 * r + 0.4,
            "alto": 0.2 * r + 0.6, "axial alto": 0.2 * r + 0.8}


def correr_sa(pts, T0, semilla, n_iter):
    n = len(pts)
    rng = random.Random(semilla)
    sched = SCHEDULES["Exponencial"](T0, TF, n_iter)
    tour, _, _ = simulated_annealing(pts, random_tour(n, rng), sched, rng, record=False)
    return tour_length(pts, tour)


def experimento(opcion, r, tmin, tmax, n_iter, instancias):
    niv = {k: tmin + f * (tmax - tmin) for k, f in fracciones(r, opcion).items()}
    reps = {k: 10 for k in niv} if opcion == "A" else {k: (10 if k == "centro" else 5) for k in niv}
    filas = []
    for n in TAMANOS:
        for idx, (k, T0) in enumerate(niv.items()):
            for rep in range(reps[k]):
                semilla = 1_000_000 * n + 1_000 * idx + rep          # semilla distinta por corrida
                filas.append(dict(tamano=n, nivel=k, temp=T0, rep=rep,
                                  costo=correr_sa(instancias[n], T0, semilla, n_iter)))
    return pd.DataFrame(filas), niv


def boot_mediana(x, K=2000, seed=0):
    """Mediana bootstrap = mediana de las K medianas remuestreadas; IC 95% por percentiles 2.5 y 97.5."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(K, len(x)), replace=True), axis=1)
    return float(np.median(meds)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def ic_media(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std() == 0:
        return float(x.mean()), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def preparar(args):
    random.seed(args.matricula)
    r = random.random()                         # UN solo r para todo el experimento
    assert args.tmin > TF, "T_min debe ser mayor que la temperatura final fija TF"
    instancias = {n: random_points(n, random.Random(args.matricula + n)) for n in TAMANOS}   # una instancia por tamano (depende de TU matricula)
    opt = {n: held_karp(p) for n, p in instancias.items()}
    return r, instancias, opt


def encabezado(P, args, r, opt, niv, opcion, titulo):
    P(f"=== TAREA 3 - OPCION {titulo} ===")
    P(f"matricula (semilla) = {args.matricula}   ->   r = {r:.6f}")
    P(f"Rango de temperatura [T_min, T_max] = [{args.tmin}, {args.tmax}]   |   iteraciones SA = {args.iter}   |   T final fija = {TF}")
    P("Niveles (fraccion -> temperatura real):")
    for k, f in fracciones(r, opcion).items():
        P(f"   {k:<11} fraccion = {f:.4f}   T = {niv[k]:.4f}")
    P("Optimo exacto de cada instancia (Held-Karp, solo referencia): "
      + ", ".join(f"n={n}: {v:.2f}" for n, v in opt.items()))
    P()


# =================================== OPCION A ===================================
def opcion_A(args, r, instancias, opt):
    P = Reporte(OUT / "log_opcion_A.txt")
    df, niv = experimento("A", r, args.tmin, args.tmax, args.iter, instancias)
    df["rel"] = df.costo / df.tamano.map(opt)            # costo relativo al optimo (analisis complementario)
    df.to_csv(OUT / "datos_opcion_A.csv", index=False)
    encabezado(P, args, r, opt, niv, "A", "A (factorial completo 3x3, 10 replicas = 90 corridas)")

    # ---- ANOVA de dos factores ----
    modelo = ols("costo ~ C(nivel) * C(tamano)", data=df).fit()
    tabla = sm.stats.anova_lm(modelo, typ=2)
    tabla["eta2_parcial"] = tabla["sum_sq"] / (tabla["sum_sq"] + tabla.loc["Residual", "sum_sq"])
    P("ANOVA de dos factores: costo ~ temperatura * tamano   (alpha = 0.05)")
    P(tabla.round(4).to_string())
    pt = tabla.loc["C(nivel)", "PR(>F)"]
    ps = tabla.loc["C(tamano)", "PR(>F)"]
    pi_ = tabla.loc["C(nivel):C(tamano)", "PR(>F)"]
    P(f"\n  temperatura: p = {pt:.4g} -> {'SIGNIFICATIVO' if pt < 0.05 else 'no significativo'}")
    P(f"  tamano:      p = {ps:.4g} -> {'SIGNIFICATIVO' if ps < 0.05 else 'no significativo'}")
    P(f"  interaccion: p = {pi_:.4g} -> {'SIGNIFICATIVO' if pi_ < 0.05 else 'no significativo'}")
    # supuestos (informativo)
    sh = stats.shapiro(modelo.resid)
    grupos = [g.costo.values for _, g in df.groupby(["tamano", "nivel"])]
    lev = stats.levene(*grupos)
    P(f"\n  Supuestos: Shapiro-Wilk de residuos p = {sh.pvalue:.4g} | Levene (9 celdas) p = {lev.pvalue:.4g}")
    P("  (si son < 0.05 el ANOVA es solo aproximado; por eso se refuerza con bootstrap, que no exige normalidad)")

    # ---- ANOVA complementario con costo relativo al optimo ----
    m2 = ols("rel ~ C(nivel) * C(tamano)", data=df).fit()
    t2 = sm.stats.anova_lm(m2, typ=2)
    P("\nANOVA complementario con costo/optimo (quita el efecto de escala entre tamanos):")
    P(t2.round(4).to_string())

    # ---- Analisis por tamano: ANOVA de un factor, Tukey y bootstrap ----
    P("\n--- Analisis por tamano ---")
    filas = []
    for n in TAMANOS:
        d = df[df.tamano == n]
        P(f"\nn = {n}")
        sin_var = d.costo.nunique() == 1
        if sin_var:
            p_n = float("nan")
            P(f"  Todas las 30 corridas dieron exactamente {d.costo.iloc[0]:.4f} (= optimo {opt[n]:.4f}): "
              "EFECTO PISO, la temperatura no puede distinguirse.")
        else:
            F, p_n = stats.f_oneway(*[d[d.nivel == k].costo.values for k in ORDEN_A])
            P(f"  ANOVA de un factor (temperatura): F = {F:.3f}, p = {p_n:.4g} -> "
              f"{'significativo' if p_n < 0.05 else 'no significativo'}")
            tk = pairwise_tukeyhsd(d.costo, d.nivel, alpha=0.05)
            P("  Tukey HSD:\n" + "\n".join("    " + l for l in str(tk).splitlines()))
        bs = {}
        for i, k in enumerate(ORDEN_A):
            bs[k] = boot_mediana(d[d.nivel == k].costo, seed=n * 10 + i)
            P(f"  {k:<6} T = {niv[k]:.3f}   mediana bootstrap = {bs[k][0]:.4f}   IC95% = [{bs[k][1]:.4f}, {bs[k][2]:.4f}]")
        mejor = min(ORDEN_A, key=lambda k: (bs[k][0], niv[k]))
        solapan = [k for k in ORDEN_A if k != mejor and bs[k][1] <= bs[mejor][2]]
        if sin_var:
            concl = "Sin diferencias (efecto piso); cualquier T alcanza el optimo"
        elif p_n >= 0.05:
            concl = "Diferencias no significativas; se elige la de menor mediana"
        elif solapan:
            concl = f"Significativo, pero su IC se traslapa con: {', '.join(solapan)}"
        else:
            concl = "Mejor nivel claramente separado (IC sin traslape)"
        P(f"  => mejor nivel: {mejor} (T = {niv[mejor]:.3f}). {concl}")
        filas.append(dict(n=n, mejor=mejor, T=niv[mejor], med=bs[mejor][0], lo=bs[mejor][1], hi=bs[mejor][2],
                          p=p_n, concl=concl))
    res = pd.DataFrame(filas)
    res.to_csv(OUT / "tabla_resumen_A.csv", index=False)
    md = ["| n | Mejor T (nivel) | Mediana bootstrap | IC 95% | p (ANOVA por tamano) | Conclusion |", "|---|---|---|---|---|---|"]
    for _, f in res.iterrows():
        pv = "n/a (sin variacion)" if math.isnan(f.p) else f"{f.p:.4g}"
        md.append(f"| {f.n} | {f['T']:.3f} ({f.mejor}) | {f.med:.4f} | [{f.lo:.4f}, {f.hi:.4f}] | {pv} | {f.concl} |")
    (OUT / "tabla_resumen_A.md").write_text("\n".join(md), encoding="utf-8")
    P("\nTABLA RESUMEN (Opcion A)\n" + "\n".join(md))
    P.guardar()

    # ---- Figuras ----
    # 1) Efectos principales
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # IC 95% con el error residual del ANOVA (MSE): no mezcla la variacion entre tamanos
    tcrit = stats.t.ppf(0.975, modelo.df_resid)
    semi = lambda g: float(tcrit * math.sqrt(modelo.mse_resid / len(g)))
    mm = [(df[df.nivel == k].costo.mean(), semi(df[df.nivel == k])) for k in ORDEN_A]
    ax[0].errorbar([niv[k] for k in ORDEN_A], [m for m, _ in mm], yerr=[e for _, e in mm], fmt="o-", capsize=4, color="#1f77b4")
    ax[0].set_xlabel("Temperatura inicial T0")
    ax[0].set_ylabel("Costo medio (longitud del tour)")
    ax[0].set_title(f"Efecto principal: temperatura (p = {pt:.3g})")
    ms = [(df[df.tamano == n].costo.mean(), semi(df[df.tamano == n])) for n in TAMANOS]
    ax[1].errorbar(list(TAMANOS), [m for m, _ in ms], yerr=[e for _, e in ms], fmt="s-", capsize=4, color="#d62728")
    ax[1].set_xticks(list(TAMANOS))
    ax[1].set_xlabel("Tamano n (ciudades)")
    ax[1].set_ylabel("Costo medio (longitud del tour)")
    ax[1].set_title(f"Efecto principal: tamano (p = {ps:.3g})")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Opcion A: efectos principales (media e IC 95% con el error del ANOVA)")
    fig.tight_layout()
    fig.savefig(OUT / "tarea3_A_efectos_principales.png", dpi=150)
    plt.close(fig)
    # 2) Interaccion temperatura x tamano (costo crudo y costo relativo al optimo)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for j, (col, etq) in enumerate((("costo", "Costo medio"), ("rel", "Costo medio / optimo"))):
        for n in TAMANOS:
            mm = [ic_media(df[(df.tamano == n) & (df.nivel == k)][col]) for k in ORDEN_A]
            ax[j].errorbar([niv[k] for k in ORDEN_A], [m for m, _ in mm], yerr=[e for _, e in mm],
                           fmt="o-", capsize=3, color=COL[n], label=f"n = {n}")
        ax[j].set_xlabel("Temperatura inicial T0")
        ax[j].set_ylabel(etq)
        ax[j].grid(alpha=0.3)
        ax[j].legend(loc="best")
    ax[0].set_title(f"Interaccion temperatura x tamano (p = {pi_:.3g})")
    ax[1].set_title("Misma grafica en escala relativa al optimo")
    fig.tight_layout()
    fig.savefig(OUT / "tarea3_A_interaccion.png", dpi=150)
    plt.close(fig)
    # 3) Boxplots por tamano
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, n in zip(ax, TAMANOS):
        d = df[df.tamano == n]
        a.boxplot([d[d.nivel == k].costo for k in ORDEN_A], tick_labels=[f"{k}\nT={niv[k]:.2f}" for k in ORDEN_A])
        a.axhline(opt[n], color="gray", ls=":", label="optimo exacto")
        a.set_title(f"n = {n}")
        a.set_ylabel("Costo final")
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle("Opcion A: distribucion del costo por nivel de temperatura (10 replicas)")
    fig.tight_layout()
    fig.savefig(OUT / "tarea3_A_boxplots.png", dpi=150)
    plt.close(fig)


# =================================== OPCION B ===================================
def opcion_B(args, r, instancias, opt):
    P = Reporte(OUT / "log_opcion_B.txt")
    df, niv = experimento("B", r, args.tmin, args.tmax, args.iter, instancias)
    df.to_csv(OUT / "datos_opcion_B.csv", index=False)
    encabezado(P, args, r, opt, niv, "B", "B (RSM, 5 niveles, replicas 5/5/10/5/5 = 30 corridas por tamano)")
    Tlo, Thi = niv["axial bajo"], niv["axial alto"]
    grid = np.linspace(Tlo, Thi, 200)
    filas = []
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, n in zip(ax, TAMANOS):
        d = df[df.tamano == n].copy()
        P(f"--- n = {n} (instancia con optimo exacto {opt[n]:.4f}) ---")
        bs = {k: boot_mediana(d[d.nivel == k].costo, seed=n * 10 + i) for i, k in enumerate(ORDEN_B)}
        for k in ORDEN_B:
            P(f"  {k:<11} T = {niv[k]:.3f}   mediana bootstrap = {bs[k][0]:.4f}   IC95% = [{bs[k][1]:.4f}, {bs[k][2]:.4f}]")
        mejor_obs = min(ORDEN_B, key=lambda k: (bs[k][0], niv[k]))
        a.scatter(d.temp + np.random.default_rng(n).normal(0, 0.02, len(d)), d.costo, s=14, alpha=0.5, color="gray", label="corridas")
        a.errorbar([niv[k] for k in ORDEN_B], [bs[k][0] for k in ORDEN_B],
                   yerr=[[bs[k][0] - bs[k][1] for k in ORDEN_B], [bs[k][2] - bs[k][0] for k in ORDEN_B]],
                   fmt="D", color=COL[n], capsize=4, label="mediana bootstrap + IC95%")
        if d.costo.nunique() == 1:
            P(f"  Las 30 corridas dieron {d.costo.iloc[0]:.4f} (= optimo): EFECTO PISO, no hay curva que ajustar ni T* util.")
            filas.append(dict(n=n, Tstar=float("nan"), Trec=float("nan"), pb2=float("nan"), plof=float("nan"),
                              mejor=mejor_obs, med=bs[mejor_obs][0], lo=bs[mejor_obs][1], hi=bs[mejor_obs][2],
                              concl="Efecto piso: cualquier T da el optimo"))
            a.set_title(f"n = {n}: efecto piso (todas = optimo)")
            a.set_xlabel("Temperatura inicial T0")
            a.set_ylabel("Costo final")
            a.legend()
            a.grid(alpha=0.3)
            P()
            continue
        mq = ols("costo ~ temp + I(temp**2)", data=d).fit()
        b0, b1, b2 = mq.params["Intercept"], mq.params["temp"], mq.params["I(temp ** 2)"]
        pb2 = mq.pvalues["I(temp ** 2)"]
        P(f"  Modelo: Costo = {b0:.4f} + {b1:.4f}*T + {b2:.4f}*T^2     R^2 = {mq.rsquared:.3f}")
        P(f"  p(beta2) = {pb2:.4g} -> curvatura {'SIGNIFICATIVA' if pb2 < 0.05 else 'no significativa'};   "
          f"ANOVA de regresion: F = {mq.fvalue:.3f}, p = {mq.f_pvalue:.4g}")
        # falta de ajuste (error puro vs. falta de ajuste)
        mc = ols("costo ~ C(nivel)", data=d).fit()
        lof = sm.stats.anova_lm(mq, mc)
        Flof, plof = float(lof.loc[1, "F"]), float(lof.loc[1, "Pr(>F)"])
        P(f"  Falta de ajuste: F = {Flof:.3f}, p = {plof:.4g} -> modelo cuadratico "
          f"{'describe bien los datos' if plof >= 0.05 else 'NO basta (falta de ajuste significativa)'}")
        # optimo de la parabola
        if b2 > 1e-12:
            Tstar = -b1 / (2 * b2)
            en_rango = Tlo <= Tstar <= Thi
            Trec = min(max(Tstar, Tlo), Thi)
            P(f"  beta2 > 0 (parabola hacia arriba): T* = -b1/(2*b2) = {Tstar:.4f}"
              + ("" if en_rango else f"  (FUERA del rango probado [{Tlo:.3f}, {Thi:.3f}]; se usa {Trec:.3f})"))
        else:
            Tstar = float("nan")
            pred = mq.predict(pd.DataFrame({"temp": [Tlo, Thi]}))
            Trec = Tlo if pred.iloc[0] <= pred.iloc[1] else Thi
            P(f"  beta2 <= 0: no hay minimo interior; el menor costo ajustado esta en el extremo T = {Trec:.3f}")
        pred = mq.get_prediction(pd.DataFrame({"temp": grid})).summary_frame(alpha=0.05)
        a.plot(grid, pred["mean"], color=COL[n], label="ajuste cuadratico")
        a.fill_between(grid, pred["mean_ci_lower"], pred["mean_ci_upper"], color=COL[n], alpha=0.2, label="IC95% de la media ajustada")
        a.axvline(Trec, color="black", ls="--", lw=1, label=f"T recomendada = {Trec:.2f}")
        a.axhline(opt[n], color="gray", ls=":", lw=1, label="optimo exacto")
        a.set_title(f"n = {n}   (p b2 = {pb2:.3g}, p falta ajuste = {plof:.3g})")
        a.set_xlabel("Temperatura inicial T0")
        a.set_ylabel("Costo final")
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
        solapan = [k for k in ORDEN_B if k != mejor_obs and bs[k][1] <= bs[mejor_obs][2]]
        concl = ("Curvatura significativa" if pb2 < 0.05 else "Curvatura NO significativa (T* poco confiable)")
        if plof < 0.05:
            concl += "; falta de ajuste significativa"
        P(f"  Mejor nivel observado: {mejor_obs} (T = {niv[mejor_obs]:.3f}), mediana bootstrap = {bs[mejor_obs][0]:.4f} "
          f"IC95% [{bs[mejor_obs][1]:.4f}, {bs[mejor_obs][2]:.4f}]" + (f"; IC traslapado con {', '.join(solapan)}" if solapan else ""))
        P()
        filas.append(dict(n=n, Tstar=Tstar, Trec=Trec, pb2=pb2, plof=plof, mejor=mejor_obs, med=bs[mejor_obs][0],
                          lo=bs[mejor_obs][1], hi=bs[mejor_obs][2], concl=concl))
    fig.suptitle("Opcion B: curva cuadratica ajustada, puntos observados y bootstrap")
    fig.tight_layout()
    fig.savefig(OUT / "tarea3_B_curvas.png", dpi=150)
    plt.close(fig)
    res = pd.DataFrame(filas)
    res.to_csv(OUT / "tabla_resumen_B.csv", index=False)
    f = lambda x, fmt="{:.3f}": "n/a" if (isinstance(x, float) and math.isnan(x)) else fmt.format(x)
    md = ["| n | T* (ajuste) | T recomendada | p(beta2) | p(falta de ajuste) | Mejor nivel observado | Mediana bootstrap | IC 95% | Nota |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, g in res.iterrows():
        md.append(f"| {g.n} | {f(g.Tstar)} | {f(g.Trec)} | {f(g.pb2, '{:.4g}')} | {f(g.plof, '{:.4g}')} | "
                  f"{g.mejor} (T={niv[g.mejor]:.3f}) | {g.med:.4f} | [{g.lo:.4f}, {g.hi:.4f}] | {g.concl} |")
    (OUT / "tabla_resumen_B.md").write_text("\n".join(md), encoding="utf-8")
    P("TABLA RESUMEN (Opcion B)\n" + "\n".join(md))
    P.guardar()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matricula", type=int, default=1234567, help="semilla para r (TU matricula)")
    ap.add_argument("--opcion", choices=["A", "B", "both"], default="both")
    ap.add_argument("--tmin", type=float, default=0.05)
    ap.add_argument("--tmax", type=float, default=12.0)
    ap.add_argument("--iter", type=int, default=100, help="iteraciones del SA por corrida")
    args = ap.parse_args()
    r, instancias, opt = preparar(args)
    if args.opcion in ("A", "both"):
        opcion_A(args, r, instancias, opt)
        print()
    if args.opcion in ("B", "both"):
        opcion_B(args, r, instancias, opt)
