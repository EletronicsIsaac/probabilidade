#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relatório 1 - Análise Exploratória de Dados (Probabilidade para Engenharia)
Base: Real Controlled-Fault Benchmark for Power-Anomaly Detection on IoT Edge Nodes
      (Tran & Tran, Zenodo, DOI 10.5281/zenodo.20565892)

USO:
    python aed_relatorio1.py CAMINHO_DA_PASTA_COM_OS_CSV
    (a pasta pode conter os CSVs soltos ou em subpastas; se baixou o .zip, descompacte antes)

O script NÃO remove nenhuma observação. Ele apenas SINALIZA problemas e registra tudo.
Saídas em ./saida: figuras/*.png, tabelas/*.csv e *.tex, resumo.json, log_limpeza.csv
"""
import sys, re, json, glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)
OUT = "saida"
FIG, TAB = f"{OUT}/figuras", f"{OUT}/tabelas"
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------------
# 0. CONFIGURAÇÃO - se a detecção automática errar, preencha manualmente aqui
#    ex.: OVERRIDE = {"tensao": "voltage_V", "corrente": "current_A", ...}
# ----------------------------------------------------------------------------
OVERRIDE = {}
PATTERNS = {  # regex (minúsculas); a ordem importa: rótulos/tempo primeiro, depois grandezas
    "tempo":     r"time|date|^t$|stamp|^ts$",
    "anomalia":  r"^anomaly$|^is_anomaly$|^label$",
    "tipo":      r"anomaly_type|fault_type",
    "categoria": r"fault_category|category",
    "severidade": r"severity",
    "evento":    r"event_id",
    "fp":        r"(^pf$|power_?factor|fator|cos)",
    "tensao":    r"^(v|volt|voltage|vrms|tensao)(_?(v|rms))?$|volt",
    "corrente":  r"^(i|current|irms|corrente)(_?(a|rms))?$|curr|(^|_)amps?(_|$)",
    "potencia":  r"^(?!.*(factor|pf))(p|w|power|potencia|active_?power)(_?w)?$|^(?!.*(factor|pf))power|(?<!re)active",
    "energia":   r"energ|kwh|wh$",
    "frequencia": r"freq|^hz$",
}
# faixas físicas plausíveis (rede monofásica 127/220 V, 60 Hz Brasil / 50 Hz Vietnã: ajuste!)
FAIXAS = {"tensao": (0, 300), "corrente": (0, 100), "potencia": (0, 25000),
          "frequencia": (40, 70), "fp": (0, 1.0001)}
UNID = {"tensao": "V", "corrente": "A", "potencia": "W", "energia": "kWh",
        "frequencia": "Hz", "fp": "adim."}
ROTULO = {"tensao": "Tensão", "corrente": "Corrente", "potencia": "Potência ativa",
          "energia": "Energia", "frequencia": "Frequência", "fp": "Fator de potência"}

# ----------------------------------------------------------------------------
# 1. CARGA
# ----------------------------------------------------------------------------
def carregar(pasta):
    arqs = sorted(glob.glob(os.path.join(pasta, "**", "*.csv"), recursive=True))
    arqs = [a for a in arqs if "feature" not in os.path.basename(a).lower()] or arqs
    if not arqs:
        sys.exit("Nenhum CSV encontrado em " + pasta)
    dfs = []
    for a in arqs:
        d = pd.read_csv(a, low_memory=False)
        d["arquivo_origem"] = os.path.basename(a)
        dfs.append(d)
        print(f"  lido {a}: {d.shape}")
    return pd.concat(dfs, ignore_index=True), arqs

def detectar(df):
    cols = {c.lower().strip(): c for c in df.columns}
    mapa = {}
    for k, pat in PATTERNS.items():
        if k in OVERRIDE:
            mapa[k] = OVERRIDE[k]; continue
        for low, orig in cols.items():
            if re.search(pat, low) and orig not in mapa.values():
                mapa[k] = orig; break
    return mapa

# ----------------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------------
def salvar_tab(df, nome, **kw):
    df.to_csv(f"{TAB}/{nome}.csv", encoding="utf-8-sig")
    try:
        df.to_latex(f"{TAB}/{nome}.tex", escape=True, **kw)
    except Exception as e:
        print("  (latex falhou para", nome, e, ")")

def salvar_fig(nome):
    plt.tight_layout()
    plt.savefig(f"{FIG}/{nome}.png", dpi=170)
    plt.close()

def desc(s):
    s = s.dropna()
    q1, q3 = s.quantile([.25, .75])
    m = s.mean()
    return pd.Series({
        "n": len(s), "média": m, "mediana": s.median(), "var": s.var(ddof=1),
        "desvio": s.std(ddof=1), "mín": s.min(), "Q1": q1, "Q3": q3, "máx": s.max(),
        "IQR": q3 - q1, "CV(%)": (s.std(ddof=1) / m * 100) if m != 0 else np.nan,
        "assimetria": stats.skew(s), "curtose(exc.)": stats.kurtosis(s)})

def iqr_mask(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return (s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)

# ----------------------------------------------------------------------------
def main(pasta):
    print("== Carregando ==")
    df, arqs = carregar(pasta)
    mapa = detectar(df)
    print("\n== Mapeamento de colunas detectado (CONFIRA!) ==")
    for k, v in mapa.items():
        print(f"  {k:11s} -> {v}")
    json.dump(mapa, open(f"{OUT}/mapa_colunas.json", "w"), indent=1, ensure_ascii=False)

    # dicionário preliminar de variáveis
    dic = pd.DataFrame({"tipo": df.dtypes.astype(str), "n_ausentes": df.isna().sum(),
                        "pct_ausentes": (df.isna().mean() * 100).round(3),
                        "n_unicos": df.nunique()})
    salvar_tab(dic, "dicionario_preliminar")
    print("\n== Esquema ==\n", dic.to_string())

    num = [mapa[k] for k in ("tensao", "corrente", "potencia", "energia", "frequencia", "fp") if k in mapa]
    inv = {v: k for k, v in mapa.items()}
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # estação
    df["estacao"] = df["arquivo_origem"].str.extract(r"(?i)(station[_\s-]?\w+|st[_\s-]?\w+|s\d)", expand=False)
    df["estacao"] = df["estacao"].fillna(df["arquivo_origem"])
    # rótulo binário
    if "anomalia" in mapa:
        a = df[mapa["anomalia"]]
        df["_anom"] = a.astype(str).str.lower().isin(["1", "true", "1.0", "yes", "anomaly", "anomalous"]).astype(int)
    else:
        df["_anom"] = 0
        print("ATENÇÃO: coluna de anomalia não detectada; ajuste OVERRIDE.")
    df["grupo"] = np.where(df["_anom"] == 1, "Anômalo", "Normal")

    # ---------------- 2. AUDITORIA / LIMPEZA (apenas sinaliza) ----------------
    log = []
    n0 = len(df)
    log.append(("Registros brutos", n0, n0, "—"))
    dup = df.drop(columns=["arquivo_origem", "estacao", "_anom", "grupo"]).duplicated().sum()
    log.append(("Linhas 100% duplicadas (sinalizadas, não removidas)", n0, n0, f"{dup} duplicatas"))
    if "tempo" in mapa:
        t = pd.to_datetime(df[mapa["tempo"]], errors="coerce")
        df["_t"] = t
        log.append(("Timestamps não interpretáveis", n0, n0, f"{int(t.isna().sum())}"))
        dts = df.sort_values(["estacao", "_t"]).groupby("estacao")["_t"].diff().dt.total_seconds()
        log.append(("Passo temporal != mediana (lacunas/saltos)", n0, n0, f"{int((dts.dropna() != dts.median()).sum())} (mediana={dts.median():.2f}s)"))
        dup_t = df.duplicated(subset=["estacao", "_t"]).sum()
        log.append(("Timestamps repetidos na mesma estação", n0, n0, f"{dup_t}"))
    for c in num:
        k = inv[c]
        log.append((f"Ausentes em {c}", n0, n0, f"{int(df[c].isna().sum())}"))
        if k in FAIXAS:
            lo, hi = FAIXAS[k]
            bad = ((df[c] < lo) | (df[c] > hi)).sum()
            zeros = (df[c] == 0).sum()
            log.append((f"{c}: fora da faixa física [{lo},{hi}] {UNID[k]}", n0, n0, f"{int(bad)}"))
            log.append((f"{c}: valores exatamente 0 (possível queda/sensor)", n0, n0, f"{int(zeros)}"))
    for col in ("tipo", "categoria", "severidade"):
        if col in mapa:
            log.append((f"Categorias distintas em {mapa[col]}", n0, n0, f"{df[mapa[col]].nunique()}"))
    logdf = pd.DataFrame(log, columns=["Verificação", "n antes", "n depois", "Resultado"])
    logdf.to_csv(f"{OUT}/log_limpeza.csv", index=False, encoding="utf-8-sig")
    salvar_tab(logdf.set_index("Verificação"), "log_limpeza", column_format="p{7cm}rrp{4cm}")

    # ---------------- 3. COMPOSIÇÃO DOS DADOS ----------------
    comp = df.groupby("estacao").agg(n=("grupo", "size"), pct_anomalo=("_anom", lambda x: x.mean() * 100))
    salvar_tab(comp.round(3), "composicao_estacoes")
    plt.figure(figsize=(5, 3.5))
    df["grupo"].value_counts().plot.bar(color=["#4c72b0", "#c44e52"])
    plt.ylabel("Nº de amostras"); plt.title("Balanço de classes"); plt.xticks(rotation=0)
    salvar_fig("balanco_classes")
    for col, nome in (("tipo", "tipos_falha"), ("severidade", "severidade_freq")):
        if col in mapa:
            vc = df.loc[df["_anom"] == 1, mapa[col]].value_counts()
            plt.figure(figsize=(7, 4)); vc.plot.barh(); plt.xlabel("Nº de amostras (anômalas)")
            plt.title(f"Frequência por {mapa[col]}"); salvar_fig(nome)
            salvar_tab(vc.to_frame("n"), nome)

    # ---------------- 4. ESTATÍSTICA DESCRITIVA ----------------
    D = pd.DataFrame({c: desc(df[c]) for c in num}).T
    D.index = [f"{ROTULO[inv[c]]} ({UNID[inv[c]]})" for c in num]
    salvar_tab(D.round(4), "descritiva_geral")
    for g in ("Normal", "Anômalo"):
        sub = df[df["grupo"] == g]
        if len(sub) > 1:
            Dg = pd.DataFrame({c: desc(sub[c]) for c in num}).T
            Dg.index = D.index
            salvar_tab(Dg.round(4), f"descritiva_{'normal' if g=='Normal' else 'anomalo'}")
    est = df.groupby("estacao")[num].agg(["mean", "std"]).round(4)
    est.columns = [f"{a}_{b}" for a, b in est.columns]
    salvar_tab(est, "descritiva_por_estacao")

    # ---------------- 5. OUTLIERS ----------------
    rows = []
    for c in num:
        m = iqr_mask(df[c].dropna()).reindex(df.index, fill_value=False)
        z = ((df[c] - df[c].mean()) / df[c].std(ddof=1)).abs() > 3
        r = {"Variável": c, "n IQR": int(m.sum()), "% IQR": round(m.mean() * 100, 3),
             "n |z|>3": int(z.sum()),
             "% dos outliers IQR rotulados anômalos": round(df.loc[m, "_anom"].mean() * 100, 2) if m.any() else np.nan,
             "% anômalos que são outliers IQR": round(m[df["_anom"] == 1].mean() * 100, 2) if (df["_anom"] == 1).any() else np.nan}
        rows.append(r)
    O = pd.DataFrame(rows).set_index("Variável")
    salvar_tab(O, "outliers")
    # outliers DENTRO do regime normal (candidatos a erro/evento não rotulado)
    nr = []
    for c in num:
        s = df.loc[df["_anom"] == 0, c].dropna()
        nr.append({"Variável": c, "n outliers IQR (só normal)": int(iqr_mask(s).sum()),
                   "% (só normal)": round(iqr_mask(s).mean() * 100, 3)})
    salvar_tab(pd.DataFrame(nr).set_index("Variável"), "outliers_somente_normal")

    # ---------------- 6. VISUALIZAÇÃO UNIVARIADA ----------------
    n = len(num); nc = 3; nr_ = int(np.ceil(n / nc))
    fig, ax = plt.subplots(nr_, nc, figsize=(12, 3.2 * nr_)); ax = np.atleast_1d(ax).ravel()
    for i, c in enumerate(num):
        s = df[c].replace([np.inf, -np.inf], np.nan).dropna(); k = 60
        ax[i].hist(s, bins=k, color="#4c72b0", log=True)
        ax[i].set_xlabel(f"{ROTULO[inv[c]]} ({UNID[inv[c]]})"); ax[i].set_ylabel("Frequência (escala log)")
    for j in range(i + 1, len(ax)): ax[j].axis("off")
    salvar_fig("histogramas")
    fig, ax = plt.subplots(nr_, nc, figsize=(12, 3.2 * nr_)); ax = np.atleast_1d(ax).ravel()
    for i, c in enumerate(num):
        dados = [df.loc[df["grupo"] == g, c].dropna() for g in ("Normal", "Anômalo")]
        ax[i].boxplot(dados, labels=["Normal", "Anômalo"], showfliers=True,
                      flierprops=dict(marker=".", markersize=2, alpha=.3))
        ax[i].set_ylabel(f"{ROTULO[inv[c]]} ({UNID[inv[c]]})")
    for j in range(i + 1, len(ax)): ax[j].axis("off")
    salvar_fig("boxplots_grupo")
    if "severidade" in mapa:
        sev = mapa["severidade"]
        for k in ("potencia", "tensao", "corrente"):
            if k in mapa:
                c = mapa[k]; ordem = sorted(df[sev].dropna().unique())
                plt.figure(figsize=(7, 3.8))
                plt.boxplot([df.loc[df[sev] == o, c].dropna() for o in ordem], labels=[str(o) for o in ordem],
                            flierprops=dict(marker=".", markersize=2, alpha=.3))
                plt.xlabel("Severidade"); plt.ylabel(f"{ROTULO[k]} ({UNID[k]})")
                salvar_fig(f"box_severidade_{k}")
    if "tempo" in mapa and "evento" in mapa and (df["_anom"] == 1).any():
        ev = df.loc[df["_anom"] == 1, mapa["evento"]].dropna()
        if len(ev):
            e0 = ev.value_counts().index[0]
            idx = df.index[df[mapa["evento"]] == e0]
            lo, hi = max(idx.min() - 120, 0), min(idx.max() + 120, len(df) - 1)
            seg = df.loc[lo:hi]
            fig, ax = plt.subplots(len(num[:4]), 1, figsize=(9, 2.2 * len(num[:4])), sharex=True)
            for a_, c in zip(np.atleast_1d(ax), num[:4]):
                a_.plot(seg[c].values, lw=.8); a_.set_ylabel(f"{inv[c]} ({UNID[inv[c]]})")
                a_.axvspan(idx.min() - lo, idx.max() - lo, color="red", alpha=.15)
            a_.set_xlabel("Amostra (1 Hz ≈ s)")
            salvar_fig("serie_evento_exemplo")

    # ---------------- 7. BIVARIADA / MULTIVARIADA ----------------
    amostra = df.sample(min(len(df), 30000), random_state=SEED)
    for cm, nome in (("pearson", "corr_pearson"), ("spearman", "corr_spearman")):
        C = df[num].corr(method=cm) if cm == "pearson" else amostra[num].corr(method=cm)
        C.index = C.columns = [inv[c] for c in C.columns]
        salvar_tab(C.round(3), nome)
        plt.figure(figsize=(6, 5)); plt.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
        plt.xticks(range(len(C)), C.columns, rotation=45, ha="right"); plt.yticks(range(len(C)), C.index)
        for i in range(len(C)):
            for j in range(len(C)): plt.text(j, i, f"{C.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(label=cm); salvar_fig(nome)
    pares = [(a, b) for a, b in [("potencia", "corrente"), ("potencia", "tensao"), ("fp", "potencia"), ("frequencia", "tensao")]
             if a in mapa and b in mapa]
    if pares:
        fig, ax = plt.subplots(1, len(pares), figsize=(4.2 * len(pares), 3.8)); ax = np.atleast_1d(ax)
        for a_, (x, y) in zip(ax, pares):
            for g, cor in (("Normal", "#4c72b0"), ("Anômalo", "#c44e52")):
                s = amostra[amostra["grupo"] == g]
                a_.scatter(s[mapa[x]], s[mapa[y]], s=3, alpha=.25, c=cor, label=g)
            a_.set_xlabel(f"{ROTULO[x]} ({UNID[x]})"); a_.set_ylabel(f"{ROTULO[y]} ({UNID[y]})")
        ax[0].legend(markerscale=4); salvar_fig("dispersoes")
    sub = [mapa[k] for k in ("tensao", "corrente", "potencia", "fp") if k in mapa]
    if len(sub) >= 3:
        pp = amostra.sample(min(len(amostra), 4000), random_state=SEED)
        k = len(sub); fig, ax = plt.subplots(k, k, figsize=(2.6 * k, 2.6 * k))
        for i, yi in enumerate(sub):
            for j, xj in enumerate(sub):
                if i == j:
                    for g, cor in (("Normal", "#4c72b0"), ("Anômalo", "#c44e52")):
                        ax[i, j].hist(pp.loc[pp["grupo"] == g, xj].dropna(), bins=30, alpha=.5, color=cor)
                else:
                    ax[i, j].scatter(pp[xj], pp[yi], s=2, alpha=.25, c=pp["_anom"].map({0: "#4c72b0", 1: "#c44e52"}))
                if i == k - 1: ax[i, j].set_xlabel(inv[xj])
                if j == 0: ax[i, j].set_ylabel(inv[yi])
        salvar_fig("pairplot")

    # Normal vs Anômalo: tamanho de efeito (subsídio às hipóteses, NÃO é o teste da Unidade 2)
    if (df["_anom"] == 1).any() and (df["_anom"] == 0).any():
        ef = []
        for c in num:
            x0 = df.loc[df["_anom"] == 0, c].dropna(); x1 = df.loc[df["_anom"] == 1, c].dropna()
            sp = np.sqrt(((len(x0) - 1) * x0.var() + (len(x1) - 1) * x1.var()) / (len(x0) + len(x1) - 2))
            ef.append({"Variável": c, "mediana normal": x0.median(), "mediana anômalo": x1.median(),
                       "d de Cohen": (x1.mean() - x0.mean()) / sp if sp > 0 else np.nan,
                       "razão de CV (anôm./norm.)": (x1.std() / abs(x1.mean())) / (x0.std() / abs(x0.mean()))
                       if x0.mean() and x1.mean() else np.nan})
        salvar_tab(pd.DataFrame(ef).set_index("Variável").round(4), "efeito_normal_vs_anomalo")
        # por categoria de falha
        for col in ("categoria", "tipo"):
            if col in mapa:
                g = df[df["_anom"] == 1].groupby(mapa[col])[num].median().round(3)
                salvar_tab(g, f"medianas_por_{col}")

    # Multicolinearidade (VIF simples por R2 de regressão auxiliar)
    X = df[num].dropna().sample(min(len(df), 50000), random_state=SEED)
    X = (X - X.mean()) / X.std(ddof=1)
    vif = {}
    for c in X.columns:
        y = X[c].values; A = X.drop(columns=c).values
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        vif[inv[c]] = 1 / (1 - r2) if r2 < 1 else np.inf
    salvar_tab(pd.Series(vif, name="VIF").round(2).to_frame(), "vif")

    json.dump({"n_total": int(n0), "n_arquivos": len(arqs), "pct_anomalo": float(df["_anom"].mean() * 100),
               "colunas": list(df.columns)[:60]}, open(f"{OUT}/resumo.json", "w"), indent=1, ensure_ascii=False)
    print("\nPronto. Veja a pasta", OUT)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
