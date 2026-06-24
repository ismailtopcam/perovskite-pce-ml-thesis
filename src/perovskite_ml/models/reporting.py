"""Gercek-vs-tahmin raporlama: okunabilir kompozisyon etiketi, saçilim grafigi
ve yakin/tipik/uzak ornek tablosu (durust hata dagilimi)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_IONS=["FA","MA","Cs","Rb","K"]; B_IONS=["Pb","Sn","Ge"]; X_IONS=["I","Br","Cl"]

def composition_label(row):
    def part(prefix, ions):
        s=[f"{i}{row[f'{prefix}_{i}']:.2f}" for i in ions
           if f"{prefix}_{i}" in row and row[f"{prefix}_{i}"]>0.01]
        return "".join(s) if s else "-"
    return f"{part('A',A_IONS)} / {part('B',B_IONS)} / {part('X',X_IONS)}"

def build_predictions_table(M_test, y_true, y_pred, doi):
    tab=pd.DataFrame({
        "DOI":doi,
        "Kompozisyon":[composition_label(r) for _,r in M_test.reset_index(drop=True).iterrows()],
        "Gercek_PCE":np.round(y_true,2),
        "Model_tahmini":np.round(y_pred,2)})
    tab["Mutlak_hata"]=np.round((tab["Gercek_PCE"]-tab["Model_tahmini"]).abs(),2)
    return tab

def stratified_examples(tab, n=4):
    closest=tab.nsmallest(n,"Mutlak_hata")
    med=tab["Mutlak_hata"].median()
    typical=tab.iloc[(tab["Mutlak_hata"]-med).abs().argsort()[:n]]
    worst=tab.nlargest(n,"Mutlak_hata")
    closest=closest.assign(Grup="yakin"); typical=typical.assign(Grup="tipik"); worst=worst.assign(Grup="uzak")
    return pd.concat([closest,typical,worst],ignore_index=True)

def save_scatter(y_true, y_pred, path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true,y_pred,s=6,alpha=0.25)
    lim=[min(y_true.min(),y_pred.min()),max(y_true.max(),y_pred.max())]
    plt.plot(lim,lim,"--",color="red",label="ideal (tahmin=gercek)")
    plt.xlabel("Gercek PCE (%)"); plt.ylabel("Model tahmini PCE (%)")
    plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()
