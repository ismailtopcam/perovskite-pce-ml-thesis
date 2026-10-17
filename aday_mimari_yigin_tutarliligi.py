# -*- coding: utf-8 -*-
"""Aday mimari-yigin tutarliligi — 'pin' etiketli kayitlarda katman sirasi denetimi (tez Bolum 4.5).

Perovskite Database'de Cell_architecture etiketi ile Cell_stack_sequence'teki gercek
katman sirasi her kayitta tutarli degildir. Bu betik ham veride, 'pin' etiketli
kayitlarin Cell_stack_sequence alanini '|' ile bolup Perovskite katmanindan hemen
onceki katman(lar)i kaydin ETL_stack_sequence / HTL_stack_sequence malzemeleriyle
eslestirir:
  - ETL perovskitten ONCE kaplanmissa -> "n-i-p dizilimli" (etiketle celisir)
  - HTL perovskitten once ise          -> "p-i-n dizilimli" (etiketle uyumlu)
  - ikisi de belirlenemiyorsa          -> "belirsiz" (ayri sayilir, gerekcesiyle)
Ayni sayim 'nip' etiketli kayitlar icin simetrik olarak (kontrol) yapilir.

Ek olarak tezde anilan iki aday uclusunun ham kayitlari listelenir:
  (pin, ETL = "SnO2-np",           HTL = "Spiro-MeOTAD")
  (pin, ETL = "TiO2-c | TiO2-mp",  HTL = "Spiro-MeOTAD")
kayit sayisi, benzersiz DOI sayisi, Cell_stack_sequence degerleri ve dizilim sinifi;
model-hazir veri ve egitim bolmesindeki karsiliklari (aday_uyumluluk_filtresi.py ile
ayni GroupShuffleSplit, seed 42, test_size 0,2) birlikte raporlanir.

Girdi : data/raw/Perovskite_database_content_all_data.csv (43.398 kayit)
        data/processed/model_ready_dataset.csv (uclu capraz kontrolu; yoksa atlanir)
Cikti : outputs/robustness/aday_mimari_yigin_tutarliligi.json + konsol ozeti
Calistirma (repo kokunden): python aday_mimari_yigin_tutarliligi.py   (~10 sn)
"""
import json
import re
from pathlib import Path

import pandas as pd

RAW = "data/raw/Perovskite_database_content_all_data.csv"
MODEL_READY = "data/processed/model_ready_dataset.csv"
OUT = Path("outputs/robustness/aday_mimari_yigin_tutarliligi.json")

COLS = ["Ref_DOI_number", "JV_default_PCE", "Cell_architecture", "Cell_stack_sequence",
        "ETL_stack_sequence", "HTL_stack_sequence"]
BOS = {"", "none", "unknown", "nan"}

# Tezde anilan ucluler: (ham ETL_stack_sequence, ham HTL_stack_sequence, model-hazir one-hot kolonlari, tez notu)
UCLULER = [
    {"ad": "pin / SnO2-np / Spiro-MeOTAD", "etl": "SnO2-np", "htl": "Spiro-MeOTAD",
     "mr_etl": "ETL_SnO2_np", "mr_htl": "HTL_Spiro_MeOTAD"},
    {"ad": "pin / TiO2-c | TiO2-mp / Spiro-MeOTAD", "etl": "TiO2-c | TiO2-mp", "htl": "Spiro-MeOTAD",
     "mr_etl": "ETL_TiO2_c___TiO2_mp", "mr_htl": "HTL_Spiro_MeOTAD"},
]


AYIRICI = re.compile(r"[|｜]")  # ASCII '|' ve tam-genislik '｜' (ham veride 11 kayit)
YAZIM_VARYANTI = {"n": 0}  # 'Perovskte', 'Preovskite' gibi yazim hatasi tasiyan katman sayaci


def katmanlar(s):
    if pd.isna(s):
        return []
    return [t.strip() for t in AYIRICI.split(str(s)) if t.strip()]


def malzeme_kumesi(s):
    return {t.lower() for t in katmanlar(s)} - BOS


def _edit_mesafesi(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def perovskit_mi(tok):
    """'Perovskite' katmani mi? Yazim hatalarini (edit mesafesi <= 2) da yakalar."""
    t = tok.lower()
    if t.startswith("perovskite"):  # 'perovskite', 'perovskite-sc', 'perovskite-qd' ...
        return True
    harf = re.sub(r"[^a-z]", "", t)
    if len(harf) >= 8 and _edit_mesafesi(harf, "perovskite") <= 2:
        YAZIM_VARYANTI["n"] += 1
        return True
    return False


def siniflandir(stack, etl, htl):
    """(sinif, kural) dondurur. sinif: nip_sirali | pin_sirali | belirsiz."""
    L = [t.lower() for t in katmanlar(stack)]
    if not L or L == ["unknown"]:
        return "belirsiz", "yigin_bos"
    idx = [i for i, t in enumerate(L) if perovskit_mi(t)]
    if not idx:
        return "belirsiz", "perovskite_katmani_yok"
    i = idx[0]
    E, H = malzeme_kumesi(etl), malzeme_kumesi(htl)
    onceki, sonraki = L[:i], L[i + 1:]
    if not onceki:
        return "belirsiz", "perovskite_ilk_katman"
    # 1) Perovskitten hemen onceki katman
    p = onceki[-1]
    if p in E and p not in H:
        return "nip_sirali", "onceki_katman_ETL"
    if p in H and p not in E:
        return "pin_sirali", "onceki_katman_HTL"
    if p in E and p in H:
        return "belirsiz", "onceki_katman_hem_ETL_hem_HTL"
    # 2) Perovskitten onceki blok (ara katman — ornegin pasivasyon — olabilir)
    blok = set(onceki)
    e_once, h_once = bool(blok & E), bool(blok & H)
    if e_once and not h_once:
        return "nip_sirali", "onceki_blok_ETL"
    if h_once and not e_once:
        return "pin_sirali", "onceki_blok_HTL"
    if e_once and h_once:
        return "belirsiz", "onceki_blok_hem_ETL_hem_HTL"
    # 3) Onceki katmanlar ETL/HTL degil (ornegin HTL-free: ITO | Perovskite | PCBM):
    #    perovskitten hemen sonraki katmana bakilir
    if sonraki:
        s = sonraki[0]
        if s in E and s not in H:
            return "pin_sirali", "sonraki_katman_ETL"
        if s in H and s not in E:
            return "nip_sirali", "sonraki_katman_HTL"
    return "belirsiz", "eslesme_yok"


def ozet(df, celisen_sinif):
    """Bir alt kume icin sinif ve kural sayimlari; 'celisen_sinif' etiketle celisen dizilimdir."""
    n = int(len(df))
    sinif = df["sinif"].value_counts().to_dict()
    kural = df["kural"].value_counts().to_dict()
    n_cel = int(sinif.get(celisen_sinif, 0))
    n_cel_onceki = int(df[(df["sinif"] == celisen_sinif) & df["kural"].str.startswith("onceki_")].shape[0])
    return {
        "n_kayit": n,
        "n_nip_sirali": int(sinif.get("nip_sirali", 0)),
        "n_pin_sirali": int(sinif.get("pin_sirali", 0)),
        "n_belirsiz": int(sinif.get("belirsiz", 0)),
        "etiketle_celisen_sinif": celisen_sinif,
        "n_etiketle_celisen": n_cel,
        "pct_etiketle_celisen": round(100 * n_cel / n, 2) if n else None,
        "n_etiketle_celisen_yalniz_onceki_katman_kurali": n_cel_onceki,
        "pct_etiketle_celisen_yalniz_onceki_katman_kurali": round(100 * n_cel_onceki / n, 2) if n else None,
        "kural_dagilimi": {k: int(v) for k, v in kural.items()},
    }


def main():
    R = pd.read_csv(RAW, encoding="utf-8-sig", low_memory=False, usecols=COLS)
    R["pce"] = pd.to_numeric(R["JV_default_PCE"], errors="coerce")
    R["doi_kanonik"] = R["Ref_DOI_number"].astype(str).str.strip().str.lower().where(R["Ref_DOI_number"].notna())
    print(f"Ham veri: {len(R):,} kayit | Cell_architecture dagilimi: "
          f"{R['Cell_architecture'].value_counts(dropna=False).to_dict()}")

    sk = [siniflandir(a, b, c) for a, b, c in zip(R["Cell_stack_sequence"], R["ETL_stack_sequence"],
                                                  R["HTL_stack_sequence"])]
    R["sinif"] = [s for s, _ in sk]
    R["kural"] = [k for _, k in sk]

    res = {
        "girdi": {"dosya": RAW, "n_kayit": int(len(R)), "n_pce_dolu": int(R["pce"].notna().sum()),
                  "n_yigin_tam_genislik_ayirici": int(R["Cell_stack_sequence"].astype(str).str.contains("｜").sum()),
                  "n_perovskite_yazim_varyanti_katman": int(YAZIM_VARYANTI["n"])},
        "siniflandirma_kurali": (
            "Cell_stack_sequence '|' (ve tam-genislik '｜') ile bolunur; ilk 'Perovskite' katmani bulunur "
            "(yazim hatalari — Perovskte, Preovskite vb., edit mesafesi <= 2 — perovskit sayilir). "
            "(1) Hemen onceki katman ETL_stack_sequence malzemesiyse 'nip_sirali', HTL_stack_sequence "
            "malzemesiyse 'pin_sirali'. (2) Degilse perovskitten onceki blokta ETL/HTL malzemesi aranir. "
            "(3) Onceki katmanlar ne ETL ne HTL ise (ornegin HTL-free) perovskitten hemen sonraki katmana "
            "bakilir (ETL sonra -> pin_sirali, HTL sonra -> nip_sirali). Kalanlar 'belirsiz'. "
            "'none'/'Unknown' malzeme sayilmaz."
        ),
        "mimari_etiketi_dagilimi": {str(k): int(v) for k, v in
                                    R["Cell_architecture"].value_counts(dropna=False).items()},
    }

    for etiket, celisen in (("pin", "nip_sirali"), ("nip", "pin_sirali")):
        A = R[R["Cell_architecture"] == etiket]
        res[etiket] = {
            "tum_kayitlar": ozet(A, celisen),
            "pce_dolu_kayitlar": ozet(A[A["pce"].notna()], celisen),
        }
        C = A[A["sinif"] == celisen]
        res[etiket]["etiketle_celisen_ornek_yiginlar_top15"] = {
            str(k): int(v) for k, v in C["Cell_stack_sequence"].value_counts().head(15).items()}
        res[etiket]["etiketle_celisen_n_doi_kanonik"] = int(C["doi_kanonik"].nunique())
        t, p_ = res[etiket]["tum_kayitlar"], res[etiket]["pce_dolu_kayitlar"]
        print(f"\n[{etiket}] etiketli: {t['n_kayit']:,} kayit (PCE dolu {p_['n_kayit']:,})")
        print(f"  nip_sirali {t['n_nip_sirali']:,} | pin_sirali {t['n_pin_sirali']:,} | belirsiz {t['n_belirsiz']:,}")
        print(f"  Etiketle celisen ({celisen}): tum kayitlar {t['n_etiketle_celisen']:,}/{t['n_kayit']:,} "
              f"= %{t['pct_etiketle_celisen']:.2f} | PCE dolu {p_['n_etiketle_celisen']:,}/{p_['n_kayit']:,} "
              f"= %{p_['pct_etiketle_celisen']:.2f}")
        print(f"  (yalniz onceki-katman kurallari: tum {t['n_etiketle_celisen_yalniz_onceki_katman_kurali']:,} "
              f"= %{t['pct_etiketle_celisen_yalniz_onceki_katman_kurali']:.2f}; PCE dolu "
              f"{p_['n_etiketle_celisen_yalniz_onceki_katman_kurali']:,} "
              f"= %{p_['pct_etiketle_celisen_yalniz_onceki_katman_kurali']:.2f})")
        print(f"  Kural dagilimi (tum): {t['kural_dagilimi']}")

    # --- Model-hazir veri ve egitim bolmesi (capraz kontrol) ---
    M, Mt = None, None
    if Path(MODEL_READY).exists():
        M = pd.read_csv(MODEL_READY, low_memory=False)
        try:
            from sklearn.model_selection import GroupShuffleSplit
            groups = M["Ref_DOI_number"].fillna("unknown_doi").astype(str).values
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_idx, _ = next(gss.split(M, M["JV_default_PCE"], groups))
            Mt = M.iloc[tr_idx]
        except ImportError:
            print("\n[UYARI] scikit-learn yok: egitim-bolmesi sayimi atlandi.")

    # --- Tezde anilan iki uclu ---
    res["ucluler"] = []
    print("\nTezde anilan ucluler (ham veri, Cell_architecture == 'pin'):")
    for u in UCLULER:
        S = R[(R["Cell_architecture"] == "pin") & (R["ETL_stack_sequence"] == u["etl"])
              & (R["HTL_stack_sequence"] == u["htl"])]
        kayit = {
            "uclu": u["ad"],
            "ham": {
                "n_kayit": int(len(S)),
                "n_pce_dolu": int(S["pce"].notna().sum()),
                "n_doi_ham": int(S["Ref_DOI_number"].nunique()),
                "n_doi_kanonik": int(S["doi_kanonik"].nunique()),
                "doi_listesi": sorted(S["doi_kanonik"].dropna().unique().tolist()),
                "yigin_dizilimleri": {str(k): int(v) for k, v in S["Cell_stack_sequence"].value_counts().items()},
                "sinif_dagilimi": {str(k): int(v) for k, v in S["sinif"].value_counts().items()},
                "kural_dagilimi": {str(k): int(v) for k, v in S["kural"].value_counts().items()},
                "pct_nip_sirali": round(100 * float((S["sinif"] == "nip_sirali").mean()), 1) if len(S) else None,
            },
        }
        if M is not None and {u["mr_etl"], u["mr_htl"], "arch_pin"} <= set(M.columns):
            for ad, D in (("model_hazir", M), ("egitim_bolmesi", Mt)):
                if D is None:
                    continue
                T = D[(D["arch_pin"] == 1) & (D[u["mr_etl"]] == 1) & (D[u["mr_htl"]] == 1)]
                kayit[ad] = {"n_kayit": int(len(T)),
                             "n_doi": int(T["Ref_DOI_number"].fillna("unknown_doi").astype(str).nunique()),
                             "doi_listesi": sorted(T["Ref_DOI_number"].dropna().astype(str).unique().tolist())}
        res["ucluler"].append(kayit)
        h = kayit["ham"]
        print(f"\n  {u['ad']}: ham {h['n_kayit']} kayit (PCE dolu {h['n_pce_dolu']}) | "
              f"DOI {h['n_doi_ham']} ham / {h['n_doi_kanonik']} kanonik | n-i-p sirali %{h['pct_nip_sirali']}")
        for st, n in h["yigin_dizilimleri"].items():
            print(f"    {n:3d} x {st}")
        for ad in ("model_hazir", "egitim_bolmesi"):
            if ad in kayit:
                print(f"    {ad}: {kayit[ad]['n_kayit']} kayit / {kayit[ad]['n_doi']} DOI")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nYazildi: {OUT}")


if __name__ == "__main__":
    main()
