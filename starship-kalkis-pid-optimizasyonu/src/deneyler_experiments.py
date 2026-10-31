"""
deneyler_experiments.py – Deney yöneticisi
  * 3 yöntem: Genetik Algoritma (GA), Karınca Kolonisi (ACO_R) ve Rastgele Arama (taban çizgisi)
  * Her yöntem N_RUNS bağımsız koşu (seed 1..N_RUNS), eşit bütçe (POP*ITER = 1800 değerlendirme)
  * Eğitim: sabit 9 senaryo (IFT-6/7/8'den 3'er, seed 0) → adil karşılaştırma
  * Test : eğitimde görülmemiş 200 senaryo (seed 12345) → ezber kontrolü
  * Karşılaştırma için elle ayarlanmış PID de test edilir
Çalıştırma:  python deneyler_experiments.py   (~3 dk)   Çıktılar: ../sonuclar_results/*.csv, *.json
"""
import json                                                                # yakınsama eğrilerini kaydetmek için
import os                                                                  # klasör işlemleri
import time                                                                # süre ölçümü
import numpy as np                                                         # sayısal işlemler
import pandas as pd                                                        # tablolar
import ayarlar_config as C                                                         # ayar defteri
import roket_modeli_rocket_model as M                                                          # sahte Booster + puan makinesi
import optimizasyon_optimization as O                                                   # GA, ACO ve rastgele arama

MANUEL = np.array([0.5, 0.05, 0.05, 0.5, 0.05, 0.05])                      # elle ayarlanmış karşılaştırma PID'i
N_TEST, TEST_SEED = 200, 12345                                             # son sınav: 200 senaryo, seed 12345


class KayitliAmac(M.AmacFonksiyonu):
    """Not tutan puan makinesi: her çağrıda (deneme sayısı, şimdiye kadarki en iyi J) kaydeder."""

    def __init__(self):
        super().__init__(C.N_SCEN, seed=0)                                 # 9 eğitim senaryosu, seed 0
        self.gecmis, self.en_iyi = [], np.inf                              # not defteri ve rekor

    def __call__(self, X):
        J = super().__call__(X)                                            # normal puanlama
        self.en_iyi = min(self.en_iyi, J.min())                            # rekoru güncelle
        self.gecmis.append((self.n_eval, self.en_iyi))                     # (deneme, en iyi J) kaydet
        return J


def test_metrikleri(x):
    """Bir ayarı 200 yeni senaryoda uçurur ve karne çıkarır."""
    sc = M.senaryolar(N_TEST, TEST_SEED)                                   # 200 test senaryosu
    r = M.simule(np.asarray(x)[None], sc)                                  # 200 uçuş
    dv = (r["v_son"] - r["v_son_ref"])[0]                                  # 150. s'deki hız hataları
    return dict(
        J_test=float(M.maliyet(r)[0]),                                     # test puanı
        IAE_ort=float(r["iae"][0].mean()),                                 # ortalama takip hatası
        v_son_hata_ort=float(np.abs(dv).mean()),                           # ortalama son hız hatası
        v_son_hata_maks=float(np.abs(dv).max()),                           # en kötü son hız hatası
        maxQ_ort_kPa=float(r["qmax"][0].mean() / 1e3),                     # ortalama max-Q
        maxQ_ihlal=int(np.sum(r["qmax"][0] > r["q_lim"][0])),              # sınırı aşan senaryo sayısı
        ivme_maks_g=float(r["amax"][0].max()),                             # en yüksek ivme (g)
        yakit_ort_ton=float(r["yakit"][0].mean() / 1e3),                   # ortalama yakıt (ton)
        TV_ort=float(r["tv"][0].mean()),                                   # ortalama gaz oynaması
    )


def main():
    os.makedirs(C.SONUC_KLASORU, exist_ok=True)                            # sonuclar/ klasörünü oluştur
    izgara = np.arange(C.POP, C.POP * C.ITER + 1, C.POP)                   # 30, 60, …, 1800 deneme ızgarası
    satirlar, egriler = [], {}                                             # koşu tablosu ve eğriler

    # a–c) her yöntem × 10 koşu
    for ad, alg in O.ALGORITMALAR.items():                                 # GA, ACO, Rastgele Arama
        egriler[ad] = []
        for s in range(1, C.N_RUNS + 1):                                   # seed 1…10
            f = KayitliAmac()                                              # her koşuya temiz puan makinesi
            t0 = time.time()                                               # süre ölçümü başlasın
            x, J, _ = alg(f, C.LOWER, C.UPPER, C.POP, C.ITER, seed=s)      # optimizasyonu çalıştır
            sure = time.time() - t0                                        # geçen süre
            ev, best = np.array(f.gecmis).T                                # deneme sayısı ve en iyi J geçmişi
            egriler[ad].append(np.interp(izgara, ev, best).tolist())       # ortak ızgaraya oturt
            satirlar.append(dict(yontem=ad, kosu=s, J_egitim=float(J), sure_s=sure,
                                 n_eval=f.n_eval, **dict(zip(C.PARAM_ADLARI, map(float, x)))))  # satır ekle
            print(f"{ad:15s} koşu {s:2d}  J={J:.4f}  {sure:.1f}s  x={np.round(x, 3)}")       # ekrana yaz

    df = pd.DataFrame(satirlar)                                            # tüm koşular tablosu
    df.to_csv(os.path.join(C.SONUC_KLASORU, "tum_kosular_all_runs.csv"), index=False)  # kaydet

    # d) özet istatistikler (eğitim J)
    ozet = df.groupby("yontem", sort=False)["J_egitim"].agg(["min", "mean", "std", "max"])  # en iyi/ort/std/en kötü
    ozet["sure_ort_s"] = df.groupby("yontem", sort=False)["sure_s"].mean()  # ortalama koşu süresi
    J_star = df["J_egitim"].min()                                          # bulunan en iyi J
    gerekli = {}                                                           # %1 yakınlığa ulaşma
    for ad, E in egriler.items():
        n = [izgara[np.argmax(np.array(e) <= J_star * 1.01)] if min(e) <= J_star * 1.01 else np.nan for e in E]
        n = np.array(n, float)                                             # ulaşamayan koşu → nan
        gerekli[ad] = (np.nanmean(n) if np.any(~np.isnan(n)) else np.nan, int(np.sum(~np.isnan(n))))
    ozet["eval_%1_ort"] = [gerekli[a][0] for a in ozet.index]              # ortalama gereken deneme
    ozet["basari_%1"] = [f"{gerekli[a][1]}/{C.N_RUNS}" for a in ozet.index]  # kaç koşu ulaştı

    # e) istatistiksel test (Mann-Whitney U): GA'ya göre
    try:
        from scipy.stats import mannwhitneyu                               # parametrik olmayan test
        a1, a2 = O.KARSILASTIR                                             # karşılaştırılacak yöntemler
        ref = df[df.yontem == a2]["J_egitim"]                              # referans: GA
        ozet[f"p_{a2}_ile"] = [np.nan if a == a2 else
                              mannwhitneyu(ref, df[df.yontem == a]["J_egitim"]).pvalue for a in ozet.index]
    except ImportError:
        print("scipy yok -> istatistiksel test atlandı (pip install scipy)")
    ozet.to_csv(os.path.join(C.SONUC_KLASORU, "ozet_egitim_training_summary.csv"))          # Bildiri Tablo 2
    print("\n", ozet.round(5))

    # f) son sınav: manuel + her yöntemin en iyi koşusu, 200 yeni senaryoda
    test = [dict(yontem="Manuel PID", **dict(zip(C.PARAM_ADLARI, MANUEL)), **test_metrikleri(MANUEL))]
    for ad in O.ALGORITMALAR:
        en = df[df.yontem == ad].sort_values("J_egitim").iloc[0]           # yöntemin en iyi koşusu
        x = en[C.PARAM_ADLARI].values.astype(float)                        # o koşunun 6 kazancı
        test.append(dict(yontem=ad, **dict(zip(C.PARAM_ADLARI, x)), **test_metrikleri(x)))
    dt = pd.DataFrame(test)
    dt.to_csv(os.path.join(C.SONUC_KLASORU, "test_sonuclari_test_results.csv"), index=False)  # Bildiri Tablo 3
    print("\n", dt.round(3).to_string(index=False))

    # g) yakınsama eğrileri
    with open(os.path.join(C.SONUC_KLASORU, "yakinsama_convergence.json"), "w") as fp:
        json.dump(dict(izgara=izgara.tolist(), egriler=egriler), fp)       # Şekil 2 için


if __name__ == "__main__":                                                 # python deneyler_experiments.py
    main()
