"""
ablasyon_ablation.py – Ablasyon çalışması: kazanç programlamanın (iki faz) katkısı
  Tek faz: tüm uçuş boyunca aynı [Kp, Ki, Kd] (3 parametre)
  İki faz: t < 75 s → G1, t ≥ 75 s → G2 (6 parametre, ana çalışma)
Çalıştırma: önce deneyler_experiments.py, sonra python ablasyon_ablation.py → ../sonuclar_results/ablasyon_ablation.csv
"""
import os                                                                  # dosya yolları
import numpy as np                                                         # sayısal işlemler
import pandas as pd                                                        # tablolar
import ayarlar_config as C                                                         # ayar defteri
import roket_modeli_rocket_model as M                                                          # sahte roket
import optimizasyon_optimization as O                                                   # GA
from deneyler_experiments import test_metrikleri                                       # 200 senaryoluk son sınav

if __name__ == "__main__":
    f = M.AmacFonksiyonu(C.N_SCEN, seed=0)                                 # 9 eğitim senaryosu
    f3 = lambda X: f(np.hstack([np.atleast_2d(X)] * 2))                    # 3 kazancı iki fazda da kullan
    lb, ub = np.array(C.LOWER[:3]), np.array(C.UPPER[:3])                  # tek fazın sınırları
    sonuc = [O.ga(f3, lb, ub, pop=C.POP, iters=C.ITER, seed=s) for s in range(1, 6)]  # GA ile 5 koşu
    x3 = min(sonuc, key=lambda r: r[1])[0]                                 # en iyi tek faz ayarı
    tek = np.hstack([x3, x3])                                              # 6 sayılık biçime çevir
    df = pd.read_csv(os.path.join(C.SONUC_KLASORU, "tum_kosular_all_runs.csv"))    # ana çalışmanın koşuları
    iki = df[df.yontem == "GA"].sort_values("J_egitim").iloc[0][C.PARAM_ADLARI].values.astype(float)  # en iyi iki faz
    satirlar = []
    for ad, g, n in (("Tek faz", tek, 3), ("İki faz", iki, 6)):            # iki yapıyı karşılaştır
        r = test_metrikleri(g)                                             # 200 test senaryosu
        satirlar.append(dict(yapi=ad, parametre=n, **dict(zip(C.PARAM_ADLARI, g)),
                             J_egitim=float(f(g[None])[0]), **r))          # eğitim ve test sonuçları
    out = pd.DataFrame(satirlar)
    out.to_csv(os.path.join(C.SONUC_KLASORU, "ablasyon_ablation.csv"), index=False)  # kaydet
    print(out[["yapi", "parametre", "Kp1", "Kp2", "J_egitim", "J_test", "IAE_ort", "TV_ort", "maxQ_ihlal"]].round(4))
