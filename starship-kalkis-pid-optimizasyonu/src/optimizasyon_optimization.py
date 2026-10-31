"""
optimizasyon_optimization.py – Genetik Algoritma (GA), Karınca Kolonisi (ACO_R) ve rastgele arama (sıfırdan, yalnızca numpy)
Arayüz: algoritma(f, lb, ub, pop, iters, seed) -> (x_best, f_best, egri)
  f : (P,D) -> (P,) toplu amaç fonksiyonu (puan makinesi)
Toplam değerlendirme bütçesi her yöntemde pop*iters = 1800 → adil karşılaştırma.
Bölüm 2: Karınca Kolonisi Optimizasyonu (ACO_R) eklendi; GA ile aynı bütçede karşılaştırılır.
"""
import numpy as np                                                         # sayısal işlemler


# ---------------------------------------------------------------- GA
def ga(f, lb, ub, pop=30, iters=60, seed=0, pc=0.9, pm=0.15, elit=2, tur=3, alfa=0.5):
    """Gerçel kodlu Genetik Algoritma (Holland, 1975; Goldberg, 1989).
    Turnuva seçimi (k=3) + BLX-α çaprazlama (Eshelman & Schaffer, 1993)
    + Gauss mutasyonu + elitizm (en iyi 2 birey korunur)."""
    r = np.random.default_rng(seed)                                        # seed'li rastgele sayı üreteci
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)                  # sınırları diziye çevir
    D, aralik = len(lb), ub - lb                                           # 6 sayı, her birinin aralığı
    X = lb + r.random((pop, D)) * aralik                                   # 30 rastgele set (başlangıç nesli)
    F = f(X)                                                               # 30 seti puanla (30 deneme)
    egri = [F.min()]                                                       # yakınsama eğrisi
    n_ev, butce = pop, pop * iters                                         # deneme sayacı ve bütçe (1800)
    while n_ev < butce:                                                    # bütçe bitene kadar nesil üret
        sira = np.argsort(F)                                               # setleri en iyiden kötüye sırala

        def secim():
            """Turnuva: rastgele 3 set çek, J'si en küçük olanı döndür."""
            aday = r.integers(0, pop, tur)                                 # 3 rastgele satır numarası
            return X[aday[np.argmin(F[aday])]]                             # en iyisinin 6 sayısı

        n_c = min(pop - elit, butce - n_ev)                                # yapılacak çocuk: 28 (son nesilde kalan kadar)
        cocuk = []                                                         # çocuk listesi
        while len(cocuk) < n_c:                                            # 28 çocuk olana kadar
            p1, p2 = secim(), secim()                                      # iki turnuva → anne, baba
            if r.random() < pc:                                            # %90 ihtimalle çaprazla
                lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)            # her sütunda küçük ve büyük değer
                c = r.uniform(lo - alfa * (hi - lo), hi + alfa * (hi - lo))  # BLX-α: aralıktan (+%50 pay) rastgele seç
            else:
                c = p1.copy()                                              # %10: çocuk annenin kopyası
            mut = r.random(D) < pm                                         # her sayı %15 ihtimalle mutasyona uğrar
            c[mut] += r.normal(0, 0.1 * aralik[mut])                       # seçilenlere küçük değişim (aralığın %10'u)
            cocuk.append(np.clip(c, lb, ub))                               # sınır kontrolü, listeye ekle
        Xc = np.array(cocuk)                                               # çocuk tablosu
        Fc = f(Xc)                                                         # çocukları puanla (+28 deneme)
        n_ev += n_c                                                        # sayacı artır
        kalan = sira[elit:pop - n_c + elit] if n_c < pop - elit else sira[:0]  # son nesilde eskilerden kalanlar
        X = np.vstack([X[sira[:elit]], X[kalan], Xc])                      # yeni nesil: 2 şampiyon + (kalan) + çocuklar
        F = np.concatenate([F[sira[:elit]], F[kalan], Fc])                 # puanları da aynı sırayla
        egri.append(F.min())                                               # bu neslin en iyi J'si
    i = np.argmin(F)                                                       # son nesildeki en iyi set
    return X[i], F[i], np.minimum.accumulate(egri)                         # en iyi set, en iyi J, eğri


# ---------------------------------------------------------------- ACO_R (Bölüm 2 – yeni)
def aco(f, lb, ub, pop=30, iters=60, seed=0, k=30, q=0.2, xi=0.85):
    """Sürekli uzay için Karınca Kolonisi Optimizasyonu, ACO_R (Socha & Dorigo, 2008).
    Feromon = en iyi k çözümden oluşan sıralı arşiv. Her karınca:
      1) arşivden bir rehber çözüm l seçer, olasılık ∝ w_l = exp(-(l-1)² / (2 q² k²))
      2) her boyutta Gauss örnekler: x_i ~ N(s_l,i , σ_i),
         σ_i = ξ · Σ_e |s_e,i − s_l,i| / (k − 1)   (arşivin o boyuttaki yayılımı)
    Yeni çözümler arşive eklenir, yalnızca en iyi k tanesi kalır (feromon güncelleme + buharlaşma).
    q küçük → iyi çözümlerin etrafında ara (sömürü); ξ büyük → yavaş daralma (keşif)."""
    r = np.random.default_rng(seed)                                        # seed'li rastgele sayı üreteci
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)                  # sınırları diziye çevir
    D = len(lb)                                                            # 6 kazanç
    A = lb + r.random((k, D)) * (ub - lb)                                  # başlangıç arşivi: 30 rastgele set
    FA = f(A)                                                              # arşivi puanla (30 deneme)
    s = np.argsort(FA); A, FA = A[s], FA[s]                                # arşivi en iyiden kötüye sırala
    rank = np.arange(k)                                                    # sıra numaraları 0…29
    w = np.exp(-rank ** 2 / (2 * (q * k) ** 2))                            # sıra ağırlığı (iyi sıra → büyük w)
    p = w / w.sum()                                                        # seçilme olasılıkları
    egri = [FA[0]]                                                         # yakınsama eğrisi
    n_ev, butce = k, pop * iters                                           # deneme sayacı ve bütçe (1800)
    while n_ev < butce:                                                    # bütçe bitene kadar
        m = min(pop, butce - n_ev)                                         # bu turdaki karınca sayısı (30)
        l = r.choice(k, size=m, p=p)                                       # her karınca bir rehber seçer
        sigma = xi * np.abs(A[None, :, :] - A[l][:, None, :]).sum(axis=1) / (k - 1)  # rehber etrafındaki yayılım
        Xn = np.clip(r.normal(A[l], sigma + 1e-12), lb, ub)                # yeni çözümleri örnekle, sınıra çek
        Fn = f(Xn)                                                         # yeni çözümleri puanla (+30 deneme)
        n_ev += m                                                          # sayacı artır
        A = np.vstack([A, Xn]); FA = np.concatenate([FA, Fn])              # arşive ekle (60 çözüm)
        s = np.argsort(FA)[:k]                                             # en iyi 30'u seç
        A, FA = A[s], FA[s]                                                # kötüler silinir (buharlaşma)
        egri.append(FA[0])                                                 # bu turun en iyi J'si
    return A[0], FA[0], np.array(egri)                                     # en iyi set, en iyi J, eğri


# ---------------------------------------------------------------- Rastgele arama (taban çizgisi)
def rastgele(f, lb, ub, pop=30, iters=60, seed=0):
    """Hiç öğrenmeyen karşılaştırma yöntemi: her turda 30 rastgele set dene, en iyisini tut."""
    r = np.random.default_rng(seed)                                        # seed'li rastgele sayı üreteci
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)                  # sınırlar
    gF, g, egri = np.inf, None, []                                         # en iyi J, en iyi set, eğri
    for _ in range(iters):                                                 # 60 tur
        X = lb + r.random((pop, len(lb))) * (ub - lb)                      # 30 rastgele set
        F = f(X)                                                           # puanla
        if F.min() < gF:                                                   # yeni rekor mu?
            gF, g = F.min(), X[np.argmin(F)].copy()                        # sakla
        egri.append(gF)                                                    # eğriye ekle
    return g, gF, np.array(egri)


ALGORITMALAR = {"GA": ga, "ACO": aco, "Rastgele Arama": rastgele}                      # deneyler_experiments.py bu listeyi çalıştırır
KARSILASTIR = ("ACO", "GA")                                                # Mann-Whitney testi: GA'ya göre
