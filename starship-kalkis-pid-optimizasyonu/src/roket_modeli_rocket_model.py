"""
roket_modeli_rocket_model.py – Sahte roket: kalkış fazında uçuş yolu boyunca 1-B dinamik model + PID gaz kolu + amaç fonksiyonu J
Vektörize: P aday PID × K senaryo aynı anda simüle edilir.

Durum denklemleri (uçuş yolu boyunca, yukarı/ileri pozitif):
    dv/dt = (T - D) / m - g * sin(gamma(t))      itki hızlandırır, hava ve yerçekimi yavaşlatır
    dh/dt = v * sin(gamma(t))                    gamma(t): uçuşun eğim programı (veriden)
    dm/dt = -T / (Isp(h) * g0)                   itki yakıt yakar
    dT/dt = (T_kom - T) / tau                    motor gecikmesi
    D     = 0.5 * rho(h) * Cd * A * v^2          hava direnci
    Isp(h)= Isp_vac - (Isp_vac - Isp_sl) * p(h)/p0   irtifayla artan verim

Kontrol (ileri beslemeli, kazanç programlı PID, hız takibi):
    e     = v_ref(t) - v_ölçülen                 pozitif: gereğinden yavaş
    a_kom = a_ref(t) + g sin(gamma) + D_tahmin/m_tahmin + Kp e + Ki ∫e + Kd de/dt
    u     = sat( m_tahmin * a_kom / (N * T_motor(h)), 0.40, 1.0 )
    faz-1 (t < 75 s, max-Q bölgesi) ve faz-2 (t ≥ 75 s) için ayrı [Kp, Ki, Kd]
"""
import numpy as np                                                         # sayısal işlemler
import ayarlar_config as C                                                         # ayar defteri
from referans_reference import REFS                                                  # 3 uçuşun hedef yolları

N_ADIM = int(round(C.T_END / C.DT)) + 1                                    # 150 s / 0,1 s + 1 = 1501 adım
_tg = np.arange(N_ADIM) * C.DT                                             # zaman ızgarası 0, 0,1, …, 150 s
VREF = np.array([R.v_ref(_tg) for R in REFS])                              # hedef hız tablosu (uçuş × adım)
AREF = np.array([R.a_ref(_tg) for R in REFS])                              # hedef ivme tablosu
SG = np.array([R.sin_gamma(_tg) for R in REFS])                            # eğim tablosu
QMAX_REF = np.array([R.q_max for R in REFS])                               # her uçuşun max-Q'su


def senaryolar(K, seed):
    """K adet belirsizlik senaryosu. Her senaryo sırayla bir uçuşa (IFT-6/7/8) atanır. Aynı seed → aynı senaryolar."""
    r = np.random.default_rng(seed)                                        # seed'li rastgele sayı üreteci
    ariza = r.random(K) < C.P_MOTOR_ARIZA                                  # hangi senaryoda motor arızası olacak (1/3)
    return dict(
        ucus=np.arange(K) % len(REFS),           # uçuş: 0,1,2,0,1,2,… → IFT-6, 7, 8 sırayla
        fm=r.uniform(0.97, 1.03, K),             # kalkış kütlesi çarpanı ±%3
        fcd=r.uniform(0.70, 1.30, K),            # sürükleme katsayısı çarpanı ±%30
        ft=r.uniform(0.97, 1.03, K),             # motor itki verimi ±%3
        t_ariza=np.where(ariza, r.uniform(20, 140, K), np.inf),   # arıza anı (20–140 s), yoksa sonsuz
        seed=int(seed),                          # gürültü üretimi için saklanır
    )


def _isp(h):
    """İrtifaya göre özgül itki: yukarı çıktıkça hava incelir, motor verimi artar."""
    return C.ISP_VAC - (C.ISP_VAC - C.ISP_SL) * np.exp(-np.maximum(h, 0) / C.H_SCALE)


def simule(gains, sc, kayit=False):
    """
    gains : (P,6) [Kp1, Ki1, Kd1, Kp2, Ki2, Kd2]; (P,3) verilirse iki fazda aynı kazanç
    sc    : senaryolar() çıktısı (K senaryo)
    dönüş : metrik sözlüğü, her biri (P,K); kayit=True ise zaman serileri de
    """
    gains = np.atleast_2d(np.asarray(gains, float))                        # en az 2 boyutlu dizi yap
    if gains.shape[1] == 3:                                                # tek PID verildiyse
        gains = np.hstack([gains, gains])                                  # iki faz için aynısını kullan
    P, K = gains.shape[0], len(sc["fm"])                                   # aday sayısı, senaryo sayısı
    G1 = [gains[:, i][:, None] for i in range(3)]                          # faz-1 kazançları (0–75 s)
    G2 = [gains[:, i][:, None] for i in range(3, 6)]                       # faz-2 kazançları (75–150 s)
    fl = sc["ucus"]                                                        # her senaryonun uçuşu

    m0 = np.ones((P, 1)) * (C.M0 * sc["fm"])[None, :]                      # senaryoya göre kalkış kütlesi
    cdA = C.CD * C.AREA * sc["fcd"][None, :]                               # senaryoya göre Cd·A
    ft = sc["ft"][None, :]                                                 # senaryoya göre itki verimi
    v = np.zeros((P, K)) + 0.5                                             # başlangıç hızı ~0
    h = np.zeros((P, K))                                                   # başlangıç irtifası 0
    m = m0.copy()                                                          # anlık kütle
    T = np.zeros((P, K))                                                   # anlık itki
    I = np.zeros((P, K))                                                   # hatanın integrali
    d_f = np.zeros((P, K))                                                 # filtrelenmiş türev
    e_prev = None                                                          # önceki adımın hatası
    u_prev = np.zeros((P, K))                                              # önceki gaz (TV için)
    noise = np.random.default_rng(sc["seed"] + 999).normal(0, C.SENSOR_NOISE, (N_ADIM, K))  # sabit sensör gürültüsü

    iae = np.zeros((P, K)); tv = np.zeros((P, K))                          # takip hatası ve gaz oynaması toplamları
    qmax = np.zeros((P, K)); amax = np.zeros((P, K))                       # en yüksek basınç ve ivme
    log = {"t": [], "v": [], "vref": [], "u": [], "q": [], "h": []} if kayit else None  # grafik için kayıt

    for k in range(N_ADIM - 1):                                            # her 0,1 s'lik adım (1500 kez)
        t = k * C.DT                                                       # şimdiki zaman
        Kp, Ki, Kd = G1 if t < C.T_SWITCH else G2                          # kazanç programlama: 75 s'de G1 → G2
        vref, aref, sg = VREF[fl, k][None, :], AREF[fl, k][None, :], SG[fl, k][None, :]  # hedef hız, ivme, eğim

        rho = C.RHO0 * np.exp(-np.maximum(h, 0) / C.H_SCALE)               # irtifadaki hava yoğunluğu
        isp = _isp(h)                                                      # irtifadaki motor verimi
        T_motor = C.MDOT_ENG * isp * C.G0                                  # motor başına anlık en büyük itki

        v_olc = v + noise[k][None, :]                                      # ölçülen (gürültülü) hız
        e = vref - v_olc                                                   # hata = hedef − ölçülen
        if e_prev is None:                                                 # ilk adımda
            e_prev = e.copy()                                              # türev sıfırdan başlasın
        d_f += ((e - e_prev) / C.DT - d_f) * (C.DT / (0.5 + C.DT))         # 0,5 s'lik filtreyle yumuşatılmış türev
        e_prev = e                                                         # sonraki adım için sakla

        m_hat = C.M0 - (m0 - m)                                            # kontrolcünün kütle tahmini
        D_hat = 0.5 * rho * C.CD * C.AREA * v_olc ** 2                     # kontrolcünün hava direnci tahmini
        a_cmd = aref + C.G0 * sg + D_hat / m_hat + Kp * e + Ki * I + Kd * d_f  # istenen ivme = FF + PID
        u_raw = m_hat * a_cmd / (C.N_ENG * T_motor)                        # gereken gaz oranı
        u = np.clip(u_raw, C.THROTTLE_MIN, 1.0)                            # %40 ile %100 arasına sınırla
        doyma = ((u_raw > 1.0) & (e > 0)) | ((u_raw < C.THROTTLE_MIN) & (e < 0))  # sınıra dayandı ve itiyor mu?
        I += np.where(doyma, 0.0, e * C.DT)                                # anti-windup: doymadaysa integrali durdur

        n_aktif = C.N_ENG - (t >= sc["t_ariza"])[None, :]                  # motor arızası: 33 → 32
        T_cmd = u * n_aktif * T_motor * ft                                 # motor komutu
        T += (T_cmd - T) * (C.DT / C.TAU_ENGINE)                           # motor gecikmesi: farkın yarısı kapanır

        D = 0.5 * rho * cdA * v ** 2                                       # gerçek hava direnci
        a = (T - D) / m - C.G0 * sg                                        # ivme
        q = 0.5 * rho * v ** 2                                             # dinamik basınç

        iae += np.abs(v - vref) * C.DT                                     # takip hatasını biriktir
        tv += np.abs(u - u_prev) * (k > 0)                                 # gaz değişimini biriktir
        qmax = np.maximum(qmax, q)                                         # en yüksek basınç
        amax = np.maximum(amax, (T - D) / m / C.G0)                        # algılanan en yüksek ivme (g)
        u_prev = u                                                         # sonraki adım için sakla

        v = v + a * C.DT                                                   # yeni hız
        h = h + v * sg * C.DT                                              # yeni irtifa
        m = m - T / (isp * C.G0) * C.DT                                    # yeni kütle (yakıt yandı)

        if kayit and k % 5 == 0:                                           # grafik için her 0,5 s'de bir kaydet
            for n_, x_ in zip(("t", "v", "vref", "u", "q", "h"), (t, v, vref, u, q, h)):
                log[n_].append(np.array(x_, copy=True))

    out = dict(iae=iae, tv=tv, qmax=qmax, amax=amax, yakit=m0 - m,          # sonuç sözlüğü
               v_son=v, v_son_ref=VREF[fl, -1][None, :] * np.ones((P, 1)),  # 150. s'deki hız ve hedefi
               q_lim=C.Q_MARJ * QMAX_REF[fl][None, :] * np.ones((P, 1)), h_son=h)  # max-Q sınırı
    if kayit:
        out["log"] = {k_: np.array(v_) for k_, v_ in log.items()}          # zaman serilerini diziye çevir
    return out


def maliyet(r):
    """
    Amaç fonksiyonu J (küçük = iyi), senaryo ortalaması:
    J = IAE/300 + ((v_son - v_ref_son)/10)^2 + 20*max(0, qmax/q_lim - 1) + 0.5*yakıt/2700 t + 0.003*TV(u)
    """
    J = (r["iae"] / 300.0                                                  # takip hatası
         + ((r["v_son"] - r["v_son_ref"]) / 10.0) ** 2                     # 150. s'deki hız hatası
         + 20.0 * np.maximum(0.0, r["qmax"] / r["q_lim"] - 1.0)            # ceza: max-Q sınırı aşıldı
         + 0.5 * r["yakit"] / 2700e3                                       # harcanan yakıt
         + 0.003 * r["tv"])                                                # gaz kolu oynaması
    return J.mean(axis=1)                                                  # 9 senaryonun ortalaması


class AmacFonksiyonu:
    """Puan makinesi: sabit eğitim senaryolarıyla J hesaplar ve deneme sayısını sayar."""

    def __init__(self, K=C.N_SCEN, seed=0):
        self.sc = senaryolar(K, seed)                                      # 9 senaryo bir kez üretilir (seed 0)
        self.n_eval = 0                                                    # deneme sayacı

    def __call__(self, X):
        X = np.atleast_2d(X)                                               # tek set de verilse tablo yap
        self.n_eval += X.shape[0]                                          # her set 1 deneme
        return maliyet(simule(X, self.sc))                                 # uçuş → J


if __name__ == "__main__":                                                 # deneme bölümü
    import time
    f = AmacFonksiyonu()                                                   # puan makinesi
    X = np.array([[0.5, 0.05, 0.05, 0.5, 0.05, 0.05], [1, .2, .2, 1, .2, .2],     # 4 elle ayar:
                  [0, 0, 0, 0, 0, 0], [4, 1.5, 1.5, 4, 1.5, 1.5]])                 # yumuşak, orta, PID kapalı, sert
    t0 = time.time(); J = f(X); print("J =", J.round(4), f"{time.time()-t0:.2f}s")  # puanlar
    r = simule(X, f.sc)                                                    # ayrıntılı sonuçlar
    for k in ["iae", "qmax", "q_lim", "yakit", "tv", "amax"]:              # 9 senaryo ortalamaları
        print(k, np.round(r[k].mean(1), 1))
    print("v_son - ref:", np.round((r["v_son"] - r["v_son_ref"])[:, :9], 1))  # 150. s'deki hız hatası (4 × 9)
