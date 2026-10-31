"""
grafikler_figures.py – Bildiri şekilleri (önce deneyler_experiments.py çalıştırılmalı)
Çalıştırma: python grafikler_figures.py  →  ../sonuclar_results/sekiller_figures/*.png (300 dpi)
"""
import json                                                                # yakınsama verisi
import os                                                                  # dosya yolları
import numpy as np                                                         # sayısal işlemler
import pandas as pd                                                        # tablolar
import matplotlib
matplotlib.use("Agg")                                                      # ekransız çizim (dosyaya kaydet)
import matplotlib.pyplot as plt                                            # grafik kütüphanesi
import ayarlar_config as C                                                         # ayar defteri
import roket_modeli_rocket_model as M                                                          # sahte roket
from referans_reference import REFS                                                  # 3 uçuşun hedef yolları
from deneyler_experiments import MANUEL, N_TEST, TEST_SEED                             # manuel PID ve test ayarları

RENK = {"GA": "#2a6fdb", "ACO": "#1e9e5a", "Rastgele Arama": "#8a8a8a", "Manuel PID": "#c0392b"}   # yöntem renkleri
UCUS_RENK = ["#7b3fa0", "#e07b00", "#222222"]                              # IFT-6, 7, 8 renkleri
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,  # ortak grafik stili
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight"})
os.makedirs(C.SEKIL_KLASORU, exist_ok=True)                                # sekiller/ klasörünü oluştur


def kaydet(fig, ad):
    """Şekli PNG olarak kaydeder ve kapatır."""
    fig.savefig(os.path.join(C.SEKIL_KLASORU, ad))
    plt.close(fig)
    print("kaydedildi:", ad)


df = pd.read_csv(os.path.join(C.SONUC_KLASORU, "tum_kosular_all_runs.csv"))        # tüm koşular
with open(os.path.join(C.SONUC_KLASORU, "yakinsama_convergence.json")) as fp:
    yk = json.load(fp)                                                     # yakınsama eğrileri
en_iyi = {a: df[df.yontem == a].sort_values("J_egitim").iloc[0][C.PARAM_ADLARI].values.astype(float)
          for a in df.yontem.unique()}                                     # her yöntemin en iyi 6 kazancı
EN = "GA"                                                                  # şekil 4–5'te gösterilen yöntem

# ---- Şekil 1: üç uçuşun kalkış verisi
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
for R, c in zip(REFS, UCUS_RENK):                                          # her uçuş için
    ax[0].plot(R.t, R.v_veri, ".", ms=2.5, color=c, alpha=0.6)             # OCR hız noktaları
    ax[0].plot(R.t, R.v_ref(R.t), color=c, lw=1.5, label=R.ad)             # yumuşatılmış hedef hız
    ax[1].step(R.t, R.h_veri / 1e3, where="post", color=c, alpha=0.4, lw=0.8)  # OCR irtifa (merdiven)
    ax[1].plot(R.t, R.h_ref / 1e3, color=c, lw=1.5, label=R.ad)            # model irtifası (rampa)
    rho = C.RHO0 * np.exp(-R.h_ref / C.H_SCALE)                            # hava yoğunluğu
    ax[2].plot(R.t, 0.5 * rho * R.v_ref(R.t) ** 2 / 1e3, color=c, lw=1.5, label=R.ad)  # dinamik basınç
ax[0].set(xlabel="Süre [s]", ylabel="Hız [m/s]", title="(a) Hız: OCR (nokta) ve yumuşatılmış")
ax[1].set(xlabel="Süre [s]", ylabel="İrtifa [km]", title="(b) İrtifa: OCR (basamak) ve model")
ax[2].set(xlabel="Süre [s]", ylabel="Dinamik basınç q [kPa]", title="(c) Dinamik basınç (max-Q)")
for a in ax:
    a.axvline(C.T_SWITCH, color="k", ls=":", lw=0.8)                       # 75 s faz geçişi
    a.legend(frameon=False, fontsize=8)
kaydet(fig, "sekil1_kalkis_verisi_ascent_data.png")

# ---- Şekil 2: yakınsama (10 koşu ortalaması ± std)
iz = np.array(yk["izgara"])                                                # deneme sayısı ekseni
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for ad, E in yk["egriler"].items():
    E = np.array(E)                                                        # 10 koşu × 60 nokta
    m, s = E.mean(0), E.std(0)                                             # ortalama ve sapma
    ax.plot(iz, m, color=RENK[ad], lw=1.8, label=ad)                       # ortalama çizgi
    ax.fill_between(iz, m - s, m + s, color=RENK[ad], alpha=0.15)          # ± std gölge
lo = df.J_egitim.min()                                                     # en iyi J (eksen alt sınırı için)
ax.set(xlabel="Amaç fonksiyonu değerlendirme sayısı", ylabel="En iyi J (10 koşu ort. ± std)",
       ylim=(lo * 0.98, lo * 1.5), title="Yakınsama: GA, ACO ve rastgele arama")
ax.legend(frameon=False)
kaydet(fig, "sekil2_yakinsama_convergence.png")

# ---- Şekil 3: kutu grafiği (10 koşunun son J dağılımı)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
adlar = list(yk["egriler"].keys())                                         # yöntem adları
bp = ax.boxplot([df[df.yontem == a]["J_egitim"].values for a in adlar], patch_artist=True, widths=0.5)
ax.set_xticks(range(1, len(adlar) + 1), adlar)
for p, a in zip(bp["boxes"], adlar):
    p.set_facecolor(RENK[a]); p.set_alpha(0.5)                             # renklendir
ax.set(ylabel="Son J değeri", title=f"{C.N_RUNS} bağımsız koşu sonunda J dağılımı")
kaydet(fig, "sekil3_kutu_grafigi_boxplot.png")

# ---- Şekil 4: motor arızalı IFT-8 senaryosu (60. s'de bir motor durur), manuel vs GA vs ACO
sen = dict(ucus=np.array([2]), fm=np.ones(1), fcd=np.ones(1), ft=np.ones(1), t_ariza=np.array([60.0]), seed=7)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for ad, x in [("Manuel PID", MANUEL), ("GA", en_iyi["GA"]), ("ACO", en_iyi["ACO"])]:
    L = M.simule(x[None], sen, kayit=True)["log"]                          # zaman serileriyle simüle et
    tt = L["t"]
    ax[0].plot(tt, (L["v"] - L["vref"])[:, 0, 0], color=RENK[ad], lw=1, label=ad)  # hız hatası
    ax[1].plot(tt, L["u"][:, 0, 0] * 100, color=RENK[ad], lw=1, label=ad)  # gaz kolu (%)
    ax[2].plot(tt, L["q"][:, 0, 0] / 1e3, color=RENK[ad], lw=1.2, label=ad)  # dinamik basınç
ax[2].axhline(C.Q_MARJ * REFS[2].q_max / 1e3, color="k", ls="--", lw=1, label="max-Q sınırı")
for a in ax:
    a.axvline(60, color="#c0392b", ls=":", lw=1)                           # arıza anı
ax[0].text(61, ax[0].get_ylim()[1] * 0.8, "motor arızası", color="#c0392b", fontsize=8)
ax[0].set(xlabel="Süre [s]", ylabel="v − v_ref [m/s]", title="(a) Hız takip hatası")
ax[1].set(xlabel="Süre [s]", ylabel="Gaz kolu [%]", title="(b) Kontrol sinyali")
ax[2].set(xlabel="Süre [s]", ylabel="q [kPa]", title="(c) Dinamik basınç")
for a in ax:
    a.legend(frameon=False, fontsize=8)
kaydet(fig, "sekil4_motor_arizasi_engine_failure.png")

# ---- Şekil 5: 200 test senaryosu (Monte Carlo)
sc = M.senaryolar(N_TEST, TEST_SEED)                                       # 200 test senaryosu
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for ad, x in [("Manuel PID", MANUEL), ("GA", en_iyi["GA"]), ("ACO", en_iyi["ACO"])]:
    r = M.simule(x[None], sc)                                              # 200 uçuş
    dv = (r["v_son"] - r["v_son_ref"])[0]                                  # 150. s'deki hız hataları
    ax[0].hist(dv, bins=np.linspace(-2, 2, 41), color=RENK[ad], alpha=0.55, label=ad)  # son hız hatası
    ax[1].hist(r["iae"][0], bins=30, color=RENK[ad], alpha=0.55, label=ad)  # toplam takip hatası
ax[0].set(xlabel="t = 150 s'de hız hatası [m/s]", ylabel="Senaryo sayısı", title=f"(a) Son hız hatası ({N_TEST} senaryo)")
ax[1].set(xlabel="IAE [m]", ylabel="Senaryo sayısı", title="(b) Toplam takip hatası")
for a in ax:
    a.legend(frameon=False, fontsize=8)
kaydet(fig, "sekil5_monte_carlo_testi_monte_carlo_test.png")
