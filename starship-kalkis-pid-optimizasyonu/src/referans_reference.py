"""
referans_reference.py – IFT-6, IFT-7, IFT-8 Booster kalkış verisinden hedef profiller
Her uçuş için: v_ref(t), a_ref(t) = dv_ref/dt, sin(gamma)(t) (uçuş yolu açısı), h_ref(t), max-Q.
Adımlar
  1) Tekrarlanan zaman damgaları atılır, 1 s ızgaraya ara değer (OCR kaynaklı atlamalar).
  2) Hız km/saat → m/s; 7. derece polinomla yumuşatılır (OCR gürültüsü).
  3) İrtifa OCR'da tam sayı km → +500 m kaydırılıp 6. derece polinom.
  4) sin(gamma) = (dh/dt) / v ; düşük hızda (v < 60 m/s) dikey tırmanış: sin(gamma) = 1.
Çalıştırma: python referans_reference.py → her uçuş için özet satırı yazdırır.
"""
import os                                                                  # dosya yolları
import numpy as np                                                         # sayısal işlemler
import pandas as pd                                                        # CSV okuma
import ayarlar_config as C                                                         # ayar defteri


def _saniye(s):
    """'SS:DD:SS' metnini saniyeye çevirir."""
    h, m, x = map(int, s.split(":"))                                       # saat, dakika, saniye
    return h * 3600 + m * 60 + x                                           # toplam saniye


def ham_veri(ucus):
    """Bir uçuşun CSV'sini okur, tekrarları atar, ilk 160 saniyeyi döndürür."""
    d = pd.read_csv(os.path.join(C.VERI_KLASORU, f"{ucus}_ucus_verisi_flight_data.csv"), encoding="utf-8-sig")  # CSV'yi oku
    d = d.drop_duplicates(subset="time", keep="first")                     # aynı saniye iki kez → ilkini tut
    d["ts"] = d["time"].map(_saniye)                                       # zaman metnini saniyeye çevir
    d = d[d.ts <= C.T_END + 10].groupby("ts").first().reset_index()        # ilk 160 s, her saniyeden bir satır
    return d


class Referans:
    """Bir uçuşun hedef hız eğrisi ve yardımcı büyüklükleri."""

    def __init__(self, ucus):
        self.ad = ucus.upper().replace("IFT", "IFT-")                      # görünen ad: IFT-6 …
        d = ham_veri(ucus)                                                 # temizlenmiş ham veri
        t_ham = d.ts.values.astype(float)                                  # ham zamanlar
        self.t = np.arange(0, C.T_END + 1.0)                               # 0, 1, …, 150 s ızgarası
        self.v_veri = np.interp(self.t, t_ham, d.booster_speed.values / 3.6)  # hız, km/saat → m/s, ızgaraya
        self.h_veri = np.interp(self.t, t_ham, d.booster_altitude.values * 1000.0)  # irtifa km → m, ızgaraya

        self.pv = np.polyfit(self.t, self.v_veri, 7)                       # hıza 7. derece polinom (yumuşatma)
        ph = np.polyfit(self.t, self.h_veri + 500.0 * (self.h_veri > 0), 6)  # irtifaya +500 m kaydırıp 6. derece
        v = np.polyval(self.pv, self.t)                                    # yumuşak hız
        hd = np.polyval(np.polyder(ph), self.t)                            # dikey hız dh/dt
        sg = np.clip(hd / np.maximum(v, 1.0), 0.05, 1.0)                   # sin(gamma) = dikey hız / hız
        sg[v < 60] = 1.0                                                   # kalkış başında dik tırmanış
        self._sg = sg                                                      # sakla
        # referans irtifa: v·sin(gamma) integrali (OCR ile tutarlılık kontrolü için)
        self.h_ref = np.concatenate([[0], np.cumsum(0.5 * (v[1:] * sg[1:] + v[:-1] * sg[:-1]))])
        rho = C.RHO0 * np.exp(-self.h_ref / C.H_SCALE)                     # irtifadaki hava yoğunluğu
        q = 0.5 * rho * v ** 2                                             # dinamik basınç
        self.q_max, self.t_qmax = q.max(), self.t[q.argmax()]              # max-Q değeri ve anı

    def v_ref(self, t):
        """t anındaki hedef hız (polinomdan)."""
        return np.polyval(self.pv, t)

    def a_ref(self, t):
        """t anındaki hedef ivme = hız polinomunun türevi (ileri besleme için)."""
        return np.polyval(np.polyder(self.pv), t)

    def sin_gamma(self, t):
        """t anındaki uçuş yolu eğimi sin(gamma)."""
        return np.interp(t, self.t, self._sg)


REFS = [Referans(u) for u in C.UCUSLAR]                                    # 3 uçuşun hedef yolları

if __name__ == "__main__":                                                 # dosya doğrudan çalıştırılırsa
    for R in REFS:                                                         # her uçuş için bir özet satırı
        print(f"{R.ad}: v(150)={R.v_ref(150):.0f} m/s, h_ref(150)={R.h_ref[-1]/1e3:.1f} km "
              f"(OCR {R.h_veri[-1]/1e3:.0f} km), max-Q={R.q_max/1e3:.1f} kPa @ t={R.t_qmax:.0f} s, "
              f"hız RMSE={np.sqrt(np.mean((R.v_ref(R.t)-R.v_veri)**2)):.2f} m/s")
