"""
ayarlar_config.py – Ayar defteri (Super Heavy + Starship kalkış fazı)
Bütün sabitler ve ayarlar burada; diğer dosyalar buradan okur.
Değerler açık kaynaklardan alınan YAKLAŞIK değerlerdir (SpaceX yayımlamaz);
belirsizlik Monte Carlo senaryolarıyla modele eklenir.
"""
import os                                                                  # dosya yolu işlemleri

# --- Dosya yolları ---
KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))         # proje ana klasörü (src'nin bir üstü)
_ADAYLAR = [os.path.join(KOK, "veri_data"),                                    # CSV'nin aranacağı 4 yer: veri/,
            KOK,                                                           # proje klasörü,
            os.path.dirname(os.path.abspath(__file__)),                    # src/,
            os.path.dirname(os.path.dirname(KOK))]                         # ve iki üst klasör (Optimizasyon)
VERI_KLASORU = next((k for k in _ADAYLAR                                   # ift8_results.csv hangisindeyse onu seç
                     if os.path.exists(os.path.join(k, "ift8_ucus_verisi_flight_data.csv"))), None)
if VERI_KLASORU is None:                                                   # hiçbirinde yoksa açık hata ver
    raise FileNotFoundError("ift6/7/8_results.csv bulunamadı; dosyaları 'veri' klasörüne koyun.")
UCUSLAR = ["ift6", "ift7", "ift8"]                                         # kullanılan 3 uçuş (ift6_results.csv …)
SONUC_KLASORU = os.path.join(KOK, "sonuclar_results")                             # CSV / JSON çıktıları
SEKIL_KLASORU = os.path.join(SONUC_KLASORU, "sekiller_figures")                   # şekiller (PNG)

# --- Fiziksel sabitler ---
G0 = 9.80665              # yerçekimi ivmesi, m/s²
RHO0 = 1.225              # deniz seviyesinde hava yoğunluğu, kg/m³
H_SCALE = 8500.0          # atmosfer ölçek yüksekliği, m

# --- Araç (Super Heavy + Starship yığını) ---
M0 = 5000e3               # kalkış kütlesi, kg (booster ~3450 t + gemi ~1550 t)
DIAMETER = 9.0            # gövde çapı, m
AREA = 3.1416 * (DIAMETER / 2) ** 2   # kesit alanı ≈ 63,6 m²
CD = 0.5                  # ortalama sürükleme katsayısı (burun önde)

# --- Raptor 2 (deniz seviyesi nozullu) × 33 ---
N_ENG = 33                # motor sayısı
MDOT_ENG = 705.0          # motor başına en büyük kütle debisi, kg/s (2,26 MN / (327 s · g0))
ISP_SL = 327.0            # deniz seviyesi özgül itki, s
ISP_VAC = 350.0           # vakumda (SL nozul) yaklaşık özgül itki, s
THROTTLE_MIN = 0.40       # en düşük gaz oranı
TAU_ENGINE = 0.20         # motor tepki gecikmesi (zaman sabiti), s

# --- Uçuş fazları ---
T_END = 150.0             # analiz sonu: kalkışın ilk 150 saniyesi
T_SWITCH = 75.0           # faz-1: kalkış + max-Q bölgesi | faz-2: üst atmosferde hızlanma

# --- Simülasyon ---
DT = 0.1                  # zaman adımı, s
SENSOR_NOISE = 1.0        # hız ölçüm gürültüsü (1-sigma), m/s
Q_MARJ = 1.05             # max-Q sınırı = 1,05 × referans uçuşun max-Q'su
P_MOTOR_ARIZA = 1 / 3     # bir senaryoda tek motor arızası olasılığı (33 → 32)

# --- PID arama sınırları [Kp1, Ki1, Kd1, Kp2, Ki2, Kd2] ---
LOWER = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                                    # alt sınırlar
UPPER = [2.0, 0.5, 1.0, 2.0, 0.5, 1.0]                                    # üst sınırlar: Kp ≤ 2, Ki ≤ 0,5, Kd ≤ 1
PARAM_ADLARI = ["Kp1", "Ki1", "Kd1", "Kp2", "Ki2", "Kd2"]                 # 6 tasarım değişkeni

# --- Optimizasyon bütçesi (adil karşılaştırma için her yöntemde aynı) ---
POP = 30                  # bir nesildeki set sayısı
ITER = 60                 # 30 × 60 = 1800 değerlendirme / koşu
N_RUNS = 10               # her yöntem için bağımsız koşu sayısı (seed 1…10)
N_SCEN = 9                # eğitim senaryosu: her uçuştan 3 (3 × 3 = 9)
