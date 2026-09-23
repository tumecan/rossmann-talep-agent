# Rossman/lstm_keras_tuner.py — Keras Tuner ile KAPSAMLI hiperparametre arama
#
# ============================================================================
# ARAMA UZAYI — kurs notundaki (RNN_LSTM_GRU rehberi, bolum 8) tum
# hiperparametreler + bu probleme ozgu birkac ek.
#
#   MIMARI
#     num_layers          1-5 tekrarli katman (derinlik)
#     units_i             32-512, step 32 (her katmanin kapasitesi)
#     hucre_tipi          LSTM / GRU / SimpleRNN
#     cift_yonlu          Bidirectional sarmalayici
#     return_sequences    ELLE ARANMAZ — kuralla belirlenir:
#                         ustunde baska tekrarli katman varsa True,
#                         son katmansa False. En cok hata yapilan yer burasi.
#
#   DUZENLILESTIRME (overfitting frenleri)
#     dropout_i           0.0-0.5
#     l2_i                1e-5..1e-2, LOG olcek (agirlik cezasi)
#     batch_norm          acik/kapali
#
#   BASLANGIC
#     init_i              glorot_uniform / he_uniform / random_normal
#
#   OPTIMIZASYON
#     optimizer           adam / sgd / rmsprop / adamax
#     learning_rate       1e-4..1e-2, LOG olcek (en etkili hiperparametre)
#     momentum            sadece sgd secilirse (0.0-0.9)
#     clipnorm            0 (kapali) / 1.0 (gradyan kirpma)
#     batch_size          8 / 16 / 32 / 64 — HyperModel.fit override ile
#
# ARAMA UZAYINDAN BILINCLI OLARAK CIKARILANLAR (butce karari)
#     recurrent_dropout   Sifirdan buyuk olunca TensorFlow'un hizlandirilmis
#                         LSTM/GRU cekirdegi devre disi kalir, egitim 5-10 kat
#                         yavaslar. CPU'da makul surede arama yapabilmek icin
#                         0'da sabitlendi. Duzenlilestirme dropout + l2 ile
#                         zaten saglaniyor.
#     aktivasyon          tanh'ta sabit. relu de ayni hizli yolu kapatiyor,
#                         ayrica tekrarli aglarda patlayan gradyan riskini
#                         artiriyor. tanh hem varsayilan hem hizli.
#     SEQ_LEN             Pencere uzunlugu VERI SEKLINI degistirir
#                         (X: [n, SEQ_LEN, 1]), tuner ayni veri uzerinde
#                         calistigi icin iceride aranamaz. Disaridan verilir:
#                             python lstm_keras_tuner.py --seq 7
#                             python lstm_keras_tuner.py --seq 28
#
# NEDEN LOG OLCEK?
# learning_rate ve l2 genis araliklarda yasar (1e-5 ... 1e-2). Dogrusal
# orneklemede 1e-5 ile 1e-4 arasi neredeyse hic ornek almaz. Log olcek
# kucuk degerlere de buyuk degerler kadar adil sans verir.
#
# NEDEN RandomSearch / Hyperband?
# GridSearch bu uzayda kombinatoryal olarak patlar. RandomSearch max_trials
# kadar rastgele dener. Hyperband ise kotu konfigurasyonlari birkac epoch'ta
# eleyip butceyi iyilere kaydirir — CPU'da belirgin sekilde daha hizlidir.
#
# SIZINTI NOTU
# Secim yalnizca VALIDATION kaybina gore yapilir. Test seti arama boyunca
# hic gorulmez; sadece en sonda, tek sefer degerlendirilir.
#
# Calistirma:
#     python lstm_keras_tuner.py --deneme 25 --hyperband   # onerilen
#     python lstm_keras_tuner.py --deneme 30               # RandomSearch
#     python lstm_keras_tuner.py --hizli --deneme 5        # boru hatti testi
#     python lstm_keras_tuner.py --seq 28 --hyperband      # farkli pencere
# ============================================================================
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

# --- CPU paralelligi: 0 = mevcut tum cekirdekleri kullan -------------------
# Not: RNN egitimi zaman adimlari boyunca SERI ilerler, paralellestirilemez.
# Bu ayarin kazanci sinirlidir; asil hizlanma arama uzayi secimlerinden gelir.
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, GRU, SimpleRNN, Dense, Dropout,
                                     Input, BatchNormalization, Bidirectional)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

from metrics import tum_metrikler

np.random.seed(42); tf.random.set_seed(42)

# ================== 0) CLI AYARLARI ==================
argv = sys.argv
SEQ_LEN    = int(argv[argv.index("--seq") + 1]) if "--seq" in argv else 14
MAX_DENEME = int(argv[argv.index("--deneme") + 1]) if "--deneme" in argv else 25
HIZLI      = "--hizli" in argv
HYPERBAND  = "--hyperband" in argv

# Hizli modda uzay daraltilir: gelistirme sirasinda tum boru hattini
# dakikalar icinde test etmek icin. Sunuma girecek kosu tam uzayla yapilmali.
MAX_LAYERS = 2 if HIZLI else 5
MAX_UNITS  = 128 if HIZLI else 512

print(f"SEQ_LEN={SEQ_LEN} | max_trials={MAX_DENEME} | "
      f"katman<= {MAX_LAYERS} | units<= {MAX_UNITS} | "
      f"tuner={'Hyperband' if HYPERBAND else 'RandomSearch'}")

# ================== 1) VERİ (univariate, diğer LSTM'lerle aynı) ==================
train = pd.read_csv("Rossman/data/train.csv", low_memory=False)
train["Date"] = pd.to_datetime(train["Date"])

s1 = train[(train["Store"] == 1) & (train["Open"] == 1)].copy()
s1 = s1.sort_values("Date").reset_index(drop=True)
seri = s1[["Date", "Sales"]].copy()

TEST_DAYS = 42
cutoff = seri["Date"].max() - pd.Timedelta(days=TEST_DAYS)
train_seri = seri[seri["Date"] <= cutoff]
test_seri  = seri[seri["Date"] >  cutoff]

# Scaler SADECE train'e fit edilir — test istatistigi sizdirilmez.
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_seri[["Sales"]])
test_scaled  = scaler.transform(test_seri[["Sales"]])

def make_seq(data, n):
    X, y = [], []
    for i in range(len(data) - n):
        X.append(data[i:i+n]); y.append(data[i+n])
    return np.array(X), np.array(y)

X_train, y_train = make_seq(train_scaled, SEQ_LEN)
X_test,  y_test  = make_seq(test_scaled,  SEQ_LEN)
print(f"Test seti: {len(test_seri)} gün | Dizi modeli tahmini: {len(X_test)} gün "
      f"(ilk {SEQ_LEN} gün girdi olarak tüketildi)")

HUCRELER = {"LSTM": LSTM, "GRU": GRU, "SimpleRNN": SimpleRNN}


# ================== 2) OPTIMIZER FABRİKASI ==================
def optimizer_kur(hp):
    """Kurs notundaki dört optimizer ailesi.
       adam    -> adaptif adım + momentum (1. ve 2. moment)
       adamax  -> Adam'ın sonsuz-norm varyantı, büyük gradyanlara daha dayanıklı
       rmsprop -> sadece adaptif adım (2. moment)
       sgd     -> sabit adım; momentum ayrıca aranıyor
    learning_rate optimizer'dan bağımsız seçilseydi anlamsız olurdu: SGD'nin
    iyi çalıştığı adım boyu Adam'ı bozar. Bu yüzden ikisi birlikte aranıyor."""
    ad = hp.Choice("optimizer", ["adam", "sgd", "rmsprop", "adamax"])
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="LOG")

    # Gradyan kırpma: patlayan gradyan frenlerinden biri (RNN'lerde klasik sorun)
    clip = hp.Choice("clipnorm", [0.0, 1.0])
    ek = {"clipnorm": clip} if clip > 0 else {}

    if ad == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, **ek)
    if ad == "adamax":
        return tf.keras.optimizers.Adamax(learning_rate=lr, **ek)
    if ad == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr, **ek)
    momentum = hp.Float("momentum", 0.0, 0.9, step=0.1)   # yalnızca sgd'de
    return tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, **ek)


# ================== 3) HYPERMODEL (build + fit) ==================
class TekrarliHyperModel(kt.HyperModel):
    """HyperModel sınıfı kullanmamızın sebebi: batch_size bir MODEL değil
    EĞİTİM parametresidir, build() içinde aranamaz. fit()'i override ederek
    onu da arama uzayına katıyoruz."""

    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(SEQ_LEN, 1)))

        num_layers = hp.Int("num_layers", 1, MAX_LAYERS)
        hucre_tipi = hp.Choice("hucre_tipi", ["LSTM", "GRU", "SimpleRNN"])
        cift_yonlu = hp.Boolean("cift_yonlu")
        batch_norm = hp.Boolean("batch_norm")

        # tanh SABIT (aranmiyor): relu, TensorFlow'un optimize edilmis tekrarli
        # katman yolunu kapatir ve egitimi belirgin sekilde yavaslatir; ayrica
        # tekrarli aglarda patlayan gradyan riskini artirir.
        AKTIVASYON = "tanh"

        for i in range(num_layers):
            son_katman = (i == num_layers - 1)

            katman = HUCRELER[hucre_tipi](
                units=hp.Int(f"units_{i}", 32, MAX_UNITS, step=32),
                activation=AKTIVASYON,
                # return_sequences KURALI: üstünde başka tekrarlı katman varsa
                # tüm zaman adımlarını ver (True); son katmansa sadece son adımı
                # ver (False), çünkü üstünde Dense var.
                return_sequences=not son_katman,
                kernel_regularizer=l2(hp.Float(f"l2_{i}", 1e-5, 1e-2, sampling="LOG")),
                kernel_initializer=hp.Choice(
                    f"init_{i}", ["glorot_uniform", "he_uniform", "random_normal"]),
                # recurrent_dropout 0'da sabit — hizlandirilmis cekirdegi
                # kapatmamak icin. Duzenlilestirme dropout + l2 ile saglaniyor.
                recurrent_dropout=0.0,
            )
            if cift_yonlu:
                # Bidirectional: diziyi ileri ve geri okuyup birleştirir.
                # Zaman serisinde tartışmalıdır ama sızıntı değildir: gelecek
                # bilgisi yalnızca pencere İÇİNDE kullanılır, pencere dışına
                # taşmaz. Tuner'ın karar vermesi için uzayda bırakıldı.
                katman = Bidirectional(katman)
            model.add(katman)

            if batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)))

        model.add(Dense(1))
        model.compile(optimizer=optimizer_kur(hp), loss="mse", metrics=["mae"])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [8, 16, 32, 64]),
            **kwargs,
        )


# ================== 4) TUNER ==================
PROJE_ADI = f"tekrarli_seq{SEQ_LEN}" + ("_hizli" if HIZLI else "")

ortak = dict(hypermodel=TekrarliHyperModel(), objective="val_loss",
             directory="Rossman/kt_dir", project_name=PROJE_ADI,
             overwrite=True, seed=42)

if HYPERBAND:
    tuner = kt.Hyperband(max_epochs=60, factor=3, **ortak)
else:
    tuner = kt.RandomSearch(max_trials=MAX_DENEME, executions_per_trial=1, **ortak)

early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

print("\nArama başlıyor...")
tuner.search(X_train, y_train, validation_split=0.1,
             epochs=100, callbacks=[early], verbose=0)

# ================== 5) EN İYİ MODEL ==================
best_hp = tuner.get_best_hyperparameters(1)[0]
print("\n===== EN İYİ HİPERPARAMETRELER =====")
for k, v in sorted(best_hp.values.items()):
    print(f"  {k}: {v}")

best_model = TekrarliHyperModel().build(best_hp)
best_model.summary()

history = TekrarliHyperModel().fit(
    best_hp, best_model, X_train, y_train,
    validation_split=0.1, epochs=150,
    callbacks=[EarlyStopping(monitor="val_loss", patience=12,
                             restore_best_weights=True)],
    verbose=0)

# ================== 6) DENEME ANALİZİ ==================
# "Adam kazandı" demek zayıf bir ifadedir — tek şanslı deneme de kazanabilir.
# Her hiperparametre için deneme sayısı, en iyi ve MEDYAN val_loss'u
# çıkarıyoruz. Medyanlar birbirine yakınsa "anlamlı fark yok" denebilir.
try:
    kayitlar = []
    for t in tuner.oracle.trials.values():
        if t.score is None:
            continue
        kayitlar.append({**t.hyperparameters.values, "val_loss": t.score})
    dn = pd.DataFrame(kayitlar)

    if not dn.empty:
        os.makedirs("Rossman/tahminler", exist_ok=True)
        dn.to_csv(f"Rossman/tahminler/_tuner_denemeler_seq{SEQ_LEN}.csv", index=False)

        print(f"\n===== DENEME ANALİZİ ({len(dn)} başarılı deneme) =====")
        incelenecek = ["hucre_tipi", "optimizer", "num_layers", "batch_size",
                       "cift_yonlu", "batch_norm", "clipnorm"]
        ozetler = []
        for kolon in incelenecek:
            if kolon not in dn.columns:
                continue
            o = (dn.groupby(kolon)["val_loss"]
                   .agg(deneme="count", en_iyi="min", medyan="median")
                   .sort_values("medyan"))
            print(f"\n-- {kolon} --")
            print(o.to_string())
            o = o.reset_index().rename(columns={kolon: "deger"})
            o.insert(0, "hiperparametre", kolon)
            ozetler.append(o)

        if ozetler:
            pd.concat(ozetler).to_csv(
                f"Rossman/tahminler/_tuner_ozet_seq{SEQ_LEN}.csv", index=False)

        # Sürekli parametreler için korelasyon: hangisi val_loss'u sürüklüyor?
        surekli = [c for c in ("learning_rate", "units_0", "dropout_0", "l2_0")
                   if c in dn.columns]
        if surekli:
            print("\n-- Sürekli parametrelerin val_loss ile korelasyonu --")
            print(dn[surekli + ["val_loss"]].corr()["val_loss"]
                  .drop("val_loss").round(3).to_string())

        # Grafik: hücre tipi ve optimizer dağılımları
        os.makedirs("Rossman/figs", exist_ok=True)
        fig, eksenler = plt.subplots(1, 2, figsize=(12, 4))
        for eksen, kolon in zip(eksenler, ["hucre_tipi", "optimizer"]):
            if kolon not in dn.columns:
                continue
            gruplar = sorted(dn[kolon].unique())
            for i, g in enumerate(gruplar):
                v = dn[dn[kolon] == g]["val_loss"]
                eksen.scatter([i] * len(v), v, alpha=0.65, s=45)
            eksen.set_xticks(range(len(gruplar)))
            eksen.set_xticklabels(gruplar, rotation=15)
            eksen.set_yscale("log")
            eksen.set_title(f"{kolon} — deneme dağılımı")
            eksen.set_ylabel("val_loss (log)")
            eksen.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"Rossman/figs/27_tuner_dagilim_seq{SEQ_LEN}.png")
        print(f"\nGrafik: figs/27_tuner_dagilim_seq{SEQ_LEN}.png")
except Exception as e:
    print("Deneme analizi atlandı:", e)

# ================== 7) TAHMİN + METRİK ==================
pred = scaler.inverse_transform(best_model.predict(X_test, verbose=0))
y_actual = scaler.inverse_transform(y_test)
sonuc = tum_metrikler(y_actual, pred, f"Tekrarlı ağ (Keras Tuner, seq={SEQ_LEN})")

# ================== 8) KAYITLAR ==================
test_dates = test_seri["Date"].iloc[SEQ_LEN:].values
os.makedirs("Rossman/tahminler", exist_ok=True)

# Ortak pencere kıyasına girecek dosya (yalnızca varsayılan SEQ_LEN=14 için;
# diğer pencereler farklı tarih aralığı üretir ve kıyası bozar)
if SEQ_LEN == 14 and not HIZLI:
    pd.DataFrame({"ds": test_dates, "gercek": y_actual.flatten(),
                  "tahmin": pred.flatten()}).to_csv(
        "Rossman/tahminler/lstm_tuned.csv", index=False)
    print("Kiyas icin kaydedildi: Rossman/tahminler/lstm_tuned.csv")
else:
    pd.DataFrame({"ds": test_dates, "gercek": y_actual.flatten(),
                  "tahmin": pred.flatten()}).to_csv(
        f"Rossman/tahminler/_tuned_seq{SEQ_LEN}.csv", index=False)
    print(f"Kaydedildi (kiyas disi): _tuned_seq{SEQ_LEN}.csv")

pd.DataFrame([best_hp.values]).to_csv(
    f"Rossman/tahminler/_tuner_params_seq{SEQ_LEN}.csv", index=False)

pd.DataFrame({"epoch": range(1, len(history.history["loss"]) + 1),
              "train_loss": history.history["loss"],
              "val_loss": history.history["val_loss"]}).to_csv(
    f"Rossman/tahminler/_tuned_loss_seq{SEQ_LEN}.csv", index=False)

# ================== 9) GRAFİKLER ==================
os.makedirs("Rossman/figs", exist_ok=True)

plt.figure(figsize=(11, 4))
plt.plot(test_dates, y_actual.flatten(), marker="o", label="Gerçek", color="black")
plt.plot(test_dates, pred.flatten(), marker="x",
         label=f"Tuned ({best_hp.values.get('hucre_tipi')})", color="darkviolet")
plt.title(f"Store 1 — Keras Tuner ile Optimize Model vs Gerçek (seq={SEQ_LEN})")
plt.ylabel("Satış (€)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"Rossman/figs/23_tuned_pred_seq{SEQ_LEN}.png")

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title(f"Tuned model — Eğitim/Doğrulama Kaybı (seq={SEQ_LEN})")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"Rossman/figs/24_tuned_loss_seq{SEQ_LEN}.png")

print(f"\nGrafikler: figs/23_tuned_pred_seq{SEQ_LEN}.png, "
      f"figs/24_tuned_loss_seq{SEQ_LEN}.png")