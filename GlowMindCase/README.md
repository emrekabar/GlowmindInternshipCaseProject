Markdown

# 🚀 Finansal İşlem Sınıflandırma API

Bu proje, finansal işlem kayıtlarını (açıklama, tutar vb.) analiz ederek işlemin ait olduğu **Satıcı**, **Müşteri** ve **Ana Hesap** bilgilerini makine öğrenmesi ile tahmin eden FastAPI tabanlı bir servistir.

## 🛠️ Özellikler

- **Çoklu Çıktı (Multi-Output):** Tek seferde 3 farklı hedef değişkeni tahmin eder.
- **NLP Desteği:** İşlem açıklamaları TF-IDF ile vektörleştirilir.
- **Algoritma Seçimi:** Random Forest, Logistic Regression veya ANN seçilebilir.
- **API First:** Tüm süreç (Eğitim, Tahmin) REST API üzerinden yönetilir.

## 📦 Kurulum

1. Gereksinimleri yükleyin:

   ```bash
   pip install -r requirements.txt
   ```

   Uygulamayı başlatın:

   ```
   bash
   python main.py
    Dokümantasyon ve Test: Tarayıcınızda http://localhost:8000/docs adresine gidin.
   ```

🚀 Kullanım Adımları
Upload: /upload endpoint'i ile data.xlsx dosyasını yükleyin.

Config: /config ile algoritmayı seçin (Örn: random_forest).

Train: /train ile modeli eğitin. Başarı metriklerini inceleyin.

Predict: /predict ile yeni veriler için tahmin alın.

🏗️ Mimari
Backend: FastAPI

ML Engine: Scikit-Learn (Pipeline, MultiOutputClassifier)

Veri İşleme: Pandas
