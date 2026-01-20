import pandas as pd
import numpy as np
import io
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLEngine:
    def __init__(self):
        self.data = None
        self.model = None
        self.algorithm = "random_forest"
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Hedef kolonlar (Y)
        self.targets = ['seller_number', 'customer_number', 'main_account']
        
    def load_data(self, file_content: bytes):
        try:
            # Dosya xlsx mi csv mi kontrol et (basit try-except)
            try:
                self.data = pd.read_excel(io.BytesIO(file_content))
            except:
                self.data = pd.read_csv(io.BytesIO(file_content))
            
            logger.info(f"Veri yüklendi. Boyut: {self.data.shape}")
        except Exception as e:
            logger.error(f"Veri yükleme hatası: {str(e)}")
            raise ValueError("Dosya formatı desteklenmiyor veya bozuk.")

    def set_config(self, algorithm: str):
        valid_algos = ["random_forest", "logistic_regression", "ann"]
        if algorithm not in valid_algos:
            raise ValueError(f"Geçersiz algoritma. Seçenekler: {valid_algos}")
        self.algorithm = algorithm
        logger.info(f"Algoritma {algorithm} olarak ayarlandı.")

    def preprocess_data(self):
        if self.data is None:
            raise ValueError("Önce veri yüklemelisiniz.")

        df = self.data.copy()

        # --- DÜZELTME 1: Target Kolonlarını Temizle (.0 sorunu ve string yapma) ---
        for col in self.targets:
            # Önce sayısal yap, hataları NaN yap, NaN'ları 0 yap
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            try:
                # Float -> Int -> String yaparak "12345.0" -> "12345" dönüşümü yap
                df[col] = df[col].astype(int).astype(str)
            except:
                # Olmazsa direkt string yap
                df[col] = df[col].astype(str)

        # 2. X ve Y Ayrımı
        X = df[['company_code', 'document_number', 'description', 'payment_type', 'amount', 'currency_code', 'transaction_type']]
        y = df[self.targets]

        # 3. Veri Tipi Düzeltmeleri
        X['amount'] = pd.to_numeric(X['amount'], errors='coerce').fillna(0)
        X['description'] = X['description'].fillna("")

        # 4. Train / Test Split
        # stratify=None çünkü multi-label'da stratify zordur, shuffle=True yeterli
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=50, train_size=150, random_state=42, shuffle=True
        )

    def get_pipeline(self):
        numeric_features = ['amount']
        categorical_features = ['company_code', 'payment_type', 'currency_code', 'transaction_type']
        text_features = 'description'

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                # --- DÜZELTME 2: TF-IDF Özellik Sayısını Artır (Daha fazla kelime görsün) ---
                ('txt', TfidfVectorizer(max_features=500), text_features) 
            ])

        # Model Seçimi
        if self.algorithm == "random_forest":
            # --- DÜZELTME 3: class_weight='balanced' (Az geçen veriyi önemse) ---
            clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        elif self.algorithm == "logistic_regression":
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        elif self.algorithm == "ann":
            clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        else:
            clf = RandomForestClassifier(class_weight='balanced')

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MultiOutputClassifier(clf))
        ])
        
        return pipeline

    def train(self):
        self.preprocess_data()
        self.model = self.get_pipeline()
        logger.info("Eğitim başlıyor...")
        self.model.fit(self.X_train, self.y_train)
        logger.info("Eğitim tamamlandı.")

        y_pred = self.model.predict(self.X_test)
        
        accuracies = {}
        for i, col in enumerate(self.targets):
            acc = accuracy_score(self.y_test[col], y_pred[:, i])
            accuracies[col] = round(acc, 4)

        self.metrics = {
            "algorithm": self.algorithm,
            "accuracy_scores": accuracies,
            "training_sample_count": len(self.X_train),
            "test_sample_count": len(self.X_test)
        }
        return self.metrics

    def predict(self, inputs: pd.DataFrame):
        if self.model is None:
            # Model eğitilmemişse bile kural çalışsın diye geçici bypass yapılabilir
            # Ama doğrusu önce hatayı fırlatmaktır.
            raise ValueError("Model henüz eğitilmedi.")
        
        # Gelen veriyi işle
        inputs['amount'] = pd.to_numeric(inputs['amount'], errors='coerce').fillna(0)
        inputs['description'] = inputs['description'].fillna("")
        
        # --- MODEL TAHMİNİ ---
        # Önce model kendi bildiğini okusun
        preds = self.model.predict(inputs)
        
        seller_pred = preds[0][0]
        customer_pred = preds[0][1]
        account_pred = preds[0][2]

        # --- İŞ KURALLARI KATMANI (BUSINESS RULE LAYER) ---
        # Veri setinde olmayan veya modelin öğrenemediği özel durumlar için müdahale
        
        desc = inputs.iloc[0]['description'].upper()
        trans_type = inputs.iloc[0]['transaction_type']
        
        # KURAL 1: Reklam ve Hizmet Bedeli Ödemesi
        # Veride bu ilişki olmadığı için manuel kural ile düzeltiyoruz.
        if "REKLAM" in desc and "HIZMET" in desc and "NEFT" in trans_type:
            seller_pred = "3200200009"
            account_pred = "1200110387"  # Veride hiç olmayan hesap numarası
            # Customer number modelden ne geldiyse o kalsın veya "0" olsun
            customer_pred = "0"

        # Sonuçları hazırla
        result = {
            "seller_number": seller_pred,
            "customer_number": customer_pred,
            "main_account": account_pred
        }
        return result