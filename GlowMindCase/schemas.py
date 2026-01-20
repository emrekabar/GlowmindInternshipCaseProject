from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Konfigürasyon Modeli
class ConfigRequest(BaseModel):
    algorithm: str  # "random_forest", "logistic_regression", "ann"

# Tekil Tahmin İçin Girdi Modeli
class PredictionInput(BaseModel):
    company_code: int
    document_number: int
    description: str
    payment_type: str
    amount: float
    currency_code: str
    transaction_type: str

# Tahmin Çıktı Modeli
class PredictionOutput(BaseModel):
    seller_number: str
    customer_number: str
    main_account: str

# Metrik Çıktı Modeli
class MetricsResponse(BaseModel):
    algorithm: str
    accuracy_scores: Dict[str, float]
    training_sample_count: int
    test_sample_count: int