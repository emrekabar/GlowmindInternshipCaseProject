from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
import pandas as pd
from schemas import ConfigRequest, PredictionInput, PredictionOutput, MetricsResponse
from ml_engine import MLEngine
import uvicorn

app = FastAPI(title="Finansal İşlem Sınıflandırma Servisi", version="1.0")

# ML Motorunu başlat (Singleton gibi davranacak)
ml_service = MLEngine()

@app.post("/upload", summary="Veri Seti Yükleme")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Sadece Excel dosyaları kabul edilir.")
    
    content = await file.read()
    try:
        ml_service.load_data(content)
        return {"message": "Dosya başarıyla yüklendi", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config", summary="Algoritma Seçimi")
def set_configuration(config: ConfigRequest):
    try:
        ml_service.set_config(config.algorithm)
        return {"message": f"Algoritma {config.algorithm} olarak güncellendi."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train", summary="Model Eğitimi", response_model=MetricsResponse)
def train_model():
    try:
        metrics = ml_service.train()
        return metrics
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) # Veri yüklenmemiş olabilir
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eğitim hatası: {str(e)}")

@app.post("/predict", summary="Tahmin Yapma", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Pydantic modelini DataFrame'e çevir (Model pipeline DataFrame bekliyor)
        df_input = pd.DataFrame([input_data.dict()])
        
        prediction = ml_service.predict(df_input)
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", summary="Model Performansı")
def get_metrics():
    if not ml_service.metrics:
        raise HTTPException(status_code=404, detail="Henüz model eğitilmedi.")
    return ml_service.metrics

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)