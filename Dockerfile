FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

### 📌 STEP 7 — Create `requirements.txt`
```
fastapi==0.110.0
uvicorn==0.27.0
mlflow==2.11.0
dagshub==0.3.8
sentence-transformers==2.6.1
scikit-learn==1.4.0
joblib==1.3.2
pandas==2.2.0
numpy==1.26.4
prometheus-client==0.20.0
pydantic==2.6.0