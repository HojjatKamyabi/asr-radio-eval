FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY asr_eval/ asr_eval/
EXPOSE 8000
CMD ["uvicorn", "asr_eval.api:app", "--host", "0.0.0.0", "--port", "8000"]
