# Small web API: send an audio file, get the transcript and the facts back.
import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from asr_eval.extract import extract
from asr_eval.metrics import wer_cer
from asr_eval.transcribe import DEFAULT_MODEL, load_audio, load_model, transcribe

MODEL_NAME = DEFAULT_MODEL
model = None
app = FastAPI(title="asr-radio-eval")


def transcribe_bytes(data):
    # load the model the first time it is needed (tests replace this function)
    global model
    if model is None:
        model = load_model(MODEL_NAME)
    return transcribe(model, load_audio(io.BytesIO(data)))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/transcribe")
def transcribe_file(file: UploadFile = File(...), reference: str | None = Form(None)):
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio file")
    text = transcribe_bytes(data)
    result = {"text": text, "facts": extract(text)}
    if reference:
        result["wer"], result["cer"] = wer_cer(reference, text)
    return result
