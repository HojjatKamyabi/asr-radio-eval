from fastapi.testclient import TestClient

from asr_eval import api

TEXT = "Mayday. This is Silver Heron, Silver Heron. My position 54 degrees 12 minutes north, 007 degrees 45 minutes east. Fire. 12 persons on board. Over."
client = TestClient(api.app)


def fake_transcriber(monkeypatch):
    monkeypatch.setattr(api, "transcribe_bytes", lambda data: TEXT)


def test_health():
    assert client.get("/health").json() == {"status": "ok", "model": api.MODEL_NAME}


def test_transcribe_without_reference(monkeypatch):
    fake_transcriber(monkeypatch)
    body = client.post("/transcribe", files={"file": ("a.wav", b"fake-audio")}).json()
    assert body["text"] == TEXT
    assert body["facts"] == {"vessel": "silver heron", "position": [54, 12, 7, 45], "distress": "fire", "persons": 12}
    assert "wer" not in body and "cer" not in body


def test_transcribe_with_reference(monkeypatch):
    fake_transcriber(monkeypatch)
    response = client.post("/transcribe", files={"file": ("a.wav", b"fake-audio")}, data={"reference": TEXT})
    assert response.json()["wer"] == 0.0 and response.json()["cer"] == 0.0


def test_empty_file_is_rejected(monkeypatch):
    fake_transcriber(monkeypatch)
    assert client.post("/transcribe", files={"file": ("a.wav", b"")}).status_code == 400
