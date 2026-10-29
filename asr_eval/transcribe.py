# Load a Whisper model and turn audio into text.
from faster_whisper import WhisperModel, decode_audio

DEFAULT_MODEL = "tiny.en"
ASR_SETTINGS = {
    "language": "en",
    "beam_size": 1,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "vad_filter": False,
}


def load_model(name=DEFAULT_MODEL):
    return WhisperModel(name, device="cpu", compute_type="int8")


def load_audio(path_or_file):
    # 16 kHz mono float32
    return decode_audio(path_or_file, sampling_rate=16000)


def transcribe(model, audio):
    segments, _ = model.transcribe(audio, **ASR_SETTINGS)
    return " ".join(s.text.strip() for s in segments).strip()
