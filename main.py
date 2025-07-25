from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect
from google.cloud import translate_v2 as translate
import os
import spacy

nlp = spacy.load("en_core_web_sm")

#Google Cloud credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/MFERLK/Desktop/Ai Task Force/translation_api/gcloud-key.json"

from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")

app = FastAPI()
translate_client = translate.Client()

class TicketRequest(BaseModel):
    text: str
    target_language: str = "en"

@app.post("/translate")
def translate_text(request: TicketRequest):
    detected_lang = detect(request.text)
    if detected_lang == request.target_language:
        return {"message": "Text already in target language.", "text": request.text}

    translation = translate_client.translate(
        request.text,
        target_language=request.target_language
    )

    sentiment_result = sentiment_model(request.text)[0]
    sentiment_label = sentiment_result['label'].lower()  # e.g., "positive", "negative", "neutral"
    sentiment_score = round(sentiment_result['score'], 2)

    doc = nlp(request.text)
    entities = list(set([ent.text for ent in doc.ents]))

    return {
    "ticket_summary": {
        "language_detected": detected_lang,
        "translated_text": translation["translatedText"],
        "sentiment": sentiment_label,
        "sentiment_confidence": sentiment_score,
        "named_entities": entities
    },
    "metadata": {
        "source_text": request.text,
        "translation_model": "Google Cloud Translation",
        "sentiment_model": "Hugging Face Transformers",
        "entity_model": "spaCy en_core_web_sm"
    },
    "assyst_display_format": "standard_v1"
}

    




