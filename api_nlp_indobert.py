from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import re

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from youtube_comment_downloader import YoutubeCommentDownloader
from urllib.parse import urlparse, parse_qs
# APP
app = FastAPI(
    title="IndoBERT Sentiment API"
)

class TextRequest(BaseModel):
    text: str

class YoutubeRequest(BaseModel):
    url: str

def normalize_youtube_url(url):

    # URL pendek
    if "youtu.be" in url:

        video_id = url.split("/")[-1].split("?")[0]

        return f"https://www.youtube.com/watch?v={video_id}"

    # URL normal
    if "youtube.com" in url:

        parsed = urlparse(url)

        query = parse_qs(parsed.query)

        if "v" in query:
            return f"https://www.youtube.com/watch?v={query['v'][0]}"

    return url


def get_youtube_comments(url):

    url = normalize_youtube_url(url)

    downloader = YoutubeCommentDownloader()

    comments = []

    for item in downloader.get_comments_from_url(url):

        text = item.get("text", "").strip()

        if text:
            comments.append(text)

    return comments
# LABEL
id2label = {
    0: "negatif",
    1: "netral",
    2: "positif"
}

# LOAD MODEL
print("Loading IndoBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    "./sentiment_model"
)

print("id2label:", model.config.id2label)
print("label2id:", model.config.label2id)

tokenizer = AutoTokenizer.from_pretrained(
    "./sentiment_model"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "./sentiment_model"
)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

print("IndoBERT Loaded")

# LOAD KEYBERT
print("Loading KeyBERT...")

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

kw_model = KeyBERT(
    embedding_model
)

print("KeyBERT Loaded")

# PREPROCESSING
def preprocess_bert(text):

    text = str(text).lower()

    text = re.sub(
        r'http\S+|www\S+',
        '',
        text
    )

    text = re.sub(
        r'<.*?>',
        '',
        text
    )

    text = re.sub(
        r'@\w+',
        '',
        text
    )

    text = re.sub(
        r'(.)\1+', 
        r'\1\1',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text

# SENTIMENT
def predict_sentiment(text):

    text = preprocess_bert(text)

    result = classifier(
        text,
        truncation=True,
        max_length=128
    )[0]

    print(result)  # DEBUG

    label_map = {
        "LABEL_0": "negatif",
        "LABEL_1": "netral",
        "LABEL_2": "positif"
    }

    return {
        "label": label_map.get(
            result["label"],
            result["label"]
        ),
        "score": round(
            result["score"],
            4
        )
    }


# TOPIC EXTRACTION
def extract_topics(comments):

    cleaned = []

    for c in comments:

        c = str(c)

        c = re.sub(
            r'http\S+|www\S+',
            '',
            c
        )

        c = re.sub(
            r'[^a-zA-Z0-9\s]',
            ' ',
            c
        )

        c = re.sub(
            r'\s+',
            ' ',
            c
        ).strip()

        if len(c.split()) >= 3:
            cleaned.append(c)

    all_text = " ".join(cleaned)

    keywords = kw_model.extract_keywords(
        all_text,
        keyphrase_ngram_range=(1, 3),
        stop_words=None,
        top_n=10
    )

    return [
        kw for kw, score in keywords
    ]

# INSIGHT GENERATOR
def generate_insight(df, topics, results):

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .to_dict()
    )

    positif = sentiment_counts.get(
        "positif",
        0
    )

    netral = sentiment_counts.get(
        "netral",
        0
    )

    negatif = sentiment_counts.get(
        "negatif",
        0
    )

    total = len(df)

    sentiment_dominan = max(
        sentiment_counts,
        key=sentiment_counts.get
    )

    if sentiment_dominan == "positif":

        opini = (
            "Mayoritas pengguna memberikan respon positif terhadap isi video."
        )

    elif sentiment_dominan == "negatif":

        opini = (
            "Mayoritas pengguna memberikan kritik atau respon negatif terhadap isi video."
        )

    else:

        opini = (
            "Mayoritas komentar bersifat netral dan lebih banyak berisi diskusi."
        )

    summary = f"""
Mayoritas komentar membahas tentang:
{', '.join(topics[:5])}

{opini}

Distribusi sentimen:

Positif : {(positif/total)*100:.2f}%
Netral  : {(netral/total)*100:.2f}%
Negatif : {(negatif/total)*100:.2f}%
"""

    return {
       "positif": round((positif/total)*100, 2),
       "netral": round((netral/total)*100, 2),
       "negatif": round((negatif/total)*100, 2),
       "topik": topics[:5],
       "kesimpulan": summary.strip(),
       "results": results
    }

# REQUEST MODEL
class PredictRequest(BaseModel):
    text: str


class AnalysisRequest(BaseModel):
    comments: list[str]


# ROOT
@app.get("/")
def root():

    return {
        "message": "IndoBERT Sentiment API Running"
    }

# PREDICT 1 COMMENT
@app.post("/predict")
def predict(data: PredictRequest):

    return predict_sentiment(
        data.text
    )

# FULL ANALYSIS
@app.post("/Analisis-Youtube")
def analisis_youtube(data: YoutubeRequest):

    comments = get_youtube_comments(
        data.url
    )

    if len(comments) == 0:

        return {
            "error": "Komentar tidak ditemukan"
        }

    print(f"Jumlah komentar: {len(comments)}")

    print("=== SAMPLE COMMENTS ===")

    for i, c in enumerate(comments[:10]):
        print(i, c)

    results = []

    for text in comments:

        pred = predict_sentiment(text)

        results.append({
            "text": text,
            "sentiment": pred["label"],
            "score": pred["score"]
        })

    df = pd.DataFrame(results)

    topics = extract_topics(comments)

    insight = generate_insight(
        df,
        topics,
        results
    )

    insight["jumlah_komentar"] = len(comments)

    return insight


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)