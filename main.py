import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import openai

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

class TextRequest(BaseModel):
    keyword: str
    text: str

def get_embedding(text: str, model="text-embedding-3-large"):
    resp = openai.embeddings.create(input=text, model=model)
    return resp.data[0].embedding

@app.post("/similarity")
def similarity(req: TextRequest):
    try:
        kw_vec = get_embedding(req.keyword)
        txt_vec = get_embedding(req.text)
        score = cosine_similarity([kw_vec], [txt_vec])[0][0]
        return {"score": round(score, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest")
def suggest(req: TextRequest):
    prompt = (
        f"Aşağıdaki metni '{req.keyword}' anahtar kelimesiyle daha uyumlu hâle getiriniz ve "
        "neden böyle önerdiğinizi kısa açıklayın:\n\n" +
        req.text
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        return {"suggestion": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
