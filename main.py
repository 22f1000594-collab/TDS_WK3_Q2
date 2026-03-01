from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
import os
import json

app = FastAPI()


class CommentRequest(BaseModel):
    comment: str


@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Analyze the sentiment of the given comment and respond ONLY with a valid JSON object. "
                        "No explanation, no markdown, no code blocks — just raw JSON. "
                        "The JSON must have exactly two fields: "
                        "'sentiment' (one of 'positive', 'negative', 'neutral') and "
                        "'rating' (integer 1-5, where 1=very negative, 3=neutral, 5=very positive)."
                    ),
                },
                {"role": "user", "content": request.comment},
            ],
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content.strip()
        result = json.loads(text)

        if result.get("sentiment") not in ("positive", "negative", "neutral"):
            raise ValueError("Invalid sentiment value")
        if not isinstance(result.get("rating"), int) or not (1 <= result["rating"] <= 5):
            raise ValueError("Invalid rating value")

        return JSONResponse(content={"sentiment": result["sentiment"], "rating": result["rating"]})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
