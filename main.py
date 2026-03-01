from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import json

app = FastAPI()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class CommentRequest(BaseModel):
    comment: str


class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Analyze the sentiment of the given comment and return a JSON object with:\n"
                        "- sentiment: exactly one of 'positive', 'negative', or 'neutral'\n"
                        "- rating: an integer from 1 to 5 (5=highly positive, 1=highly negative, 3=neutral)"
                    ),
                },
                {"role": "user", "content": request.comment},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"],
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                            },
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        result = json.loads(response.choices[0].message.content)
        return JSONResponse(content=result, media_type="application/json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
