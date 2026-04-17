import json
import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from groq import AsyncGroq
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("resume-matcher")

app = FastAPI(title="Resume/Job Match Scorer")
templates = Jinja2Templates(directory="templates")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))

groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# label → (env var, local filename)
PROFILES = {
    "upwork": "UPWORK_PROFILE",
    "resume": "RESUME_PROFILE",
}

SYSTEM_PROMPT = """You are an expert technical recruiter. Analyze profile-to-job fit.
Respond with valid JSON only — no markdown, no extra text. Use exactly this schema:
{
  "overall_score": <integer 0-100>,
  "score_label": <"Excellent Match"|"Strong Match"|"Good Match"|"Partial Match"|"Weak Match">,
  "summary": <2-3 sentence executive summary>,
  "matched_skills": [<skills candidate has that align with JD>],
  "missing_skills": [<skills required by JD but absent from profile>],
  "strengths": [<strong selling points relevant to this role>],
  "recommendations": [<actionable suggestions to improve fit>],
  "culture_fit_notes": <brief culture/soft-skill fit notes>
}"""


def load_profile(profile_type: str) -> str | None:
    return os.getenv(PROFILES[profile_type], "").strip() or None


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class AnalyzeRequest(BaseModel):
    profile_type: str
    job_description: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(payload: AnalyzeRequest):
    async def error_stream(msg: str):
        yield sse("error", {"error": msg})

    if payload.profile_type not in PROFILES:
        return StreamingResponse(error_stream("Invalid profile type."), media_type="text/event-stream")

    profile = load_profile(payload.profile_type)
    if not profile:
        return StreamingResponse(
            error_stream(f"Profile empty. Set {PROFILES[payload.profile_type]} in your environment variables."),
            media_type="text/event-stream",
        )

    job_description = payload.job_description.strip()
    if not job_description:
        return StreamingResponse(error_stream("Job description is required."), media_type="text/event-stream")

    user_message = (
        f"{payload.profile_type.upper()} PROFILE:\n{profile}\n\n"
        f"JOB DESCRIPTION:\n{job_description}"
    )

    async def event_stream():
        start, token_count, buffer = time.perf_counter(), 0, ""

        yield sse("status", {"stage": "connecting", "message": "Connecting to Groq..."})

        try:
            stream = await groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                response_format={"type": "json_object"},
                max_tokens=MAX_TOKENS,
                stream=True,
            )
            async for chunk in stream:
                fragment = chunk.choices[0].delta.content or ""
                if not fragment:
                    continue
                buffer += fragment
                token_count += 1
                if token_count % 5 == 0:
                    yield sse("progress", {
                        "elapsed": round(time.perf_counter() - start, 1),
                        "tokens":  token_count,
                        "stage":   "generating",
                    })
        except Exception as e:
            log.error("[groq] %s", e)
            yield sse("error", {"error": str(e)})
            return

        dur = time.perf_counter() - start
        log.info("done — %d tokens %.2fs (%.0f t/s)", token_count, dur, token_count / dur if dur else 0)

        try:
            yield sse("done", json.loads(buffer))
        except json.JSONDecodeError:
            yield sse("error", {"error": "Failed to parse result. Please try again."})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
