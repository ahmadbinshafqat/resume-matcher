import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from groq import AsyncGroq
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("resume-matcher")

app = FastAPI(title="JobFit AI — Job Board & Match Scorer")
templates = Jinja2Templates(directory="templates")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
MAX_TOKENS_INTERVIEW = int(os.getenv("MAX_TOKENS_INTERVIEW", 2048))

groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

PROFILES = {
    "upwork": "UPWORK_PROFILE",
    "resume": "RESUME_PROFILE",
}

JOBS_FILE = Path(__file__).parent / "jobs.json"

SYSTEM_PROMPT = """You are an expert technical recruiter. Analyze profile-to-job fit.
Respond with valid JSON only — no markdown, no extra text. Use exactly this schema:
{
  "overall_score": <integer 0-100>,
  "score_label": <"Excellent Match"|"Strong Match"|"Good Match"|"Partial Match"|"Weak Match">,
  "dimension_scores": {"technical": <0-100>, "experience": <0-100>, "leadership": <0-100>},
  "summary": <2-3 sentence executive summary>,
  "matched_skills": [<skills candidate has that align with JD>],
  "missing_skills": [<skills required by JD but absent from profile>],
  "strengths": [<strong selling points relevant to this role>],
  "recommendations": [<actionable suggestions to improve fit>],
  "application_tips": [<2-3 specific tips for tailoring this application or cover letter>],
  "culture_fit_notes": <brief culture/soft-skill fit notes>
}"""

INTERVIEW_PROMPT = """You are a senior career coach and interview strategist.
Given a candidate profile and job description, create a comprehensive interview preparation guide.
Respond with valid JSON only — no markdown, no extra text. Use exactly this schema:
{
  "elevator_pitch": <2-3 sentence personalized pitch tailored to this specific role>,
  "likely_questions": [<8-10 interview questions specific to this role>],
  "talking_points": [<6-8 key accomplishments from the profile that map directly to this JD>],
  "technical_topics": [<5-7 technical areas to review or brush up on before the interview>],
  "questions_to_ask": [<5 smart, thoughtful questions the candidate should ask the interviewer>],
  "red_flags_to_address": [<2-4 potential concerns the interviewer might raise and how to proactively address them>]
}"""


def load_profile(profile_type: str) -> str | None:
    return os.getenv(PROFILES[profile_type], "").strip() or None


def load_jobs() -> list:
    try:
        return json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load jobs.json: %s", exc)
        return []


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class AnalyzeRequest(BaseModel):
    profile_type: str
    job_description: str


def _build_user_message(profile_type: str, job_description: str, profile: str) -> str:
    return (
        f"{profile_type.upper()} PROFILE:\n{profile}\n\n"
        f"JOB DESCRIPTION:\n{job_description}"
    )


async def _stream_groq(system: str, user_message: str, max_tokens: int):
    """Async generator that yields SSE strings from a Groq stream."""
    start, token_count, buffer = time.perf_counter(), 0, ""

    yield sse("status", {"stage": "connecting", "message": "Connecting to AI..."})

    try:
        stream = await groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
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
        yield sse("error", {"error": "Failed to parse AI response. Please try again."})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/jobs")
async def get_jobs():
    jobs = load_jobs()
    return Response(
        content=json.dumps(jobs),
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=300, stale-while-revalidate=60"},
    )


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

    if not payload.job_description.strip():
        return StreamingResponse(error_stream("Job description is required."), media_type="text/event-stream")

    user_message = _build_user_message(
        payload.profile_type, payload.job_description.strip(), profile
    )

    return StreamingResponse(
        _stream_groq(SYSTEM_PROMPT, user_message, MAX_TOKENS),
        media_type="text/event-stream",
    )


@app.post("/prepare")
async def prepare(payload: AnalyzeRequest):
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

    if not payload.job_description.strip():
        return StreamingResponse(error_stream("Job description is required."), media_type="text/event-stream")

    user_message = _build_user_message(
        payload.profile_type, payload.job_description.strip(), profile
    )

    return StreamingResponse(
        _stream_groq(INTERVIEW_PROMPT, user_message, MAX_TOKENS_INTERVIEW),
        media_type="text/event-stream",
    )


@app.get("/robots.txt")
async def robots_txt():
    return Response(
        content="User-agent: *\nAllow: /\n",
        media_type="text/plain",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/sitemap.xml")
async def sitemap_xml(request: Request):
    base = str(request.base_url).rstrip("/")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{base}/</loc><changefreq>daily</changefreq><priority>1.0</priority></url>\n"
        "</urlset>"
    )
    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
