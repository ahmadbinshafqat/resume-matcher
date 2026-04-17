# Resume / Job Match Scorer

A single-page web app that scores how well your profile matches a job description using **Groq** (LLaMA 3.3 70B). Paste a JD, pick your profile source, get an instant structured analysis.

## Features

- Two profile sources: **Upwork** or **Remote Platforms resume**
- Live progress bar and token counter while the AI generates
- Structured output: match score, matched/missing skills, strengths, recommendations, culture fit

## Tech Stack

- **Backend:** FastAPI + Python
- **AI:** Groq API (`llama-3.3-70b-versatile`)
- **Frontend:** Vanilla JS + SSE streaming

## Local Setup

**1. Clone and create a virtual environment**
```bash
git clone <your-repo-url>
cd resume-matcher
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment**

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

**4. Add your profiles**

In `.env`, paste your profile content as env var values:
```
UPWORK_PROFILE="Your full Upwork bio, skills, and work history..."
RESUME_PROFILE="Your full resume content..."
```

**5. Run**
```bash
python main.py
```

Open [http://localhost:8000](http://localhost:8000).

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Your Groq API key |
| `GROQ_MODEL` | No | Model to use (default: `llama-3.3-70b-versatile`) |
| `MAX_TOKENS` | No | Max tokens to generate (default: `1024`) |
| `UPWORK_PROFILE` | Yes* | Your Upwork profile content |
| `RESUME_PROFILE` | Yes* | Your resume content |

*At least one profile must be set.
