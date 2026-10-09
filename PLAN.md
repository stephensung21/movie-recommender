# Movie Recommender — Project Plan

An ensemble movie recommendation system that combines **collaborative filtering**, **content-based filtering**, and **LLM-based recommendation**, exposed through a **conversational chatbot** and eventually a **Letterboxd-style web app**.

---

## 1. Goals

| # | Goal | Success looks like |
|---|------|--------------------|
| 1 | Recommend movies from a user's ratings/history | Beats a popularity baseline on offline metrics (NDCG@10, Recall@20) |
| 2 | Recommend movies similar to a given movie | "More like *Arrival*" returns sensible, diverse results |
| 3 | Conversational recommendations | User types "slow-burn sci-fi like Arrival, no horror, something with Amy Adams" and gets grounded, explained picks |
| 4 | Letterboxd-like product | Log films, rate (½-star), review, watchlist, lists, profile, and a personalized "For You" feed |

**Non-goals (for v1):** streaming availability, social graph recommendations, mobile apps.

---

## 2. High-Level Architecture

```
                         ┌──────────────────────────────┐
                         │      Web UI (Next.js)        │
                         │  diary · ratings · lists ·   │
                         │  film pages · chat panel     │
                         └──────────────┬───────────────┘
                                        │ REST / SSE
                         ┌──────────────▼───────────────┐
                         │      API (FastAPI)           │
                         │  /recommend  /similar  /chat │
                         └───┬──────────┬───────────┬───┘
                             │          │           │
              ┌──────────────▼──┐  ┌────▼──────┐ ┌──▼─────────────────┐
              │ Ensemble Engine │  │ Chat Agent│ │ App data (users,   │
              │  (candidate gen │◄─┤ (LLM +    │ │ ratings, reviews,  │
              │   + ranker)     │  │  tools)   │ │ lists) — Postgres  │
              └─┬──────┬──────┬─┘  └───────────┘ └────────────────────┘
                │      │      │
        ┌───────▼┐ ┌───▼────┐ ┌▼────────┐
        │  CF    │ │Content │ │  LLM    │
        │ model  │ │ model  │ │ recomm. │
        └───┬────┘ └───┬────┘ └────┬────┘
            │          │           │
        ┌───▼──────────▼───────────▼────┐
        │ Feature store / artifacts      │
        │ embeddings (pgvector / FAISS), │
        │ trained models, movie metadata │
        └────────────────────────────────┘
```

**Core idea:** each of the three recommenders acts as a **candidate generator**. Their candidates are merged and passed to a **ranker** that produces the final list. The chatbot is an LLM agent that calls the same engine as a tool.

---

## 3. Data

### 3.1 Sources

| Source | What it gives us | Notes |
|--------|------------------|-------|
| [MovieLens 32M](https://grouplens.org/datasets/movielens/) (start with `ml-latest-small` for dev) | User–movie ratings, tags, timestamps | Training data for CF. Includes `links.csv` mapping to TMDB/IMDb IDs |
| [TMDB API](https://developer.themoviedb.org/) | Overviews, genres, cast, crew, keywords, posters, backdrops, release dates, runtime, language | Content features + UI images. Requires attribution in the UI |
| (Optional) IMDb non-commercial datasets | Extra cast/crew, ratings counts | Personal/non-commercial use only |
| Our own app | Real users' ratings, reviews, watchlists | Feeds back into CF once the app has users |

### 3.2 Pipeline (`data/`)

1. **Ingest** MovieLens CSVs → Parquet.
2. **Enrich** each movie via TMDB using `links.csv` (`tmdbId`); cache raw JSON locally to respect rate limits.
3. **Clean**: drop movies with no TMDB match, dedupe, normalize names (cast, directors, keywords).
4. **Build tables**: `movies`, `people`, `movie_people`, `genres`, `keywords`, `ratings`.
5. **Split** ratings by **time** (per-user leave-last-N-out or global time cutoff) → `train / val / test`. Never random-split; it leaks the future.
6. Version datasets (DVC or dated Parquet snapshots) so experiments are reproducible.

---

## 4. The Three Recommenders

### 4.1 Collaborative Filtering (CF)

Learns from *who rated what*. Strong for users with history; useless for brand-new users and brand-new movies (cold start).

| Stage | Model | Library |
|-------|-------|---------|
| Baseline | Popularity (global & per-genre), item-mean | pandas |
| v1 | Item–item kNN (cosine on rating vectors) | `scikit-learn` / `implicit` |
| v1 | Matrix factorization — SVD (explicit ratings) and ALS/BPR (implicit "watched/liked") | `surprise`, `implicit` |
| v2 | LightFM (hybrid MF with item features — helps cold start) | `lightfm` |
| v3 (stretch) | Two-tower neural model or sequential model (SASRec) | PyTorch |

**Outputs:** user embeddings, item embeddings, `score_cf(user, movie)`, and `similar_cf(movie)` (items co-liked by the same users).

### 4.2 Content-Based Filtering (CB)

Learns from *what a movie is*. Works for any movie with metadata, including new releases, and powers "more like this."

**Feature sets per movie:**
- **Structured:** genres, director(s), top-billed cast, writers, keywords, decade, runtime bucket, original language, certification.
- **Text:** overview + tagline (+ optionally aggregated user tags/reviews).

**Representations:**
1. **Sparse:** TF-IDF / one-hot over genres, people, keywords (weighted — director and keywords usually matter more than 8th-billed actor).
2. **Dense:** sentence embeddings of a "movie document" (title + overview + genres + keywords + director + cast) using an embedding model (e.g. `bge-small` / `all-MiniLM` via `sentence-transformers`, or a hosted embedding API).
3. **Combined** vector = weighted concat, stored in a vector index (**pgvector** in Postgres, or FAISS for offline experiments).

**User profile** = rating-weighted average of the embeddings of movies they liked (minus disliked), optionally time-decayed.

**Outputs:** `score_cb(user, movie)` = cosine(user_profile, movie_vec); `similar_cb(movie)` = nearest neighbors.

### 4.3 LLM-Based Recommendation

Uses a large language model's world knowledge of films (themes, tone, "vibe," critical reception) — things that are hard to get from ratings or metadata alone.

**Roles the LLM plays:**
1. **Query understanding** — turn free text into structured intent:
   ```json
   {"seed_movies": ["Arrival"], "people": ["Amy Adams"], "genres_include": ["Science Fiction"],
    "genres_exclude": ["Horror"], "mood": ["slow-burn", "cerebral"], "year_range": null}
   ```
2. **Candidate generation** — "Given this user's top-rated films and request, suggest 20 films." Gives serendipitous, thematically related picks.
3. **Re-ranking & explanation** — given the top ~30 candidates from the ensemble *with their metadata*, pick and order the best 10 and write a one-line "why you'll like it" for each.

**Grounding (critical):** LLMs hallucinate titles and facts.
- Every LLM-suggested title is resolved to a catalog ID via exact match → fuzzy match (title + year) → vector search fallback. Unresolvable titles are dropped and logged (track a **hallucination rate** metric).
- Use **retrieval-augmented generation**: the re-ranker only sees candidates and facts pulled from our database, never free recall.
- Request structured (JSON / tool-call) output, not prose to parse.

**Model choice:** use a fast, cheap model (e.g. Claude Haiku) for query parsing and a stronger model (e.g. Claude Sonnet) for the chat agent and re-ranking. Cache responses keyed on (prompt, user-profile hash) to control cost and latency.

---

## 5. The Ensemble

### 5.1 Two-stage design: candidate generation → ranking

```
User / query
   │
   ├─► CF top-200 ────────┐
   ├─► Content top-200 ───┼─► union + dedupe (~300–500)
   ├─► LLM top-20 ────────┤        │
   └─► Popular/trending ──┘        ▼
                          Filters (already seen, user exclusions,
                          hard constraints from chat)
                                   │
                                   ▼
                          Ranker (weighted blend → LTR model)
                                   │
                                   ▼
                          Diversity re-rank (MMR) → top-K
                                   │
                                   ▼
                          (optional) LLM rerank + explanations
```

### 5.2 Ranker — built in increments

1. **Weighted hybrid (v1):**
   `score = w_cf·z(cf) + w_cb·z(cb) + w_llm·llm_flag + w_pop·z(pop)` with z-normalized scores. Tune weights on the validation set (grid / Optuna).
2. **Adaptive weights:** weights depend on the user's history length — cold-start users lean on content + LLM + popularity; heavy users lean on CF.
3. **Learning-to-rank (v2):** LightGBM `LambdaRank` with features:
   - each model's score and rank, which generators produced the candidate
   - movie features (popularity, year, avg rating, runtime, genre overlap with user)
   - user features (# ratings, avg rating, genre entropy)
   - user–movie features (director/actor previously rated by user, embedding similarity)
4. **Diversity:** Maximal Marginal Relevance to avoid ten near-identical sequels; optionally cap per-franchise / per-director.

### 5.3 Cold start strategy

| Situation | Strategy |
|-----------|----------|
| New user, 0 ratings | Onboarding: pick 10+ films you love from a poster grid → content + LLM + popularity |
| New user, few ratings | Blend shifts gradually from CB/LLM to CF |
| New movie, no ratings | Content embeddings + LightFM item features; LLM knows about it if pre-cutoff |

---

## 6. Chatbot

A conversational agent where the user can describe a movie, actor, genre, mood, or seed film and get recommendations.

### 6.1 Design: LLM agent with tools

The LLM does the talking and reasoning; our system supplies the facts and the rankings.

| Tool | Purpose |
|------|---------|
| `search_movies(query, filters)` | Title / keyword / vector search in the catalog |
| `get_movie(movie_id)` | Full metadata (cast, crew, overview, ratings) |
| `search_people(name)` | Resolve actors/directors → filmography |
| `similar_movies(movie_ids, k)` | Ensemble "more like these" |
| `recommend_for_user(user_id, constraints, k)` | Full personalized ensemble with filters |
| `get_user_history(user_id)` | What they've watched/rated (to avoid repeats and personalize) |
| `add_to_watchlist(movie_id)` | Action from chat (UI phase) |

### 6.2 Flow

1. User: *"Something like Parasite but less violent, maybe from the last 5 years."*
2. Agent calls `search_movies("Parasite")` → resolves ID.
3. Agent calls `similar_movies([parasite_id], constraints={year_min: 2021, exclude_tags: ["graphic violence"]})`.
4. Agent reads results, picks 5, explains each in one sentence, asks a follow-up ("Want more Korean cinema or more class-satire in general?").
5. Conversation state (seeds, constraints, rejected titles) is kept so "not that one, something funnier" works.

### 6.3 Guardrails

- Only recommend movies returned by tools (system prompt + post-check that every title in the answer has a catalog ID).
- Keep the agent on-topic (movies/TV); politely decline unrelated requests.
- Stream responses (SSE) to the UI.

### 6.4 Interfaces over time

1. CLI (`python -m recommender.chat`) for development
2. Streamlit / Gradio prototype for quick demos
3. Chat panel embedded in the web app

---

## 7. Evaluation

### 7.1 Offline metrics (time-split test set)

- **Accuracy:** Recall@K, Precision@K, NDCG@K, MAP@K, Hit Rate@K (K = 10, 20)
- **Rating prediction (CF explicit):** RMSE / MAE (secondary — ranking matters more)
- **Beyond accuracy:** catalog coverage, intra-list diversity, novelty (mean popularity rank), serendipity
- **Cold-start slices:** report metrics separately for users with <5, 5–20, >20 ratings

### 7.2 Ablations

Report every metric for: popularity, CF only, CB only, LLM only, CF+CB, CF+CB+LLM (weighted), CF+CB+LLM (LTR). This table is the main result of the project.

### 7.3 LLM-specific

- Hallucination rate (titles not in catalog), constraint violation rate (e.g. recommended horror when excluded)
- Cost and latency per request
- Small hand-labeled set of chat prompts (~50) judged by a human or an LLM-as-judge rubric

### 7.4 Online (once the app has users)

Click-through on recommendations, add-to-watchlist rate, rating of recommended films, thumbs up/down in chat. A/B test ranker versions.

Track all experiments with **MLflow** (params, metrics, artifacts).

---

## 8. Letterboxd-Style Web App

### 8.1 Features

**MVP**
- Auth (email + OAuth)
- Film page: poster, backdrop, synopsis, cast & crew, avg rating, rating histogram, "similar films"
- Log a film: watched date, ½-star rating (0.5–5), like ♥, review text, rewatch flag
- Diary (chronological log) and profile page (favorites, stats)
- Watchlist
- Search (films, people)
- **"For You"** personalized feed from the ensemble
- Onboarding: pick films you love to solve cold start
- **Chat panel** ("Ask for a recommendation")

**Later**
- Lists (ranked/unranked, public/private)
- Follow users, activity feed, likes/comments on reviews
- Import from Letterboxd CSV export (lets users bring their history → instant CF signal)
- Year-in-review stats page
- Explanations on recs ("Because you liked *Arrival* and *Sicario*")

### 8.2 Design notes

- Dark theme, poster-grid-centric layout, hover states with quick rate/like/watchlist actions.
- Keep the visual style *inspired by* Letterboxd but with our own branding and name.
- TMDB attribution in footer (required by TMDB terms).

### 8.3 Data model (Postgres)

```
users(id, username, email, created_at, ...)
movies(id, tmdb_id, ml_id, title, year, overview, runtime, poster_path, ...)
people(id, tmdb_id, name, ...)          movie_people(movie_id, person_id, role, order)
genres / movie_genres / keywords / movie_keywords
logs(id, user_id, movie_id, watched_on, rating, liked, review, rewatch)
watchlist(user_id, movie_id, added_at)
lists(id, user_id, title, description, ranked, public)   list_items(list_id, movie_id, position, note)
follows(follower_id, followee_id)
movie_embeddings(movie_id, model_name, vector)            -- pgvector
recommendations_cache(user_id, generated_at, payload)
chat_sessions(id, user_id, ...)   chat_messages(session_id, role, content, created_at)
rec_feedback(user_id, movie_id, source, signal, created_at)
```

---

## 9. Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language (ML/backend) | Python 3.11+ | ML ecosystem |
| Data | pandas / polars, Parquet | Fast local processing |
| ML | scikit-learn, implicit, surprise, lightfm, LightGBM, sentence-transformers, PyTorch (stretch) | Standard recsys tools |
| Vector search | pgvector (prod), FAISS (experiments) | One database to run |
| LLM | Anthropic Claude API (tool use, structured output) | Agent + rerank + parsing |
| Experiment tracking | MLflow | Compare ablations |
| API | FastAPI + Pydantic | Async, typed, OpenAPI docs |
| DB / cache | PostgreSQL, Redis | App data + rec cache |
| Frontend | Next.js (React, TypeScript), Tailwind CSS, shadcn/ui | Fast to build a polished UI |
| Auth | NextAuth / Auth.js (or Clerk) | |
| Jobs | Cron / Celery / Prefect for nightly retrain & embedding refresh | |
| Infra | Docker Compose locally; deploy to Fly.io / Render / Railway (API + DB) and Vercel (frontend) | Cheap, simple |
| Quality | pytest, ruff, mypy, pre-commit, GitHub Actions CI | |

---

## 10. Proposed Repository Layout

```
movie-recommender/
├── PLAN.md
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── .env.example                # TMDB_API_KEY, ANTHROPIC_API_KEY, DATABASE_URL
├── data/                       # gitignored raw/processed data
├── notebooks/                  # EDA, experiments
├── src/recommender/
│   ├── data/                   # ingest, enrich (TMDB), clean, split
│   ├── models/
│   │   ├── baselines.py
│   │   ├── collaborative.py    # kNN, SVD, ALS, LightFM
│   │   ├── content.py          # TF-IDF + embeddings, user profiles
│   │   └── llm.py              # query parsing, LLM candidates, rerank, grounding
│   ├── ensemble/
│   │   ├── candidates.py
│   │   ├── features.py
│   │   ├── ranker.py           # weighted blend, LightGBM LTR
│   │   └── diversity.py        # MMR
│   ├── chat/
│   │   ├── agent.py
│   │   ├── tools.py
│   │   └── prompts.py
│   ├── evaluation/             # metrics, splits, ablation runner
│   └── api/                    # FastAPI app, routers, schemas
├── web/                        # Next.js frontend
├── scripts/                    # train.py, evaluate.py, build_index.py
└── tests/
```

---

## 11. Roadmap

Each phase ends with something runnable.

### Phase 0 — Setup (½ week)
- [ ] Repo scaffold, `pyproject.toml`, ruff/pytest/pre-commit, CI
- [ ] Get TMDB and Anthropic API keys; `.env.example`
- [ ] Docker Compose with Postgres + pgvector

### Phase 1 — Data (1 week)
- [ ] Download MovieLens (small for dev, 32M later)
- [ ] TMDB enrichment with local cache
- [ ] Clean tables + time-based train/val/test split
- [ ] EDA notebook (rating distribution, sparsity, long tail)

### Phase 2 — Baselines & evaluation harness (1 week)
- [ ] Metrics module (Recall/NDCG/MAP/coverage/diversity/novelty)
- [ ] Popularity baseline, ablation runner, MLflow logging

### Phase 3 — Collaborative filtering (1–2 weeks)
- [ ] Item–item kNN, SVD, ALS/BPR; hyperparameter tuning
- [ ] `similar_cf` and `score_cf` interfaces

### Phase 4 — Content-based (1 week)
- [ ] TF-IDF + dense embeddings, vector index
- [ ] User profile vectors, `similar_cb`, `score_cb`

### Phase 5 — LLM recommender (1–2 weeks)
- [ ] Query parser with structured output
- [ ] LLM candidate generator + title grounding (fuzzy + vector)
- [ ] LLM reranker with explanations; response caching
- [ ] Hallucination & constraint-violation metrics

### Phase 6 — Ensemble (1–2 weeks)
- [ ] Candidate merging + filtering
- [ ] Weighted hybrid with tuned / adaptive weights
- [ ] LightGBM LambdaRank ranker, MMR diversity
- [ ] **Full ablation table** — the headline result

### Phase 7 — Chatbot (1–2 weeks)
- [ ] Tool definitions over the ensemble
- [ ] Agent loop with conversation state and guardrails
- [ ] CLI, then Streamlit/Gradio demo
- [ ] Chat eval set (~50 prompts)

### Phase 8 — API (1 week)
- [ ] FastAPI: `/movies`, `/search`, `/recommend`, `/similar`, `/chat` (SSE), `/logs`, `/watchlist`
- [ ] Recommendation caching in Redis; nightly batch precompute

### Phase 9 — Web app MVP (3–4 weeks)
- [ ] Next.js scaffold, auth, design system (dark theme)
- [ ] Film page, search, log/rate/review, diary, watchlist, profile
- [ ] Onboarding flow, "For You" feed, chat panel
- [ ] Letterboxd CSV import

### Phase 10 — Production & feedback loop (ongoing)
- [ ] Deploy (Docker → Fly/Render + Vercel)
- [ ] Log rec impressions/clicks/feedback; retrain CF on app data nightly
- [ ] Monitoring (latency, LLM cost, error rates); A/B tests of rankers
- [ ] Social features, lists, stats

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM hallucinates movies/facts | Ground every title to catalog IDs; RAG; tools-only recommendations; measure hallucination rate |
| LLM cost/latency | Small model for parsing, caching, LLM only on top-K, precompute batch recs |
| Cold start (users & movies) | Onboarding, content + LLM generators, LightFM, adaptive weights |
| Popularity bias / filter bubble | MMR diversity, novelty metric, popularity-penalized features |
| Data leakage in evaluation | Strict time-based splits; fit all preprocessing on train only |
| MovieLens ≠ our users | Treat MovieLens as pretraining; retrain on app data once available; Letterboxd import |
| API terms (TMDB, IMDb) | Cache responsibly, attribute TMDB, keep IMDb data non-commercial |
| Scope creep from the UI | Ship the ML + chatbot first; UI is phased with a strict MVP list |

---

## 13. Open Questions

1. Explicit ratings only, or also implicit signals (watched, liked, watchlisted) — likely both, with separate models.
2. Hosted embeddings vs. local `sentence-transformers` (cost vs. quality).
3. Movies only, or also TV later?
4. Public deployment with real users, or a portfolio/demo project? (Affects auth, moderation, and infra choices.)
5. Name and branding for the app.

---

## 14. Immediate Next Steps

1. Scaffold the repo (Phase 0).
2. Write the MovieLens + TMDB ingestion scripts.
3. Build the evaluation harness and popularity baseline so every later model is measured from day one.
