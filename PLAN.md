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
| [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) (start with `ml-latest-small` for dev) | 32M ratings + 2M tags from 200,948 users on 87,585 movies (collected Oct 2023, released May 2024) | Training data for CF. `links.csv` maps `movieId → imdbId, tmdbId`. **No public redistribution** — keep it out of git |
| [TMDB API](https://developer.themoviedb.org/) | Overviews, genres, cast, crew, keywords, posters, backdrops, release dates, runtime, language | Content features + UI images. Free for **non-commercial** use with attribution (TMDB logo + "uses the TMDB API but is not endorsed or certified by TMDB"); commercial use needs TMDB's written approval. Back off on HTTP 429 |
| (Optional) IMDb non-commercial datasets | Extra cast/crew, ratings counts | Personal/non-commercial use only |
| Letterboxd export (user-uploaded) | ZIP of CSVs: `ratings.csv`, `diary.csv`, `watched.csv`, `watchlist.csv`, … | Films are identified by Letterboxd URL + title/year, **not** TMDB ID → needs a matching step (title + year fuzzy match). Verify columns against a real export before building the importer |
| Our own app | Real users' ratings, reviews, watchlists | Feeds back into CF once the app has users |

> TMDB terms/rate limits and the Letterboxd export format were checked via third-party summaries — confirm against TMDB's terms page and a real export before relying on them.

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
| v1 | Matrix factorization — SVD (explicit ratings) and ALS/BPR (implicit "watched/liked") | `scikit-surprise` (≥1.1.5), `implicit` |
| v2 | Hybrid MF with item features (helps cold start) + content→CF-embedding mapper (§5.3) | `cornac`, or ~30 lines of our own code |
| v3 (stretch) | Two-tower neural model or sequential model (SASRec) | Own PyTorch implementation (check Cornac first) |

> **Not using:** LightFM (no release since 1.17 in March 2023, likely to break on current Python/NumPy) and RecBole (no release since 1.2.1 in Feb 2025). See §10a.

**Outputs:** user embeddings, item embeddings, `score_cf(user, movie)`, and `similar_cf(movie)` (items co-liked by the same users).

### 4.2 Content-Based Filtering (CB)

Learns from *what a movie is*. Works for any movie with metadata, including new releases, and powers "more like this."

**Feature sets per movie:**
- **Structured:** genres, director(s), top-billed cast, writers, keywords, decade, runtime bucket, original language, certification.
- **Text:** overview + tagline (+ optionally aggregated user tags/reviews).

**Representations:**
1. **Sparse:** TF-IDF / one-hot over genres, people, keywords (weighted — director and keywords usually matter more than 8th-billed actor).
2. **Dense:** sentence embeddings of a "movie document" (title + overview + genres + keywords + director + cast) via `sentence-transformers`, run locally. Candidates, smallest to strongest: `all-MiniLM-L6-v2` (fast baseline) → `nomic-embed-text-v1.5` or `Qwen3-Embedding-0.6B` (mid-size) → `BGE-M3` (strong, multilingual). Public benchmarks don't test movie descriptions, so pick by running 2–3 of them on our own "more like this" examples.
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

**Model choice** (Anthropic API prices per million input / output tokens, as of Oct 2026):

| Role | Model | Price |
|------|-------|-------|
| Query parsing, high-volume structured tasks | Claude Haiku 5.5 (`claude-haiku-5-5`) | $0.10 / $0.50 |
| Chat agent, re-ranking, explanations | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | $2 / $10 |
| Judge for chat evaluation (§7.3) | Claude Opus 5.5 (`claude-opus-5-5`) | $4 / $20 |

**API features to use:**
- **Structured outputs** and `strict: true` tools for guaranteed-valid JSON. The 5.5 models don't support forcing a specific tool call (`tool_choice` `any`/`tool`), so use `auto` + a prompt instruction + `strict: true`, or structured outputs.
- **Prompt caching** for the fixed system prompt and tool definitions in the chat agent.
- **Message Batches API** (about 50% cheaper, asynchronous) for every offline evaluation run.
- Also cache responses ourselves, keyed on (prompt, user-profile hash), to control cost and latency.

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

Cold start means having too little data to recommend well. There are three kinds.

**1. New user (no ratings yet)** — CF can't help, so the other recommenders carry them.
- **Onboarding:** pick 5–10 films you love from a poster grid. Choose grid films that are well-known (people have seen them), span genres and moods, and include some that split opinion — a film everyone likes tells us little.
- **History import:** Letterboxd / IMDb CSV upload turns a new user into a regular user on day one.
- **Chat as onboarding:** "Tell me three movies you love and one you hated" seeds the taste profile.
- **Use the history-free recommenders:** content-based (from onboarding picks), LLM (needs only a few seeds), and popular/trending within the chosen genres.
- **Shift weight as ratings arrive:** e.g. `w_cf = n / (n + k)` with `n` = number of ratings and `k` ≈ 10–20, sliding smoothly from content/LLM toward CF. The LTR ranker can learn this itself if given `n` as a feature.
- **Exploration slots:** for new users, reserve 1–2 slots per list for films outside their stated taste, so we learn more than they told us.

**2. New movie (no ratings yet)**
- **Content-based** works from release day (overview, cast, keywords).
- **Content → CF mapping:** train a small model that predicts a movie's CF embedding from its content features, so a brand-new movie can be scored like any other (the DropoutNet idea). Cornac's hybrid models are an alternative.
- **LLM:** knows movies released before its training cutoff; for newer ones, put the TMDB overview and details in the prompt instead of relying on its memory.
- **Guaranteed exposure:** a small, time-limited boost for new releases so they collect ratings instead of staying invisible.

**3. New system (the app has no users yet)** — train on MovieLens first, retrain on app data as it accumulates; Letterboxd import helps most here.

**Measuring it** (see §7.1):
- *New users:* truncate test users to their first 0 / 1 / 3 / 5 / 10 ratings and plot accuracy vs. number of ratings for each recommender. CF should overtake content/LLM somewhere around 10–20.
- *New movies:* hold ~10% of movies out of CF training entirely and measure how well each approach recommends them.

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
- **Cold-start slices:** report metrics separately for users with <5, 5–20, >20 ratings, plus the truncated-history and held-out-movie experiments from §5.3
- Use `ranx` (or LensKit's built-in metrics) rather than hand-writing NDCG/MAP

### 7.2 Ablations

Report every metric for: popularity, CF only, CB only, LLM only, CF+CB, CF+CB+LLM (weighted), CF+CB+LLM (LTR). This table is the main result of the project. The CF+CB vs. CF+CB+LLM rows answer "is the LLM worth its cost?"

### 7.3 LLM recommender evaluation

**In short:** the LLM's picks are "appropriate" if they (1) match what users actually went on to watch and like, (2) are real, unseen, follow the request, and come with true reasons, (3) aren't just the same famous films for everyone, and (4) for open-ended chat, fit what was asked. Only (4) needs judgment; everything else is scored against real ratings and movie data.

**Level 1 — Offline accuracy (comparable to the other models)**
- **A. Open-ended top-K:** give the LLM a test user's ~20 most recent liked films from the training period, ask for 20 recommendations (structured output), match titles to catalog IDs, and compute Recall@K / NDCG@K / Hit Rate against held-out ratings. Expect low numbers — guessing the exact next film out of tens of thousands is hard.
- **B. Ranking a fixed candidate list** (the standard setup in LLM-rec research): 1 film the user actually liked in the test period + 19 randomly sampled unwatched films. Ask the LLM to rank them; measure Hit Rate@1/5/10 and mean rank. Run CF and CB on the *same* 20 candidates. This is the fair comparison and tests the LLM's actual job in the ensemble (re-ranking).
- **Fairness:** fixed sample of ~500–1,000 users shared by all models (controls API cost); run 3× and report mean and spread; run via the Batch API.
- **Position bias:** shuffle candidate order and re-run. LLMs are known to favor items near the top of a list; large ranking changes under shuffling are a measured problem, not noise.

**Level 2 — Reliability (fully automatic, checked against the database)**

| Metric | What it measures |
|---|---|
| Hallucination rate | % of suggested titles that don't match any catalog movie |
| Already-seen rate | % of picks the user has already rated |
| Duplicate rate | Same film twice, or near-duplicates (sequel, remake) |
| Constraint violations | e.g. "no horror" or "after 2015" broken — checked against TMDB metadata |
| Format failures | Output that can't be parsed |
| Explanation accuracy | Every factual claim ("directed by Villeneuve", "stars Amy Adams") verified against TMDB. A good pick with false reasons still breaks trust |

**Level 3 — Quality beyond accuracy**
- Coverage, novelty, diversity — is it personalizing, or always suggesting the IMDb Top 250?
- Separate results for movies released before vs. after the model's training cutoff, to see how much it depends on the details we put in the prompt.
- Its contribution to the ensemble (§7.2).

**Level 4 — Chatbot evaluation**
- A fixed **test set of ~50–100 chat prompts**: single seed film ("like Arrival"), person ("Amy Adams dramas"), mood ("cozy rainy-day movie"), stacked constraints ("90s, under 2 hours, not English"), multi-turn refinement ("funnier", "no, older"), vague requests, and off-topic requests.
- **Automatic checks:** constraint satisfaction, hallucinated and already-seen picks, factual claims — all against the database.
- **Judge model** for relevance, variety and explanation quality, using a written scoring guide. This is the **only** place an LLM judge is used.
  - Judge is a different, stronger model than the one answering (Claude Opus 5.5 judging Claude Sonnet 5.5), to avoid it going easy on its own outputs.
  - Validate it first: hand-grade ~30 answers and check the judge mostly agrees; repeat whenever the scoring guide changes.
  - Prefer **side-by-side comparisons** (version A vs. B) over 1–10 scores, and swap the A/B order on a second run to cancel order bias.
  - The judge never decides facts — anything checkable is checked against data.
  - Fallback: if we'd rather not rely on a judge, hand-grade the set (~1 hour per comparison at 50 prompts).
- Track cost and response time per conversation.
- The `/claude-api` skill's `build-eval` and `hillclimb` commands can scaffold this test set and iterate prompts against it.

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
| Language (ML/backend) | Python 3.12+ (LensKit needs ≥3.12.5) | ML ecosystem |
| Data | pandas / polars, Parquet | Fast local processing |
| ML | scikit-learn, implicit, scikit-surprise, Cornac, LightGBM, sentence-transformers, PyTorch (stretch) | Standard recsys tools (see §10a) |
| Evaluation | ranx, LensKit | Tested ranking metrics |
| Title matching | rapidfuzz | Ground LLM titles to catalog IDs |
| Vector search | pgvector (prod), FAISS (experiments) | One database to run |
| LLM | Anthropic Claude API — Haiku 5.5 / Sonnet 5.5 / Opus 5.5 (§4.3) | Agent + rerank + parsing + judge |
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
│   │   ├── collaborative.py    # kNN, SVD, ALS/BPR, Cornac hybrids
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

## 10a. Libraries, Research & Skills to Review Before Building

### Libraries (latest PyPI releases, checked Oct 2026)

| Purpose | Library | Latest release | Notes |
|---|---|---|---|
| CF on implicit signals | `implicit` | 0.7.3 (May 2026) | Fast ALS/BPR, optional GPU. Main CF choice |
| CF on star ratings | `scikit-surprise` | 1.1.5 (May 2026) | Revived after years without a release; older versions failed to install on newer Python, so pin ≥1.1.5 |
| Hybrid / research models | `cornac` | 3.0.1 (Sep 2026) | Replaces LightFM; built-in model comparison workflow |
| Evaluation framework | `lenskit` | 2026.4.0 (Sep 2026) | Rebuilt in 2025; needs Python ≥3.12.5; many online tutorials target the old API |
| Ranking metrics | `ranx` | 0.3.21 (Aug 2025) | NDCG, Recall, MAP |
| Learning-to-rank | `lightgbm` | 4.7.0 (Jul 2026) | `LGBMRanker` (LambdaRank) |
| Text embeddings | `sentence-transformers` | 6.1.0 (Sep 2026) | Local embedding models (§4.2) |
| Vector search | `pgvector` | 0.5.1 (Oct 2026) | Postgres vector search client |
| Fuzzy title matching | `rapidfuzz` | 3.14.6 (Aug 2026) | LLM title grounding, Letterboxd import |

**Avoid:** `lightfm` (last release 1.17, March 2023), `recbole` (last release 1.2.1, Feb 2025), `recpack` (last release Dec 2023).

Use Python 3.12+ (LensKit requires ≥3.12.5).

### Research to read first

- [Beyond Utility: Evaluating LLM as Recommender](https://arxiv.org/html/2411.00331v1) — LLMs are best at re-ranking and with short histories, show strong candidate-position bias, and some models invent items. Motivates the shuffling and hallucination checks in §7.3.
- [Large Language Models as Recommender Systems: A Study of Popularity Bias](https://arxiv.org/pdf/2406.01285) — movie-specific; LLMs showed less popularity bias than expected, and prompting reduced it further at some accuracy cost.
- [Can LLMs Recommend as well as Modern RecSys? (Criteo meta-review, Oct 2025)](https://www.criteo.com/wp-content/uploads/2021/06/Can_LLMs_Recommend_as_well_as_Modern_RecSys__A_Meta_Review___v2.pdf) — LLMs help with flexibility and cold start but trail classic models on raw accuracy; supports using the LLM inside the ensemble, not alone.
- [A Survey on Large Language Models for Recommendation](https://arxiv.org/pdf/2305.19860) (2023) — background.

### Skills

- **Claude Code skills:** `/claude-api` (current Claude API reference; `build-eval` and `hillclimb` for the chat test set) and `/session-start-hook` (run tests and linting automatically in cloud sessions).
- **To learn:** leakage-free recommender evaluation (time splits, sampled-candidate ranking, NDCG/Recall); matrix factorization basics (ALS vs. BPR, implicit vs. explicit feedback); LLM tool use and structured outputs; pgvector and approximate nearest-neighbor search.

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
- [ ] Cold-start experiments: truncated user histories, held-out movies (§5.3)

### Phase 3 — Collaborative filtering (1–2 weeks)
- [ ] Item–item kNN, SVD, ALS/BPR; hyperparameter tuning
- [ ] Cornac hybrid model and/or content→CF embedding mapper for new movies
- [ ] `similar_cf` and `score_cf` interfaces

### Phase 4 — Content-based (1 week)
- [ ] TF-IDF + dense embeddings, vector index
- [ ] Compare 2–3 embedding models on hand-picked "more like this" examples
- [ ] User profile vectors, `similar_cb`, `score_cb`

### Phase 5 — LLM recommender (1–2 weeks)
- [ ] Query parser with structured output
- [ ] LLM candidate generator + title grounding (fuzzy + vector)
- [ ] LLM reranker with explanations; response caching
- [ ] Offline accuracy: open-ended top-K and 20-candidate ranking, via Batch API (§7.3 Level 1)
- [ ] Position-bias check (shuffled candidate order)
- [ ] Reliability metrics: hallucination, already-seen, duplicates, constraint violations, explanation accuracy (§7.3 Level 2)

### Phase 6 — Ensemble (1–2 weeks)
- [ ] Candidate merging + filtering
- [ ] Weighted hybrid with tuned / adaptive weights
- [ ] LightGBM LambdaRank ranker, MMR diversity
- [ ] **Full ablation table** — the headline result

### Phase 7 — Chatbot (1–2 weeks)
- [ ] Tool definitions over the ensemble
- [ ] Agent loop with conversation state and guardrails
- [ ] CLI, then Streamlit/Gradio demo
- [ ] Chat test set (~50–100 prompts) with automatic checks
- [ ] Judge model (Opus 5.5) validated against ~30 hand-graded answers; side-by-side comparisons

### Phase 8 — API (1 week)
- [ ] FastAPI: `/movies`, `/search`, `/recommend`, `/similar`, `/chat` (SSE), `/logs`, `/watchlist`
- [ ] Recommendation caching in Redis; nightly batch precompute

### Phase 9 — Web app MVP (3–4 weeks)
- [ ] Next.js scaffold, auth, design system (dark theme)
- [ ] Film page, search, log/rate/review, diary, watchlist, profile
- [ ] Onboarding flow, "For You" feed, chat panel
- [ ] Letterboxd CSV import (title + year matching to TMDB IDs)

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
| Cold start (users & movies) | Onboarding, history import, content + LLM generators, content→CF mapping, adaptive weights (§5.3) |
| Popularity bias / filter bubble | MMR diversity, novelty metric, popularity-penalized features |
| Data leakage in evaluation | Strict time-based splits; fit all preprocessing on train only |
| MovieLens ≠ our users | Treat MovieLens as pretraining; retrain on app data once available; Letterboxd import |
| API terms (TMDB, IMDb, MovieLens) | Cache responsibly, attribute TMDB, get TMDB approval before any commercial use, keep IMDb data non-commercial, don't redistribute MovieLens |
| Unmaintained dependencies | Avoid LightFM/RecBole; pin versions; check release history before adding a library (§10a) |
| LLM position bias in re-ranking | Shuffle candidate order; measure ranking stability (§7.3) |
| Scope creep from the UI | Ship the ML + chatbot first; UI is phased with a strict MVP list |

---

## 13. Open Questions

1. Explicit ratings only, or also implicit signals (watched, liked, watchlisted) — likely both, with separate models.
2. Which local embedding model wins on our data (§4.2)?
3. Movies only, or also TV later?
4. Public deployment with real users, or a portfolio/demo project? (Affects auth, moderation, and infra choices.)
5. Name and branding for the app.
6. Chat evaluation: keep the validated judge model, or hand-grade only (§7.3 Level 4)?

---

## 14. Immediate Next Steps

1. Scaffold the repo (Phase 0).
2. Write the MovieLens + TMDB ingestion scripts.
3. Build the evaluation harness and popularity baseline so every later model is measured from day one.
