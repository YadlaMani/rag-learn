# RAG Search Engine

Built this to understand how search actually works starting from raw keyword matching all the way to RAG and image search. Each approach had gaps that the next one fixes. The code and this doc follow that same progression.

Dataset is a JSON collection of movies (`data/movies.json`).

---

## The Stack

```
data/               movie corpus + golden evaluation set
cli/
├── keyword_search_lib/     inverted index, TF-IDF, BM25
├── semantic_search_lib/    dense embeddings, chunking, cosine similarity
├── hybrid_search_lib/      weighted fusion, RRF, query enhancement, re-ranking
├── multimodal_search_lib/  CLIP image embeddings
├── keyword_search_cli.py
├── semantic_search_cli.py
├── hybrid_search_cli.py
├── augmented_generation_cli.py
├── evaluation_cli.py
├── describe_image_cli.py
└── multimodal_search_cli.py
cache/              persisted embeddings and index files
```

---

## Evolution of Search

### Stage 1: Text Preprocessing

**[`cli/keyword_search_lib/preprocessor.py`](cli/keyword_search_lib/preprocessor.py)**

Before matching anything, the text needs cleaning. Three steps:

1. **Normalize** lowercase, strip punctuation, split on whitespace
2. **Stopword removal** drop words from `data/stopwords.txt` (the, a, in, …) that show up everywhere and mean nothing
3. **Stemming** Porter stemmer reduces words to their root: `running → run`, `searches → search`

Both the query and the documents go through the same pipeline so they're in the same token space when you compare them.

**Problem:** Now that you have clean tokens you can look up which documents contain a word. But that's just a yes/no no ranking, no sense of which document is more relevant.

---

### Stage 2: Inverted Index + Basic Keyword Search

**[`cli/keyword_search_lib/`](cli/keyword_search_lib/)**

Map every stem to the set of doc IDs that contain it. At query time, tokenize the query the same way, look up each token, union the doc sets, return the titles. The index is saved to `cache/` as pickle files so you don't rebuild it every run.

```
uv run cli/keyword_search_cli.py build
uv run cli/keyword_search_cli.py search "bear london"
```

**Problem:** A document either matches or it doesn't. There's no score so you can't say which result is better than another.

---

### Stage 3: TF-IDF

**[`cli/keyword_search_lib/`](cli/keyword_search_lib/)**

Two numbers multiplied to get a relevance score:

- **TF (Term Frequency)** how many times does the term appear in this doc? More → more relevant.
- **IDF (Inverse Document Frequency)** `log(N / df)`. Terms that appear in fewer documents carry more weight. "bear" in 5 of 500 movies means more than "movie" in 490 of 500.

```
uv run cli/keyword_search_cli.py idf "bear"
uv run cli/keyword_search_cli.py tfidf 42 "bear"
```

**Problem:** TF has no ceiling. A long document that mentions "bear" 20 times will score higher than a short one mentioning it 5 times, even if the short one is a better match. Repetition keeps adding to the score linearly which doesn't reflect actual relevance.

---

### Stage 4: BM25

**[`cli/keyword_search_lib/`](cli/keyword_search_lib/)**

BM25 fixes the two issues with TF-IDF:

- **TF saturation** the `k1` parameter (default 1.5) puts a ceiling on term frequency gains. The 20th mention barely adds anything over the 5th.
- **Document length normalization** the `b` parameter (default 0.75) penalizes longer documents so a short tight synopsis isn't beaten just by a longer one repeating the same word more.

BM25 also tweaks the IDF formula to `log((N - df + 0.5) / (df + 0.5) + 1)` to handle edge cases better.

```
uv run cli/keyword_search_cli.py bm25search "bear london"
uv run cli/keyword_search_cli.py bm25idf "bear"
uv run cli/keyword_search_cli.py bm25tf 42 "bear"
```

**Problem:** BM25 still only scores exact token matches. Search "cozy animated film" and a movie described as "heartwarming Pixar adventure" gets a score of zero no shared stems, no match, even though it's clearly what you're looking for. Keyword search can't handle vocabulary differences.

---

### Stage 5: Semantic Search

**[`cli/semantic_search_lib/`](cli/semantic_search_lib/)**

Instead of matching tokens, encode text into vectors where similar meaning ends up in similar positions. No shared words needed "cozy animated film" and "heartwarming Pixar adventure" will be close in vector space.

**Model:** `all-MiniLM-L6-v2` 384-dimensional vectors, fast enough to run locally.

**Scoring:** Cosine similarity between the query vector and each document vector. Measures the angle between them closer angle means more similar meaning.

Embeddings are computed once and cached to `cache/movie_embeddings.npy`. Next run just loads from disk.

```
uv run cli/semantic_search_cli.py verify
uv run cli/semantic_search_cli.py verify_embeddings
uv run cli/semantic_search_cli.py search "cozy animated adventure"
```

#### Chunked Semantic Search

Compressing a long movie description into one vector loses detail specifics from the middle of the text get averaged out. Instead, split each description into sentence-based chunks, embed each chunk separately, and score the document by its best-matching chunk.

```
uv run cli/semantic_search_cli.py embed_chunks
uv run cli/semantic_search_cli.py search_chunked "bear escapes london"
```

Chunk embeddings cached to `cache/chunk_embeddings.npy` + `cache/chunk_metadata.pkl`.

**Problem:** Semantic search is weak on exact matches. Searching "Paddington 2014" should immediately return the right movie but cosine similarity might rank a vaguely related film higher. Keyword search is good at exact matches, semantic is good at fuzzy ones. They fail in different places, so combining them should beat either alone.

---

### Stage 6: Hybrid Search

**[`cli/hybrid_search_lib/`](cli/hybrid_search_lib/)**

Two ways to combine BM25 and semantic results:

#### Weighted Fusion

Normalize both score lists to [0, 1], then blend with a weight:

```
hybrid = (1 - alpha) * bm25_score + alpha * semantic_score
```

Alpha 0.0 is pure BM25, 1.0 is pure semantic. You control the balance at query time.

```
uv run cli/hybrid_search_cli.py weighted-search "bear london" --alpha 0.3
```

#### Reciprocal Rank Fusion (RRF)

Rather than blending raw scores (which are on different scales), blend the ranks:

```
rrf_score = Σ  1 / (k + rank_i)
```

`k` (default 60) controls how much rank 1 dominates over lower ranks. Higher k flattens the difference, lower k makes the top result matter a lot more. Each retriever contributes a rank, scores are summed, results re-sorted. Works better than weighted fusion in practice because you don't need to normalize scores first.

```
uv run cli/hybrid_search_cli.py rrf-search "bear london" -k 60
```

**Problem:** Both retrievers are only as good as the query. Users misspell things, use vague descriptions, or phrase queries in ways that don't match the corpus. Bad query in, bad results out.

---

### Stage 7: Query Enhancement + Re-ranking

**[`cli/hybrid_search_lib/utils.py`](cli/hybrid_search_lib/utils.py)**

#### Query Enhancement (before retrieval)

Pass the query through an LLM (Gemma 4-31b-it via Gemini) before searching. Three modes:

- **spell** fix typos only, leave everything else alone
- **rewrite** rephrase into a cleaner search query using movie domain knowledge  
  `"that bear movie where leo gets attacked" → "The Revenant Leonardo DiCaprio bear attack"`
- **expand** add synonyms and related terms to the query to catch more matches

```
uv run cli/hybrid_search_cli.py rrf-search "cozy bear animted" --enhance spell
uv run cli/hybrid_search_cli.py rrf-search "bear london marmalade" --enhance rewrite
```

#### Re-ranking (after retrieval)

Fetch a bigger candidate set (5× what you want), then re-order it:

- **individual** LLM scores each result 0–10 against the query, one at a time
- **batch** LLM ranks the whole list in one shot, returns ordered IDs
- **cross_encoder** `cross-encoder/ms-marco-TinyBERT-L2-v2` looks at the query and document together in one pass, which is more accurate than comparing embeddings separately but slower

```
uv run cli/hybrid_search_cli.py rrf-search "bear london" --rerank-method cross_encoder
```

#### Evaluation

Precision@k, Recall@k, F1 against `data/golden_dataset.json` a set of queries with known correct answers to measure how well the whole pipeline is doing.

```
uv run cli/evaluation_cli.py --limit 5
```

**Problem:** You still get back a list of movies. The user has to read through them, figure out which ones are relevant, and piece together an answer themselves.

---

### Stage 8: Retrieval Augmented Generation

**[`cli/augmented_generation_cli.py`](cli/augmented_generation_cli.py)**

Retrieve relevant movies, stuff them into a prompt, let the LLM write the answer. The model only talks about movies that were actually retrieved it's grounded in the search results rather than making things up.

Four modes:

- **rag** direct answer grounded in retrieved movies
- **summarize** synthesize across multiple results into one summary
- **citations** same as summarize but attributes claims back to specific titles
- **question** casual conversational answers for exploratory questions

```
uv run cli/augmented_generation_cli.py rag "bear movie with emotional ending"
uv run cli/augmented_generation_cli.py question "what should I watch tonight if I liked Paddington"
uv run cli/augmented_generation_cli.py citations "animated movies about animals in cities"
```

Queries are expanded with `expand` before retrieval to get better coverage.

---

### Stage 9: Multimodal Search

**[`cli/multimodal_search_lib/`](cli/multimodal_search_lib/)** | **[`cli/multimodal_search_cli.py`](cli/multimodal_search_cli.py)**

CLIP (`clip-ViT-B-32`) puts images and text in the same 512-dimensional vector space. A photo of Paddington bear and the text "bear in London" end up near each other even though they share no tokens.

Movie text embeddings are built once and cached to `cache/clip_embeddings.npy` + `cache/clip_docmap.pkl`. At search time, the image is encoded and compared against all the text embeddings with cosine similarity.

```
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg
uv run cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg
```

#### Image Query Rewriting

**[`cli/describe_image_cli.py`](cli/describe_image_cli.py)**

You can also use an LLM to look at the image and rewrite a text query based on what it sees, then feed that into regular text search.

```
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "bear in London"
```

---

## Running

```bash
# install deps
uv sync

# build indexes first
uv run cli/keyword_search_cli.py build
uv run cli/semantic_search_cli.py verify_embeddings
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg  # builds clip cache on first run

# search
uv run cli/hybrid_search_cli.py rrf-search "bear london" --enhance rewrite --rerank-method cross_encoder

# rag
uv run cli/augmented_generation_cli.py rag "animated bear movie family friendly"

# evaluate
uv run cli/evaluation_cli.py --limit 5
```

`GEMINI_API_KEY` in `.env` is needed for query enhancement, LLM re-ranking, and RAG commands.

---

## Cache Files

| File                         | What it stores                         |
| ---------------------------- | -------------------------------------- |
| `cache/index.pkl`            | Inverted index (term → doc IDs)        |
| `cache/docmap.pkl`           | doc ID → movie metadata                |
| `cache/term_frequencies.pkl` | per-doc term counts                    |
| `cache/doc_lengths.pkl`      | document lengths for BM25              |
| `cache/movie_embeddings.npy` | semantic embeddings (all-MiniLM-L6-v2) |
| `cache/chunk_embeddings.npy` | per-chunk embeddings                   |
| `cache/chunk_metadata.pkl`   | chunk → movie mapping                  |
| `cache/clip_embeddings.npy`  | CLIP text embeddings for image search  |
| `cache/clip_docmap.pkl`      | doc ID → movie metadata for CLIP index |

---

Built following the [Learn Retrieval Augmented Generation](https://www.boot.dev/courses/learn-retrieval-augmented-generation) course on boot.dev.
