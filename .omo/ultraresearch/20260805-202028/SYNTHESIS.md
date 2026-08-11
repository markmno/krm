# Ultraresearch Synthesis: Competency-Role Models from Russian Job Vacancy Text (2026)

**Workers**: 12 parallel searches · **Sources**: 20+ · **Key paper**: arXiv:2407.19816

## Executive Summary

For building competency-role models from Russian job vacancy text in 2026, the evidence points to a clear technology stack:

1. **Embedding model**: `deepvk/USER-bge-m3` for quality (Encodechka Mean S=0.799) or `sergeyzh/rubert-tiny-turbo` for speed (111MB, Mean S=0.749, 5.5ms on CPU)
2. **Skill extraction**: Fine-tuned RuBERT NER (outperforms LLMs per arXiv:2407.19816)
3. **Text preprocessing**: Natasha ecosystem (razdel + slovnet for tokenization, morphology, NER)
4. **Competency clustering**: BERTopic with Russian sentence embeddings + c-TF-IDF
5. **Graph analysis**: Co-occurrence networks with networkx from extracted skills
6. **Data source**: HH.ru public API (`GET /vacancies`) — search endpoint returns `key_skills` field

The definitive benchmark for Russian sentence encoders is **encodechka** (avidale/encodechka, 246★), with **ruMTEB** (SberDevices/embeddings-benchmark) recommended for retrieval/reranking tasks.

## Findings by Theme

### 1. Russian NLP 2026 — Embedding Models

**navec is NOT the best choice for sentence-level tasks.** It is word-level only (GloVe + quantization). For competency modeling from vacancy text, sentence/paragraph embeddings are required.

**Encodechka Leaderboard Top Models** [Source 1]:
| Model | Quality (Mean S) | Size | CPU time |
|-------|-----------------|------|----------|
| deepvk/USER-bge-m3 | 0.799 | 1.4 GB | 523 ms |
| intfloat/multilingual-e5-large | 0.780 | 2.1 GB | 507 ms |
| deepvk/USER-base | 0.772 | 473 MB | 33 ms |
| sergeyzh/rubert-tiny-turbo | 0.749 | 111 MB | 5.5 ms |
| intfloat/multilingual-e5-base | 0.761 | 1.0 GB | 131 ms |
| cointegrated/rubert-tiny2 | 0.704 | 111 MB | 5.5 ms |

**Recommendation**: `deepvk/USER-bge-m3` for offline batch processing; `sergeyzh/rubert-tiny-turbo` for production API.

**Newer models (2024-2025)**: `deepvk/USER2-base` (March 2025 update), `sergeyzh/rubert-mini-frida`, `sergeyzh/BERTA`.

**ruMTEB benchmark** (SberDevices, 2024): 23 Russian tasks including retrieval and reranking [Source 2]. Recommended over Encodechka for retrieval-focused pipelines.

**Evidence** ([Source 1](https://github.com/avidale/encodechka)):
```
#1: deepvk/USER-bge-m3 (VK fine-tuned BGE-M3 for Russian) — Mean S=0.799
#5 overall: sergeyzh/rubert-tiny-turbo — Mean S=0.749, only 111MB, 312-dim
```

### 2. Skill Extraction from Russian Job Vacancies

**Key finding from arXiv:2407.19816** [Source 3]: Encoder-based NER (DeepPavlov RuBERT fine-tuned) **outperforms LLMs** (Saiga/Mistral, RuGPT) for skill extraction from Russian job vacancies — better accuracy, precision, recall, AND faster inference.

This paper is from HSE University (2024), with a labeled dataset of 4,000 training + 1,472 test vacancies. The dataset is NOT publicly available but the methodology is clear.

**Recommended approach**:
1. Annotate ~500-1000 vacancies with BIO skill tags
2. Fine-tune `DeepPavlov/rubert-base-cased` NER head
3. Traditional NER > LLM prompting for this task

**Evidence** ([Source 3](https://arxiv.org/abs/2407.19816)):
> "Results indicate that traditional NER models, especially DeepPavlov RuBERT NER tuned, outperform LLMs across various metrics including accuracy, precision, recall, and inference time."

### 3. Competency Clustering with BERTopic

BERTopic + Russian sentence embeddings is the recommended approach [Source 4]:

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Best quality
model = SentenceTransformer("deepvk/USER-bge-m3")
# Best speed
# model = SentenceTransformer("sergeyzh/rubert-tiny-turbo")

topic_model = BERTopic(
    embedding_model=model,
    nr_topics="auto",  # automatic topic reduction
    min_topic_size=10,
    calculate_probabilities=True
)
topics, probs = topic_model.fit_transform(skill_descriptions)

# Hierarchical merging for competency hierarchies
hierarchical_topics = topic_model.hierarchical_topics(skill_descriptions)

# Get competency labels via c-TF-IDF
topic_info = topic_model.get_topic_info()
```

**Evidence** ([Source 4](https://github.com/maartengr/bertopic)): c-TF-IDF topic representation, hierarchical topic merging via `hierarchical_topics()`, scipy ward linkage.

### 4. Russian NLP Ecosystem (Natasha Project)

The Natasha ecosystem [Source 5] provides all preprocessing needed:
- **Razdel**: Tokenization, sentence segmentation
- **Slovnet**: Morphology (POS, feats), Syntax parsing, NER (PER/LOC/ORG)
- **Navec**: Word embeddings (use only for word-level, not sentence-level)
- **Yargy**: Rule-based fact extraction
- **Pymorphy2**: Lemmatization

```python
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, NewsNERTagger, Doc
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
ner_tagger = NewsNERTagger(emb)
doc = Doc(vacancy_text)
doc.segment(segmenter)
doc.tag_morph(morph_tagger)
doc.tag_ner(ner_tagger)
```

**Evidence** ([Source 5](https://github.com/natasha/natasha)): 1.3k stars, MIT license, production-ready CPU inference.

### 5. HH.ru API

**Status**: Public API is well-documented at `https://api.hh.ru/openapi/redoc` [Source 6]. The vacancy search endpoint (`GET /vacancies`) supports text search, professional roles filtering, and returns structured data:

Key fields in vacancy response:
- `name`: Job title
- `description`: Full text description
- `key_skills`: Array of skill objects `[{name: "Python"}]`
- `professional_roles`: Array of `[{id, name}]` — HH.ru's own taxonomy
- `experience`, `schedule`, `employment`, `area`

The `key_skills` field provides structured skill data that can be used as ground truth for NER training.

**No evidence of "structured skills since 2025"** — the key_skills field has existed for years. No major API changes detected.

**HuggingFace datasets**: `trewwxsav/IT_vacancies_from_hh.ru` — IT vacancies dataset [Source 7].

**Evidence** ([Source 6](https://github.com/hhru/api)): 617 stars, 760 commits, OpenAPI spec at `https://api.hh.ru/openapi/specification/public`.

### 6. Existing Implementations & Papers

| Resource | Type | Relevance |
|----------|------|-----------|
| arXiv:2407.19816 | Paper (HSE, 2024) | **Directly on skill extraction from Russian vacancies** |
| avidale/encodechka | Benchmark (246★) | Russian sentence encoder comparison — choose your model |
| natasha/natasha | Library (1.3k★) | Full Russian NLP pipeline (tokenize, morph, NER) |
| natasha/navec | Library (219★) | Russian word embeddings |
| natasha/nerus | Corpus | 700K annotated Russian news texts |
| hhru/api | API docs (617★) | HH.ru vacancy data source |
| maartengr/bertopic | Library | Topic modeling with embeddings |
| cointegrated/rubert-tiny2 | Model (HF) | Widely used Russian sentence encoder |
| deepvk/USER-bge-m3 | Model (HF) | Best Russian encoder per Encodechka |
| trewwxsav/IT_vacancies_from_hh.ru | Dataset (HF) | IT vacancies from hh.ru |

### 7. LLMs for Russian (YandexGPT, GigaChat)

No direct evidence found for YandexGPT or GigaChat being used for skill extraction at scale. The arXiv paper shows that for Russian skill extraction specifically, fine-tuned encoder-based models outperform LLMs. For general text understanding tasks, `deepvk/USER-bge-m3` provides SOTA embeddings.

## Recommended Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│ HH.ru API   │────▶│ Preprocessing │────▶│ Skill Extraction│
│ /vacancies  │     │ (Natasha)     │     │ (RuBERT NER)   │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                 │
                      ┌──────────────────────────┘
                      ▼
┌───────────────┐     ┌──────────────┐     ┌───────────────┐
│ Competency    │◀────│ BERTopic     │◀────│ Sentence       │
│ Taxonomy      │     │ Clustering   │     │ Embeddings     │
│               │     │              │     │ (USER-bge-m3)  │
└───────────────┘     └──────────────┘     └───────────────┘
        │
        ▼
┌───────────────┐     ┌──────────────┐
│ Role Modeling │────▶│ Co-occurrence │
│ (job→skills)  │     │ Graph (netx)  │
└───────────────┘     └──────────────┘
```

## Sources (Ranked)

1. **avidale/encodechka** (GitHub, 246★) — Russian sentence encoder benchmark. Reliability: High. [Link](https://github.com/avidale/encodechka)
2. **ruMTEB** (SberDevices via MTEB) — Russian embedding benchmark with 23 tasks. [Post](https://habr.com/ru/companies/sberdevices/articles/831150/), [Code](https://github.com/embeddings-benchmark/mteb)
3. **arXiv:2407.19816** — Matkin et al. (2024), "Skill Extraction from Russian Job Vacancies." Reliability: High (peer-reviewed pre-print). [Link](https://arxiv.org/abs/2407.19816)
4. **maartengr/bertopic** (GitHub) — Topic modeling with embeddings. Reliability: High. [Link](https://github.com/maartengr/bertopic)
5. **natasha/natasha** (GitHub, 1.3k★) — Russian NLP pipeline. Reliability: High. [Link](https://github.com/natasha/natasha)
6. **hhru/api** (GitHub, 617★) — HH.ru API docs. Reliability: High (official). [Link](https://github.com/hhru/api)
7. **trewwxsav/IT_vacancies_from_hh.ru** (HuggingFace) — IT vacancies dataset. [Link](https://huggingface.co/datasets/trewwxsav/IT_vacancies_from_hh.ru)
8. **HuggingFace Russian Sentence Similarity Models** — Top downloaded models list. [Link](https://huggingface.co/models?pipeline_tag=sentence-similarity&language=ru&sort=downloads)

## Gaps

- **No open-source competency model builder found**: Most HR tech is proprietary (HH.ru, SuperJob internal tools)
- **No public labeled dataset**: The arXiv paper's 4,000-vacancy dataset is not public. Need to create one using HH.ru API `key_skills` field as weak labels
- **No YandexGPT/GigaChat skill extraction evaluation**: LLM-based approach for Russian skill extraction untested beyond the arXiv paper
- **HH.ru API `key_skills` quality**: Unknown how comprehensive or consistent the structured skills are
- **No graph-based competency modeling tools found**: Would need custom implementation with networkx or igraph

## Expansion Trace

- **Wave 1**: 12 parallel searches (Context7 × 3, GitHub grep × 4, web fetch × 5)
- **Key leads investigated**: Encodechka leaderboard, arXiv paper, HH.ru API docs, BERTopic multilingual, HuggingFace Russian models, Natasha ecosystem
- **Convergence**: All axes covered, evidence for every recommendation
