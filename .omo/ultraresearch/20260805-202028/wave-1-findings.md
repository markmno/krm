# Wave 1 - Saturation Findings

## Axis 1: Russian NLP 2026 SOTA

### Word Embeddings
- **navec** (natasha/navec): Still the standard for Russian word embeddings. 50MB, 500K vocab, GloVe + quantization. 219★. Used by all Natasha ecosystem projects.
  - Source: https://github.com/natasha/navec
  - NOT the best for sentence-level tasks. Word-level only.
  
### Sentence/Paragraph Embeddings (Encodechka Leaderboard)
- **#1: deepvk/USER-bge-m3** (VK): Fine-tuned BGE-M3 for Russian. Best Mean S=0.799. 1.4GB, 1024-dim.
- **#2: BAAI/bge-m3**: Same base, slightly worse on Russian. 2.2GB.
- **#3: intfloat/multilingual-e5-large-instruct**: 2.1GB, Mean S=0.784.
- **Best small: sergeyzh/rubert-tiny-turbo**: 111MB, 312-dim, Mean S=0.749, CPU 5.5ms. Excellent quality/speed ratio.
- **Best balanced: deepvk/USER-base**: 473MB, 768-dim, Mean S=0.772.
- **Serious contender: intfloat/multilingual-e5-base**: 1GB, 768-dim, Mean S=0.761.
- **cointegrated/rubert-tiny2**: 111MB, 312-dim, Mean S=0.704. Widely used, good baseline.
- **sentence-transformers/LaBSE**: Good multilingual, 1.8GB, Mean S=0.739.
- Source: https://github.com/avidale/encodechka (246★)
- Benchmark: https://huggingface.co/spaces/Samoed/Encodechka
- **ruMTEB (SberDevices)**: Russian MTEB benchmark with 23 tasks including retrieval/reranking. Recommended over Encodechka for full evaluation.
  - Code: https://github.com/embeddings-benchmark/mteb
  - Post: https://habr.com/ru/companies/sberdevices/articles/831150/

### Russian-specific Models (HuggingFace Top Downloads, sentence-similarity, Russian):
1. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (59.4M downloads)
2. intfloat/multilingual-e5-small (15.6M)
3. intfloat/multilingual-e5-base (7.15M)
4. sentence-transformers/LaBSE (764k)
5. deepvk/USER-bge-m3 (405k)
6. cointegrated/rubert-tiny2 (362k)
7. deepvk/USER-base (62k)
8. deepvk/USER2-base (17k, newer)
9. sergeyzh/rubert-tiny-turbo (20k)
10. sergeyzh/rubert-mini-frida (11k)
11. sergeyzh/BERTA (13k)
- Source: https://huggingface.co/models?pipeline_tag=sentence-similarity&language=ru&sort=downloads

### Russian NLP Ecosystem (Natasha Project)
- **natasha** (1.3k★): Tokenization (Razdel), Morphology (Slovnet), NER, Syntax, Embeddings (Navec), Fact extraction (Yargy)
  - Source: https://github.com/natasha/natasha
- **DeepPavlov**: Conversational AI framework with ruBERT models
  - Source: https://github.com/deeppavlov/DeepPavlov
- **Nerus**: 700K silver-standard Russian news corpus with POS/NER/syntax markup
  - Source: https://github.com/natasha/nerus

## Axis 2: Competency Modeling Tools

### Key Paper
- **arXiv:2407.19816** (2024): "Comparative Analysis of Encoder-Based NER and Large Language Models for Skill Extraction from Russian Job Vacancies"
  - Authors: Matkin, Smirnov, Usanin, Ivanov, Sobyanin, Paklina, Parshakov (HSE University)
  - Dataset: 4,000 labeled Russian job vacancies (train) + 1,472 test
  - **Key finding**: DeepPavlov RuBERT NER tuned OUTPERFORMS LLMs (Saiga/Mistral, RuGPT) on skill extraction F1
  - Best model: Encoder-based NER (RuBERT fine-tuned) - higher accuracy, precision, recall, AND faster inference
  - Source: https://arxiv.org/abs/2407.19816
  
### Open Source Tools
- **ESCO (European Skills/Competences)**: EU taxonomy of skills. No Russian-specific version found.
- No open-source "competency model builder" found. Most HR tech is proprietary.
- **DeepPavlov NER**: Can be fine-tuned for skill extraction from Russian text.

## Axis 3: Job Vacancy Analysis Best Practices

### BERTopic for Russian Text
- Use `BERTopic(embedding_model=SentenceTransformer("deepvk/USER-bge-m3"))` for best quality
- Or `embedding_model=SentenceTransformer("sergeyzh/rubert-tiny-turbo")` for speed
- c-TF-IDF topic representation, hierarchical topic merging with `hierarchical_topics()`
- Source: https://github.com/maartengr/bertopic

### Skill Extraction Approach (from arXiv paper):
1. Label dataset of vacancies with skill annotations (BIO tagging)
2. Fine-tune RuBERT NER model
3. Traditional NER > LLMs for this task per the paper

### Multilingual Sentence Transformers
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - most downloaded
- `intfloat/multilingual-e5-*` family - very strong
- For clustering: encode vacancies, then HDBSCAN/UMAP

## Axis 4: HH.ru Specifics

### API
- HH.ru has a public API: https://dev.hh.ru/ and https://github.com/hhru/api
- The API returns vacancy descriptions, key skills (somewhat structured), professional roles
- No evidence of "structured skills since 2025" changes found

### Datasets
- **HuggingFace**: `trewwxsav/IT_vacancies_from_hh.ru` - IT vacancies dataset
- **HuggingFace**: Various `hh-rlhf-ru` datasets (RLHF data from HH, not vacancy analysis)
- No large-scale annotated skill extraction dataset publicly available beyond the 4,000 in the arXiv paper

### Authoritative Work
- The arXiv paper (Matkin et al., 2024) is from HSE University - the leading Russian economics university, suggesting academic interest in this space

## Key Recommendations

### Embedding Model Choice:
| Use Case | Model | Size | Quality (Mean S) |
|----------|-------|------|-------------------|
| Best quality | deepvk/USER-bge-m3 | 1.4GB | 0.799 |
| Balanced | deepvk/USER-base | 473MB | 0.772 |
| Best speed/size | sergeyzh/rubert-tiny-turbo | 111MB | 0.749 |
| Most downloaded | intfloat/multilingual-e5-base | 1GB | 0.761 |
| Widely used Russian | cointegrated/rubert-tiny2 | 111MB | 0.704 |

### Pipeline Recommendation:
1. Fetch vacancies via HH.ru API
2. Preprocess with Natasha (tokenization, lemmatization, NER)
3. Extract skills via fine-tuned RuBERT NER (per arXiv paper method)
4. Embed extracted skills/descriptions with deepvk/USER-bge-m3 or rubert-tiny-turbo
5. Cluster skills with BERTopic + HDBSCAN
6. Build co-occurrence graph with networkx
7. Normalize job titles via Natasha NER + fuzzy matching
