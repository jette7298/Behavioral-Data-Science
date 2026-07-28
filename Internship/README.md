# BDS Internship Project - Code Snippets

This folder contains examples for the code I used to complete this project. It is structured in the 3 phases I worked in:
automated question generation (AQG), digital-twin response generation, and
digital-twin response analysis. The files document the core methods used in the work but they do not reproduce the full 
pipeline due to compliance and confidentiality issues.

There is no data uploaded, as this is confidential. Referenced column names are paths are generic.

## Strcture

| Report topic | File | Main function(s) |
|---|---|---|
| Item-bank cleaning | `src/automated_question_generation/item_bank_cleaning.py` | `clean_item_bank` |
| Embeddings and FAISS | `src/automated_question_generation/embedding_index.py` | `embed_items`, `build_faiss_index` |
| Retrieval and reranking | `src/automated_question_generation/retrieval_reranking.py` | `retrieve_and_rerank` |
| Streamlit UI layout | `src/automated_question_generation/streamlit_interface.py` | `main` |
| Flesch + Kruskal-Wallis + Dunn | `src/automated_question_generation/evaluation.py` | `evaluate_readability` |
| Semantic similarity + statistical tests | `src/automated_question_generation/evaluation.py` | `evaluate_semantic_similarity` |
| Persona sampling | `src/digital_twins_generation/sample_personas.py` | `sample_personas` |
| Digital-twin generation call | `src/digital_twins_generation/generate_responses.py` | `build_prompt`, `generate_response`, `save_response` |
| Human/synthetic dataset combination | `src/digital_twins_generation/combine_datasets.py` | `combine_datasets` |
| Bootstrap confidence interval | `src/digital_twin_analysis/categorical_analysis.py` | `bootstrap_percentage_gap_ci` |
| Chi-square + Benjamini-Hochberg | `src/digital_twin_analysis/categorical_analysis.py` | `compare_categorical_variables` |
| MATTR | `src/digital_twin_analysis/text_analysis.py` | `mattr` |
| Shannon entropy | `src/digital_twin_analysis/text_analysis.py` | `shannon_entropy` |
| Pairwise and nearest-human similarity | `src/digital_twin_analysis/text_analysis.py` | `semantic_similarity` |
| Sentiment analysis | `src/digital_twin_analysis/text_analysis.py` | `analyze_sentiment` |



The embedding and retrieval examples assume that Ollama is running locally
and that the selected embedding model has already been pulled. The isolated
Streamlit UI example is started from this folder with:

```bash
streamlit run src/automated_question_generation/streamlit_interface.py
```

The digital-twin generation example represents the call used in a managed
sandbox. It expects an already-authenticated Azure/OpenAI-compatible client.
Authentication, secret retrieval, tenant configuration, and internal endpoint
construction which are not included here for confidentiality reasons. 
The file is therefore evidence of the generation method, not a locally executable authentication setup.
