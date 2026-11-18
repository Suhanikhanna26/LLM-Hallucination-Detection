# LLM Hallucination Detector

Evidence-grounded question answering and claim-level hallucination-risk analysis for uploaded research-paper PDFs.

## Pipeline
PDF extraction → page-aware chunks → `all-MiniLM-L6-v2` embeddings → FAISS + keyword retrieval → Gemini answer generation → claim extraction → batched `roberta-large-mnli` verification → project-defined risk score.

## Local setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Gemini API key to `.env`, then run:

```powershell
uvicorn app:app --env-file .env --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000`.

## Configuration
- `GEMINI_API_KEY`: required for real generation.
- `GEMINI_MODEL`: model ID available to your Google AI Studio API key. Verify it in AI Studio before deployment.
- `DEMO_MODE`: defaults to `false`; real analysis endpoints reject demo mode so mock answers cannot be mistaken for real results.
- `NLI_BATCH_SIZE`: claim/evidence pairs per inference batch (default 8).
- `MAX_UPLOAD_MB`: PDF upload limit (default 30).
- `DATA_DIR`, `UPLOAD_DIR`: persistent storage locations.

## Render hosting
Includes `render.yaml` and `Procfile`. The Blueprint config uses one Uvicorn worker and a persistent disk mounted at `/var/data`; enter `GEMINI_API_KEY` as a secret in Render. Review plan and disk charges before deploying. MiniLM and RoBERTa model downloads/inference can require substantial memory and CPU. Test a complete upload → retrieval → Gemini → NLI request on the selected instance before sharing the URL.

## Persistence
PDFs and document chunk metadata are stored under `UPLOAD_DIR` and `DATA_DIR`. Document metadata is restored after restart and FAISS indexes are rebuilt lazily from saved chunks. Keep one worker because the active document/index cache is process-local. A persistent disk is required to retain uploads/history across restarts.

## Risk-score interpretation
`risk_score = (contradicted_claims + 0.5 × unknown_claims) / total_claims`

SUPPORTED contributes 0, CONTRADICTED contributes 1, UNKNOWN contributes 0.5. This is a project-defined heuristic, not a universally standardized metric. If there are no extracted claims, risk is `UNCERTAIN`; a zero score with UNKNOWN claims is also `UNCERTAIN`.

## Claim extraction and NLI
Claim/evidence pairs are inferred in batches. Sentence splitting protects common academic abbreviations (`et al.`, `e.g.`, `i.e.`, `Fig.`), but remains a heuristic and may need refinement for unusual formatting.

## Existing HaluEval benchmark results
These are the project's existing results (200 validation examples and 400 test examples), not recomputed by this application.

| Metric | NLI baseline | RAG + NLI |
|---|---:|---:|
| Accuracy | 81.25% | 80.25% |
| Precision | 90.85% | 88.05% |
| Recall | 69.50% | 70.00% |
| F1 | 78.75% | 77.99% |
| ROC-AUC | 80.86% | 80.72% |

The app does not claim RAG + NLI outperforms the baseline. HaluEval evaluation and PDF-based document-grounded QA are related but distinct evaluation settings.

## Limitations
- Gemini model availability depends on API key, account, region and Google's current model catalog. Confirm `GEMINI_MODEL`.
- MiniLM and `roberta-large-mnli` require model downloads and substantial memory/disk; small/free instances may not be sufficient.
- FAISS indexes are rebuilt from persisted chunks after restart.
- Scanned/image-only PDFs need OCR, which is not included.
- The risk score is a heuristic and should be described as such in academic evaluation.
