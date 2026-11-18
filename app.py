import os
import re
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
DATA = Path(os.getenv("DATA_DIR", str(BASE / "data")))
UPLOADS = Path(os.getenv("UPLOAD_DIR", str(BASE / "uploads")))
DATA.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = DATA / "history.json"
DOCUMENTS_FILE = DATA / "documents.json"

DEMO_MODE = os.getenv("DEMO_MODE", "false").strip().lower() == "true"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

app = FastAPI(title="LLM Hallucination Detector")
DOCS: dict[str, dict[str, Any]] = {}
EMBEDDER = None
NLI_TOKENIZER = None
NLI_MODEL = None
GEMINI_CLIENT = None


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_history():
    return load_json(HISTORY_FILE, [])


def save_history(items):
    HISTORY_FILE.write_text(json.dumps(items[-100:], indent=2), encoding="utf-8")


def save_documents():
    serializable = []
    for doc in DOCS.values():
        serializable.append({
            "id": doc["id"], "filename": doc["filename"], "pages": doc["pages"],
            "chunks": doc["chunks"], "path": doc["path"], "created": doc["created"]
        })
    temp = DOCUMENTS_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(serializable), encoding="utf-8")
    temp.replace(DOCUMENTS_FILE)


def restore_documents():
    for item in load_json(DOCUMENTS_FILE, []):
        path = Path(item.get("path", ""))
        if path.exists():
            item["index"] = None  # Rebuilt lazily from the persisted chunks.
            DOCS[item["id"]] = item


restore_documents()


def get_embedder():
    global EMBEDDER
    if EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return EMBEDDER


def get_nli():
    global NLI_TOKENIZER, NLI_MODEL
    if NLI_MODEL is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        NLI_TOKENIZER = AutoTokenizer.from_pretrained("roberta-large-mnli")
        NLI_MODEL = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        NLI_MODEL.eval()
    return NLI_TOKENIZER, NLI_MODEL


def get_gemini():
    global GEMINI_CLIENT
    if GEMINI_CLIENT is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured. Add it to the server environment.")
        try:
            from google import genai
            GEMINI_CLIENT = genai.Client(api_key=key)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Could not initialize Gemini: {exc}") from exc
    return GEMINI_CLIENT


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_pages(pages, chunk_words=170, overlap=35):
    chunks = []
    for page_no, text in enumerate(pages, 1):
        words = clean_text(text).split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunks.append({"text": " ".join(words[start:end]), "page": page_no})
            if end == len(words):
                break
            start = max(end - overlap, start + 1)
    return chunks


def build_index(doc):
    if doc.get("index") is None:
        import faiss
        emb = get_embedder()
        vectors = emb.encode(
            [c["text"] for c in doc["chunks"]],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        doc["index"] = index
    return doc["index"]


def keyword_score(question, text):
    q = set(re.findall(r"[a-zA-Z0-9]{3,}", question.lower()))
    t = set(re.findall(r"[a-zA-Z0-9]{3,}", text.lower()))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def retrieve(doc, question, top_k=5):
    index = build_index(doc)
    emb = get_embedder()
    qv = emb.encode([question], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(qv, min(max(top_k * 4, 12), len(doc["chunks"])))
    candidates = []
    broad_terms = re.search(
        r"\b(main contribution|objective|purpose|aim|methodology|method|findings|results|conclusion|summary|what is this paper about)\b",
        question.lower()
    )
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        ch = doc["chunks"][int(idx)]
        sem = float(dist)
        kw = keyword_score(question, ch["text"])
        page_bonus = 0.10 if broad_terms and ch["page"] <= 3 else 0.0
        score = 0.70 * sem + 0.25 * kw + page_bonus
        candidates.append((score, sem, ch))
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected, seen = [], set()
    for score, sem, ch in candidates:
        key = (ch["page"], ch["text"][:100])
        if key in seen:
            continue
        seen.add(key)
        selected.append({**ch, "retrieval_score": round(float(score), 4), "semantic_score": round(float(sem), 4)})
        if len(selected) >= top_k:
            break
    return selected


def grounded_answer(question, evidence):
    if DEMO_MODE:
        raise HTTPException(status_code=503, detail="Demo mode is enabled. Set DEMO_MODE=false for real document analysis.")
    client = get_gemini()
    context = "\n\n".join([f"[Page {e['page']}] {e['text']}" for e in evidence])
    prompt = f"""You are answering a question about a research paper. Use ONLY the evidence below. Do not add facts that are not supported by the evidence. If the evidence is insufficient, say so clearly. Keep the answer concise and factual.

Question: {question}

Evidence:
{context}

Answer:"""
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        answer = getattr(response, "text", None)
        if not answer or not answer.strip():
            raise RuntimeError("Gemini returned an empty response.")
        return answer.strip()
    except HTTPException:
        raise
    except Exception as exc:
        # Avoid returning API keys or raw provider details to the browser.
        print(f"Gemini generation error ({type(exc).__name__}): {exc}")
        raise HTTPException(
            status_code=502,
            detail="Gemini could not generate an answer. Check the API key, configured model name, quota and provider status."
        ) from exc


def extract_claims(answer):
    answer = re.sub(r"^(based on (the )?(provided|retrieved) evidence[,;:]?\s*)", "", answer.strip(), flags=re.I)
    if re.search(r"\b(insufficient evidence|not enough evidence|cannot determine|unable to determine)\b", answer, re.I):
        return []
    # Protect common academic abbreviations before splitting on sentence punctuation.
    protected = answer
    abbreviations = ["et al.", "e.g.", "i.e.", "Fig.", "Figs.", "Dr.", "Prof.", "vs.", "No.", "Eq.", "Refs."]
    replacements = {}
    for i, abbr in enumerate(abbreviations):
        token = f"__ABBR{i}__"
        protected = re.sub(re.escape(abbr), token, protected, flags=re.I)
        replacements[token] = abbr
    protected = re.sub(r"(?<=\b[A-Z])\.(?=\s+[A-Z])", "__INITIAL_PERIOD__", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", protected)
    claims = []
    for part in parts:
        p = part
        for token, abbr in replacements.items():
            p = p.replace(token, abbr)
        p = p.replace("__INITIAL_PERIOD__", ".").strip(" -•\n")
        if len(p.split()) >= 5 and not p.endswith(":"):
            claims.append(p)
    return claims[:8]


def verify_claims(claims, evidence):
    if DEMO_MODE:
        raise HTTPException(status_code=503, detail="Demo mode is enabled; NLI verification is disabled.")
    if not claims or not evidence:
        return []
    tokenizer, model = get_nli()
    pairs = [(e["text"], claim, ci, ei) for ci, claim in enumerate(claims) for ei, e in enumerate(evidence)]
    all_probs = []
    batch_size = int(os.getenv("NLI_BATCH_SIZE", "8"))
    try:
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            encoded = tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            with torch.inference_mode():
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs.tolist())
    except Exception as exc:
        print(f"NLI inference error ({type(exc).__name__}): {exc}")
        raise HTTPException(status_code=500, detail="Claim verification failed while running the NLI model.") from exc

    results = []
    for ci, claim in enumerate(claims):
        claim_pairs = [(all_probs[k], evidence[ei]) for k, (_, _, claim_i, ei) in enumerate(pairs) if claim_i == ci]
        best_ent = max(((float(p[2]), ev) for p, ev in claim_pairs), key=lambda x: x[0], default=(0.0, None))
        best_con = max(((float(p[0]), ev) for p, ev in claim_pairs), key=lambda x: x[0], default=(0.0, None))
        if best_con[0] >= 0.50 and best_con[0] > best_ent[0]:
            status, conf, ev = "CONTRADICTED", best_con[0], best_con[1]
        elif best_ent[0] >= 0.50:
            status, conf, ev = "SUPPORTED", best_ent[0], best_ent[1]
        else:
            status, conf, ev = "UNKNOWN", max(best_ent[0], best_con[0]), best_ent[1] or best_con[1]
        results.append({
            "claim": claim, "status": status, "confidence": round(float(conf), 4),
            "page": ev["page"] if ev else None, "evidence": ev["text"] if ev else ""
        })
    return results


def score_claims(claims):
    if not claims:
        return 0.0, "UNCERTAIN"
    total = len(claims)
    contradicted = sum(c["status"] == "CONTRADICTED" for c in claims)
    unknown = sum(c["status"] == "UNKNOWN" for c in claims)
    # Project-defined risk heuristic: UNKNOWN contributes half as much as contradiction.
    score = (contradicted + 0.5 * unknown) / total
    if contradicted and score >= 0.5:
        risk = "HIGH"
    elif contradicted or (unknown and score > 0):
        risk = "MEDIUM"
    elif unknown:
        risk = "UNCERTAIN"
    else:
        risk = "LOW"
    return round(score, 4), risk


class AnalyzeRequest(BaseModel):
    document_id: str
    question: str


@app.get("/")
def root():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok", "demo_mode": DEMO_MODE,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "gemini_model": GEMINI_MODEL,
        "restored_documents": len(DOCS)
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    doc_id = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{doc_id}.pdf"
    content = await file.read()
    if not content:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(content) > int(os.getenv("MAX_UPLOAD_MB", "30")) * 1024 * 1024:
        raise HTTPException(413, "PDF is too large. Maximum upload size is 30 MB.")
    path.write_bytes(content)
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read this PDF: {exc}") from exc
    if not any(clean_text(p) for p in pages):
        path.unlink(missing_ok=True)
        raise HTTPException(400, "No extractable text was found in this PDF.")
    chunks = chunk_pages(pages)
    if not chunks:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "No usable text chunks could be created from this PDF.")
    doc = {
        "id": doc_id, "filename": Path(filename).name, "pages": len(pages),
        "chunks": chunks, "index": None, "path": str(path),
        "created": datetime.now().isoformat()
    }
    DOCS[doc_id] = doc
    try:
        if not DEMO_MODE:
            build_index(doc)
        save_documents()
    except Exception as exc:
        DOCS.pop(doc_id, None)
        path.unlink(missing_ok=True)
        print(f"Document indexing error ({type(exc).__name__}): {exc}")
        raise HTTPException(status_code=500, detail="Could not build the document search index. Check server memory and model dependencies.") from exc
    return {"document_id": doc_id, "filename": doc["filename"], "pages": len(pages), "chunks": len(chunks), "status": "ready"}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    doc = DOCS.get(req.document_id)
    if not doc or not Path(doc["path"]).exists():
        raise HTTPException(404, "Document not found. Upload it again.")
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "Please enter a question.")
    if DEMO_MODE:
        raise HTTPException(503, "Demo mode is enabled. Set DEMO_MODE=false for real analysis.")
    try:
        evidence = retrieve(doc, q, 5)
        if not evidence:
            raise HTTPException(422, "No relevant evidence could be retrieved from this document.")
        answer = grounded_answer(q, evidence)
        claims = extract_claims(answer)
        verified = verify_claims(claims, evidence)
        score, risk = score_claims(verified)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Analysis error ({type(exc).__name__}): {exc}")
        raise HTTPException(status_code=500, detail="Analysis failed. Check server logs for the underlying error.") from exc
    result = {
        "answer": answer, "hallucination_score": score, "risk": risk,
        "claims": verified, "retrieved_evidence": evidence,
        "timestamp": datetime.now().isoformat(),
        "document": {"filename": doc["filename"], "pages": doc["pages"], "chunks": len(doc["chunks"])}
    }
    history = load_history()
    history.append({
        "id": uuid.uuid4().hex[:10], "document": doc["filename"], "question": q,
        "hallucination_score": score, "risk": risk, "claims": len(verified),
        "timestamp": result["timestamp"], "result": result
    })
    save_history(history)
    return result


@app.get("/api/history")
def history():
    return load_history()[::-1]


@app.get("/api/benchmark")
def benchmark():
    return {
        "validation_samples": 200, "test_samples": 400, "agreement": 0.98,
        "baseline": {"accuracy": 0.8125, "precision": 0.9085, "recall": 0.695, "f1": 0.7875, "roc_auc": 0.8086},
        "proposed": {"accuracy": 0.8025, "precision": 0.8805, "recall": 0.700, "f1": 0.7799, "roc_auc": 0.8072}
    }


@app.get("/api/document/{document_id}")
def document(document_id: str):
    doc = DOCS.get(document_id)
    if not doc or not Path(doc["path"]).exists():
        raise HTTPException(404, "Document not found")
    return {"document_id": doc["id"], "filename": doc["filename"], "pages": doc["pages"], "chunks": len(doc["chunks"])}
