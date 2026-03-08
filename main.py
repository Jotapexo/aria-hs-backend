"""
ARIA v4 — Backend FastAPI
==========================
Stack:
  - Gemini 2.0 Flash (Google) → cerebro IA — GRATIS
  - Web Speech API (browser)  → voz TTS sin costo ni configuración
  - ChromaDB + sentence-transformers → RAG con PDFs reales
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai

# RAG
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

# ── Config ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"
DOCS_DIR       = Path("docs")
CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 100

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="ARIA H&S API", version="4.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── ChromaDB ──────────────────────────────────────────────────────────
chroma_client = chromadb.Client()
embed_fn      = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
collection  = chroma_client.get_or_create_collection(name="aria_hs", embedding_function=embed_fn)
docs_indexed = []

def chunk_text(text: str) -> list:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 50]

def index_pdfs():
    global docs_indexed
    if not DOCS_DIR.exists():
        print(f"[ARIA] Carpeta '{DOCS_DIR}' no encontrada.")
        return
    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print("[ARIA] No hay PDFs en docs/")
        return
    all_chunks, all_ids, all_metas = [], [], []
    chunk_id = 0
    for pdf_path in pdfs:
        try:
            reader = PdfReader(str(pdf_path))
            text   = "\n".join(p.extract_text() or "" for p in reader.pages)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{pdf_path.stem}_{chunk_id}")
                all_metas.append({"source": pdf_path.name, "chunk": i})
                chunk_id += 1
            docs_indexed.append({"name": pdf_path.name, "pages": len(reader.pages), "chunks": len(chunks)})
            print(f"[ARIA] ✓ {pdf_path.name} — {len(reader.pages)} págs, {len(chunks)} fragmentos")
        except Exception as e:
            print(f"[ARIA] ✗ Error en {pdf_path.name}: {e}")
    if all_chunks:
        collection.upsert(documents=all_chunks, ids=all_ids, metadatas=all_metas)
        print(f"[ARIA] ✓ {len(all_chunks)} fragmentos indexados.")

index_pdfs()

# ── System Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres ARIA (Asistente de Riesgos Industriales Autónomo), asistente virtual de
Seguridad y Salud Ocupacional para Holcim Ecuador, planta Latacunga.

PERSONALIDAD: Profesional, empática, clara. Autoritativa ante situaciones de peligro.

NORMATIVA BASE:
- Decreto Ejecutivo 2393: Reglamento SST Ecuador
- Resolución CD 513 IESS: Seguro de Riesgos del Trabajo
- NTE INEN ISO 45001:2018: Sistema de Gestión SSO
- NTE INEN 439: Señalización industrial
- OSHA 29 CFR 1910

CONTEXTO HOLCIM ECUADOR:
- Procesos: cantera → molienda → horno rotatorio (1450°C) → silo → ensacado
- EPP básico: casco, lentes, chaleco naranja, botas punta acero, tapones auditivos
- Señales: Rojo=peligro, Amarillo=precaución, Verde=evacuación, Azul=obligatorio

REGLAS:
1. Español siempre. Máximo 3-4 oraciones por turno.
2. Si hay CONTEXTO DE DOCUMENTOS, úsalo como prioridad y cita la fuente.
3. Cada 3-4 respuestas haz UNA pregunta de verificación.
4. Ante peligro inmediato: tono URGENTE."""

TOUR_STEPS = [
    "Bienvenido al tour de inducción SSO de Holcim Ecuador, planta Latacunga. Iniciamos en la ZONA DE ACCESO. Toda persona debe registrarse, portar identificación y verificar su EPP básico: casco, lentes, chaleco naranja y botas punta de acero. Obligación del empleador según el artículo 11 del Decreto 2393. ¿Tienes todo tu equipo?",
    "Continuamos en el ÁREA DE MOLIENDA. El ruido supera los 95 decibeles, sobre el límite de 85. La protección auditiva NRR 25 es obligatoria. Retirarla aunque sea un instante puede causar daño auditivo permanente. ¿Usas correctamente tus protectores auditivos?",
    "Llegamos al HORNO ROTATORIO, la zona de mayor riesgo: hasta 1450 grados Celsius. Todo mantenimiento requiere procedimiento LOTO completo y permiso de trabajo en caliente. ¿Conoces los 7 pasos del bloqueo y etiquetado?",
    "Visitamos los SILOS DE CEMENTO, espacios confinados con riesgo de asfixia. Nadie ingresa sin permiso firmado, oxígeno entre 19.5 y 23.5%, detector de gases y vigía externo. ¿Sabes usar el detector de gases?",
    "Finalizamos en ENSACADO Y DESPACHO. Tráfico intenso de montacargas: respeta los pasillos amarillos. En emergencia: activa alarma, evacúa por rutas verdes, repórtate en tu punto de encuentro. ¿Conoces tu punto de encuentro?"
]

# ── RAG ───────────────────────────────────────────────────────────────
def get_rag_context(question: str) -> str:
    if collection.count() == 0:
        return ""
    try:
        n = min(4, collection.count())
        results = collection.query(query_texts=[question], n_results=n)
        if not results["documents"] or not results["documents"][0]:
            return ""
        fragments = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            fragments.append(f"[Fuente: {meta.get('source','?')}]\n{doc}")
        return "\n\n---\n\n".join(fragments)
    except Exception as e:
        print(f"[ARIA] RAG error: {e}")
        return ""

# ── Modelos ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history: list = []
    use_rag: bool = True

# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ARIA v4.4 online", "model": GEMINI_MODEL}

@app.get("/api/status")
async def status():
    return {
        "online":       True,
        "gemini":       bool(GEMINI_API_KEY),
        "google_tts":   False,   # TTS lo maneja el browser con Web Speech API
        "model":        GEMINI_MODEL,
        "voice":        "Web Speech API (browser)",
        "docs_indexed": docs_indexed,
        "total_chunks": collection.count(),
        "tour_steps":   len(TOUR_STEPS),
    }

@app.get("/api/tour/{step}")
async def get_tour_step(step: int):
    if step < 0 or step >= len(TOUR_STEPS):
        raise HTTPException(404, "Paso no encontrado")
    return {"step": step, "total": len(TOUR_STEPS), "text": TOUR_STEPS[step]}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY no configurada en Railway")

    context = get_rag_context(req.question) if req.use_rag else ""
    system  = SYSTEM_PROMPT
    if context:
        system += f"\n\nCONTEXTO DE DOCUMENTOS OFICIALES:\n{context}"

    history_gemini = []
    for h in req.history[-12:]:
        role = "user" if h["role"] == "user" else "model"
        history_gemini.append({"role": role, "parts": [h["content"]]})

    try:
        model        = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=system)
        chat_session = model.start_chat(history=history_gemini)
        response     = chat_session.send_message(req.question)
        return {
            "answer":   response.text,
            "rag_used": bool(context),
            "model":    GEMINI_MODEL,
        }
    except Exception as e:
        raise HTTPException(500, f"Gemini error: {str(e)}")
