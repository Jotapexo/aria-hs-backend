"""
ARIA v4 — Backend FastAPI con RAG real
=======================================
Stack 100% Google + RAG con PDFs reales:
  - Gemini 2.0 Flash → cerebro IA (gratis)
  - Google Cloud TTS → voz neural femenina (gratis)
  - ChromaDB + sentence-transformers → RAG con tus PDFs
  - PDFs en carpeta docs/ del repositorio

Al arrancar el servidor:
  1. Lee todos los PDFs de la carpeta docs/
  2. Los fragmenta y genera embeddings locales
  3. Los indexa en ChromaDB en memoria
  4. Cada consulta busca los fragmentos más relevantes
  5. Gemini responde con ese contexto real
"""

import os
import base64
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai

# ── RAG ───────────────────────────────────────────────────────────────
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

# ── Config ────────────────────────────────────────────────────────────
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL        = "gemini-2.0-flash"
GOOGLE_TTS_VOICE    = os.getenv("GOOGLE_TTS_VOICE", "es-US-Neural2-A")
GOOGLE_TTS_LANGUAGE = os.getenv("GOOGLE_TTS_LANGUAGE", "es-US")
DOCS_DIR            = Path("docs")          # carpeta con tus PDFs en el repo
CHUNK_SIZE          = 800                   # caracteres por fragmento
CHUNK_OVERLAP       = 100

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="ARIA H&S API", version="4.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── ChromaDB en memoria ───────────────────────────────────────────────
chroma_client    = chromadb.Client()
embed_fn         = SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
collection       = chroma_client.get_or_create_collection(name="aria_hs", embedding_function=embed_fn)
docs_indexed     = []

# ── Indexar PDFs al arrancar ──────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 50]

def index_pdfs():
    global docs_indexed
    if not DOCS_DIR.exists():
        print(f"[ARIA] Carpeta '{DOCS_DIR}' no encontrada — solo conocimiento base disponible.")
        return

    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[ARIA] No hay PDFs en '{DOCS_DIR}'.")
        return

    print(f"[ARIA] Indexando {len(pdfs)} PDFs...")
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
            print(f"[ARIA]   ✓ {pdf_path.name} — {len(reader.pages)} páginas, {len(chunks)} fragmentos")
        except Exception as e:
            print(f"[ARIA]   ✗ Error en {pdf_path.name}: {e}")

    if all_chunks:
        # Insertar en lotes de 100
        for i in range(0, len(all_chunks), 100):
            collection.add(
                documents=all_ids[i:i+100],
                ids=all_ids[i:i+100],
                metadatas=all_metas[i:i+100],
            )
        # Insertar documentos reales (ChromaDB los vectoriza con embed_fn)
        collection.upsert(
            documents=all_chunks,
            ids=all_ids,
            metadatas=all_metas,
        )
        print(f"[ARIA] ✓ {len(all_chunks)} fragmentos indexados en ChromaDB.")

# Indexar al arrancar
index_pdfs()

# ── System Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres ARIA (Asistente de Riesgos Industriales Autónomo), asistente virtual de
Seguridad y Salud Ocupacional para entornos industriales ecuatorianos, específicamente Holcim Ecuador.

PERSONALIDAD: Profesional, empática, clara. Voz femenina calmada pero autoritativa ante peligros.

NORMATIVA BASE:
- Decreto Ejecutivo 2393: Reglamento SST Ecuador
- Resolución CD 513 IESS: Seguro de Riesgos del Trabajo
- NTE INEN ISO 45001:2018: Sistema de Gestión SSO
- NTE INEN 439: Señalización industrial
- OSHA 29 CFR 1910: Industria general

CONTEXTO HOLCIM ECUADOR (Planta Latacunga):
- Procesos: cantera → molienda → horno rotatorio (1450°C) → silo → ensacado
- EPP básico: casco, lentes, chaleco naranja, botas punta acero, tapones auditivos
- Señales: Rojo=peligro, Amarillo=precaución, Verde=evacuación, Azul=obligatorio

REGLAS DE RESPUESTA:
1. Español siempre. Máximo 3-4 oraciones por turno (se convierte a voz).
2. Prioriza el CONTEXTO DE DOCUMENTOS si está disponible — es información oficial de Holcim.
3. Cita la fuente cuando uses información de un documento (ej: "según el Manual del Contratista...").
4. Cada 3-4 respuestas, haz UNA pregunta de verificación.
5. Ante peligro inmediato: tono URGENTE y claro."""

TOUR_STEPS = [
    "Bienvenido al tour de inducción SSO de Holcim Ecuador, planta Latacunga. Iniciamos en la ZONA DE ACCESO. Toda persona debe registrarse, portar identificación visible y verificar su EPP básico: casco, lentes, chaleco naranja y botas punta de acero. Obligación del empleador según el artículo 11 del Decreto 2393. ¿Tienes todo tu equipo?",
    "Continuamos en el ÁREA DE MOLIENDA. El ruido supera los 95 decibeles, sobre el límite de 85. La protección auditiva NRR 25 es obligatoria en todo momento. Retirarla aunque sea un instante puede causar daño auditivo permanente. ¿Usas correctamente tus protectores auditivos?",
    "Llegamos al HORNO ROTATORIO, la zona de mayor riesgo: temperaturas de hasta 1450 grados Celsius. Todo mantenimiento requiere procedimiento LOTO completo y permiso de trabajo en caliente. ¿Conoces los 7 pasos del bloqueo y etiquetado?",
    "Visitamos los SILOS DE CEMENTO, espacios confinados con riesgo de asfixia. Nadie ingresa sin permiso firmado, oxígeno entre 19.5 y 23.5%, detector de gases y vigía externo permanente. ¿Sabes usar el detector de gases?",
    "Finalizamos en ENSACADO Y DESPACHO. Tráfico intenso de montacargas: respeta los pasillos peatonales amarillos. En emergencia: activa alarma, evacúa por rutas verdes, repórtate en tu punto de encuentro. ¿Conoces tu punto de encuentro asignado?"
]

# ── RAG — buscar fragmentos relevantes ───────────────────────────────
def get_rag_context(question: str, n_results: int = 4) -> str:
    if collection.count() == 0:
        return ""
    try:
        results = collection.query(query_texts=[question], n_results=min(n_results, collection.count()))
        if not results["documents"] or not results["documents"][0]:
            return ""
        sources = [m.get("source","?") for m in results["metadatas"][0]]
        fragments = []
        for doc, src in zip(results["documents"][0], sources):
            fragments.append(f"[Fuente: {src}]\n{doc}")
        return "\n\n---\n\n".join(fragments)
    except Exception as e:
        print(f"[ARIA] RAG error: {e}")
        return ""

# ── Modelos Pydantic ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history: list = []
    use_rag: bool = True

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None

# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ARIA v4.3 online", "model": GEMINI_MODEL}

@app.get("/api/status")
async def status():
    return {
        "online":       True,
        "gemini":       bool(GEMINI_API_KEY),
        "model":        GEMINI_MODEL,
        "google_tts":   bool(GEMINI_API_KEY),
        "voice":        GOOGLE_TTS_VOICE,
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
        system += f"\n\nCONTEXTO DE DOCUMENTOS OFICIALES (usa esto como prioridad):\n{context}"

    history_gemini = []
    for h in req.history[-12:]:
        role = "user" if h["role"] == "user" else "model"
        history_gemini.append({"role": role, "parts": [h["content"]]})

    try:
        model        = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=system)
        chat_session = model.start_chat(history=history_gemini)
        response     = chat_session.send_message(req.question)
        return {
            "answer":       response.text,
            "rag_used":     bool(context),
            "sources":      list({m.get("source","?") for m in collection.query(query_texts=[req.question], n_results=2)["metadatas"][0]}) if context else [],
            "model":        GEMINI_MODEL,
        }
    except Exception as e:
        raise HTTPException(500, f"Gemini error: {str(e)}")

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY no configurada en Railway")

    voice     = req.voice or GOOGLE_TTS_VOICE
    lang_code = "-".join(voice.split("-")[:2])
    payload   = {
        "input": {"text": req.text},
        "voice": {"languageCode": lang_code, "name": voice},
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate":  0.95,
            "pitch":         2.0,
            "volumeGainDb":  1.0,
        },
    }
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GEMINI_API_KEY}"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Google TTS error: {r.text}")

    return {
        "audio_base64": r.json()["audioContent"],
        "format":       "mp3",
        "voice":        voice,
    }
