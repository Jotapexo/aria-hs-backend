"""
ARIA v4 — Backend FastAPI (liviano, sin ChromaDB)
==================================================
Stack minimalista que cabe en Railway gratis:
  - Gemini 2.0 Flash → cerebro IA
  - RAG liviano: lee PDFs con pypdf, busca por palabras clave
  - Sin ChromaDB ni sentence-transformers (muy pesados)
  - Imagen final ~300 MB
"""

import os
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from pypdf import PdfReader

# ── Config ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"
DOCS_DIR       = Path("docs")
CHUNK_SIZE     = 600

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="ARIA H&S API", version="4.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Cargar PDFs en memoria al arrancar ────────────────────────────────
knowledge_base = []   # lista de {"source": str, "text": str}
docs_indexed   = []

def load_pdfs():
    if not DOCS_DIR.exists():
        print(f"[ARIA] Sin carpeta docs/ — solo conocimiento base.")
        return
    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print("[ARIA] No hay PDFs en docs/")
        return
    for pdf_path in pdfs:
        try:
            reader = PdfReader(str(pdf_path))
            text   = "\n".join(p.extract_text() or "" for p in reader.pages)
            # Dividir en fragmentos de CHUNK_SIZE caracteres
            for i in range(0, len(text), CHUNK_SIZE):
                chunk = text[i:i + CHUNK_SIZE].strip()
                if len(chunk) > 80:
                    knowledge_base.append({"source": pdf_path.name, "text": chunk})
            docs_indexed.append({"name": pdf_path.name, "pages": len(reader.pages)})
            print(f"[ARIA] ✓ {pdf_path.name} — {len(reader.pages)} páginas")
        except Exception as e:
            print(f"[ARIA] ✗ {pdf_path.name}: {e}")
    print(f"[ARIA] ✓ {len(knowledge_base)} fragmentos listos.")

load_pdfs()

# ── RAG liviano por palabras clave ────────────────────────────────────
def get_rag_context(question: str, top_k: int = 4) -> str:
    if not knowledge_base:
        return ""
    # Tokenizar pregunta
    words = set(re.findall(r'\w+', question.lower()))
    stop  = {"el","la","los","las","de","del","en","que","es","un","una","y","o","a","con","por","para","se","su","al"}
    words -= stop

    # Puntuar cada fragmento
    scored = []
    for item in knowledge_base:
        text_lower = item["text"].lower()
        score = sum(1 for w in words if w in text_lower)
        if score > 0:
            scored.append((score, item))

    # Ordenar por relevancia y tomar los top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top:
        return ""

    fragments = [f"[Fuente: {item['source']}]\n{item['text']}" for _, item in top]
    return "\n\n---\n\n".join(fragments)

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
2. Si hay CONTEXTO DE DOCUMENTOS úsalo como prioridad y cita la fuente.
3. Cada 3-4 respuestas haz UNA pregunta de verificación.
4. Ante peligro inmediato: tono URGENTE."""

TOUR_STEPS = [
    "Bienvenido al tour de inducción SSO de Holcim Ecuador, planta Latacunga. Iniciamos en la ZONA DE ACCESO. Toda persona debe registrarse, portar identificación y verificar su EPP básico: casco, lentes, chaleco naranja y botas punta de acero. Obligación del empleador según el artículo 11 del Decreto 2393. ¿Tienes todo tu equipo?",
    "Continuamos en el ÁREA DE MOLIENDA. El ruido supera los 95 decibeles, sobre el límite de 85. La protección auditiva NRR 25 es obligatoria. Retirarla aunque sea un instante puede causar daño auditivo permanente. ¿Usas correctamente tus protectores auditivos?",
    "Llegamos al HORNO ROTATORIO, la zona de mayor riesgo: hasta 1450 grados Celsius. Todo mantenimiento requiere procedimiento LOTO completo y permiso de trabajo en caliente. ¿Conoces los 7 pasos del bloqueo y etiquetado?",
    "Visitamos los SILOS DE CEMENTO, espacios confinados con riesgo de asfixia. Nadie ingresa sin permiso firmado, oxígeno entre 19.5 y 23.5%, detector de gases y vigía externo. ¿Sabes usar el detector de gases?",
    "Finalizamos en ENSACADO Y DESPACHO. Tráfico intenso de montacargas: respeta los pasillos amarillos. En emergencia activa alarma, evacúa por rutas verdes y repórtate en tu punto de encuentro. ¿Conoces tu punto de encuentro?"
]

# ── Modelos ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history: list = []
    use_rag: bool = True

# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ARIA v4.5 online", "model": GEMINI_MODEL}

@app.get("/api/status")
async def status():
    return {
        "online":       True,
        "gemini":       bool(GEMINI_API_KEY),
        "google_tts":   False,
        "model":        GEMINI_MODEL,
        "voice":        "Web Speech API (browser)",
        "docs_indexed": docs_indexed,
        "total_chunks": len(knowledge_base),
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
        system += f"\n\nCONTEXTO DE DOCUMENTOS OFICIALES (prioridad máxima):\n{context}"

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
