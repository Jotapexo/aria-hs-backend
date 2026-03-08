"""
ARIA v4 — Backend FastAPI
==========================
Stack 100% Google, 100% GRATIS:
  - Gemini 2.0 Flash → cerebro IA (1,500 req/día gratis)
  - Google Cloud TTS → voz neural femenina (1M chars/mes gratis)

Deploy: Railway.app
"""

import os
import base64
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"

# Google Cloud TTS usa la misma API key de Google
GOOGLE_TTS_VOICE    = os.getenv("GOOGLE_TTS_VOICE", "es-US-Neural2-A")  # femenina suave
GOOGLE_TTS_LANGUAGE = os.getenv("GOOGLE_TTS_LANGUAGE", "es-US")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="ARIA H&S API", version="4.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── System Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres ARIA (Asistente de Riesgos Industriales Autónomo), asistente virtual de
Seguridad y Salud Ocupacional para entornos industriales ecuatorianos, específicamente Holcim Ecuador.

PERSONALIDAD: Profesional, empática, clara. Voz femenina calmada pero autoritativa ante peligros.

NORMATIVA QUE APLICAS:
- Decreto Ejecutivo 2393: Reglamento SST Ecuador
- Resolución CD 513 IESS: Seguro de Riesgos del Trabajo
- NTE INEN ISO 45001:2018: Sistema de Gestión SSO
- NTE INEN 439: Señalización industrial
- OSHA 29 CFR 1910: Industria general
- Código del Trabajo Arts. 410-434

CONTEXTO HOLCIM ECUADOR (Planta Latacunga):
- Procesos: cantera → trituradora → molienda → horno rotatorio (1450°C) → silo → ensacado
- EPP básico: casco, lentes, chaleco naranja, botas punta acero, tapones auditivos
- Zonas críticas: horno rotatorio, silos, área explosivos, espacios confinados
- Señales: Rojo=peligro, Amarillo=precaución, Verde=evacuación, Azul=obligatorio

REGLAS DE RESPUESTA:
1. Español siempre. Máximo 3-4 oraciones por turno (se convierte a voz).
2. Cita la normativa cuando sea relevante.
3. Cada 3-4 respuestas, haz UNA pregunta de verificación al usuario.
4. Ante peligro inmediato: tono URGENTE y claro."""

# ── Base de conocimiento H&S ──────────────────────────────────────────
HS_KNOWLEDGE = {
    "epp": "EPP por zona: General: casco, lentes, chaleco naranja, botas punta acero. Ruido >85dB: protectores NRR>=25. Alturas >1.8m: arnés 5 puntos + doble línea de vida. Químicos: respirador N95, guantes nitrilo, traje Tyvek. Eléctrico: guantes dieléctricos, zapatos dieléctricos.",
    "loto": "LOTO 7 pasos: 1)Notificar 2)Identificar energías 3)Apagar 4)Aislar 5)Candado personal+etiqueta 6)Liberar energía residual 7)Verificar. NUNCA retires el candado de otra persona.",
    "alturas": "Alturas >1.8m: permiso específico, arnés 5 puntos ANSI Z359, doble línea de vida, anclaje 2268kg, casco con barboquejo. Inspección visual antes de cada uso. Decreto 2393 Art. 32.",
    "espacios_confinados": "Espacio confinado: permiso firmado, O2 19.5-23.5%, LEL<10%, detector gases 4-en-1, vigía externo constante, equipo de rescate listo antes de ingresar.",
    "emergencias": "Emergencia: 1)Activa alarma 2)Llama emergencias 3)Evacúa por rutas verdes 4)No uses ascensores 5)Punto de encuentro 6)No regreses sin autorización. Accidente: notificar IESS max 10 días hábiles (CD 513).",
    "quimicos": "Materiales peligrosos: revisa MSDS antes de manipular. Derrame: confina área, usa EPP de la MSDS, ventila, absorbe con material inerte. Notifica al supervisor inmediatamente.",
    "decreto_2393": "Decreto 2393 Art.11 empleador: proveer EPP sin costo, capacitar, mantener condiciones seguras, investigar accidentes. Art.13 trabajador: usar EPP, reportar condiciones inseguras.",
    "senalizacion": "INEN 439: Rojo=peligro/prohibición, Amarillo=precaución, Verde=evacuación/seguridad, Azul=obligación. Señales visibles a mínimo 10 metros.",
    "iso45001": "ISO 45001:2018: ciclo PHVA, participación trabajadores obligatoria. Adoptada como NTE INEN ISO 45001 en Ecuador.",
    "iess_cd513": "CD 513 IESS: grave notificar en 24h, leve en 10 días hábiles. Prestaciones: atención médica, subsidio 75% salario, pensión por invalidez.",
}

TOUR_STEPS = [
    "Bienvenido al tour de inducción SSO de Holcim Ecuador, planta Latacunga. Iniciamos en la ZONA DE ACCESO. Toda persona debe registrarse, portar identificación visible y verificar su EPP básico: casco, lentes, chaleco naranja y botas punta de acero. Obligación del empleador según el artículo 11 del Decreto 2393. ¿Tienes todo tu equipo?",
    "Continuamos en el ÁREA DE MOLIENDA. El ruido supera los 95 decibeles, sobre el límite de 85. La protección auditiva NRR 25 es obligatoria en todo momento. Retirarla aunque sea un instante puede causar daño auditivo permanente. ¿Usas correctamente tus protectores auditivos?",
    "Llegamos al HORNO ROTATORIO, la zona de mayor riesgo: temperaturas de hasta 1450 grados Celsius. Todo mantenimiento requiere procedimiento LOTO completo, permiso de trabajo en caliente, y nunca se trabaja solo. ¿Conoces los 7 pasos del bloqueo y etiquetado?",
    "Visitamos los SILOS DE CEMENTO, espacios confinados con riesgo de asfixia. Nadie ingresa sin permiso firmado, oxígeno entre 19.5 y 23.5%, detector de gases y vigía externo permanente. ¿Sabes usar el detector de gases?",
    "Finalizamos en ENSACADO Y DESPACHO. Tráfico intenso de montacargas: respeta los pasillos peatonales amarillos. En emergencia: activa alarma, evacúa por rutas verdes, repórtate en tu punto de encuentro. Accidentes se notifican al IESS según CD 513. ¿Conoces tu punto de encuentro?"
]

def get_rag_context(question: str) -> str:
    q = question.lower()
    keywords = {
        "epp":                ["epp","casco","lentes","chaleco","botas","guantes","protección"],
        "loto":               ["loto","bloqueo","etiquetado","candado","energía"],
        "alturas":            ["altura","arnés","caída","escalera","andamio"],
        "espacios_confinados":["confinado","silo","tanque","oxígeno","gases"],
        "emergencias":        ["emergencia","evacuación","alarma","accidente","incendio"],
        "quimicos":           ["químico","derrame","msds","ácido","sustancia"],
        "decreto_2393":       ["decreto","2393","reglamento","obligación","empleador"],
        "senalizacion":       ["señal","señalización","inen 439","color"],
        "iso45001":           ["iso","45001","gestión","phva"],
        "iess_cd513":         ["iess","cd 513","seguro","prestación","notificar"],
    }
    relevant = [HS_KNOWLEDGE[k] for k, kws in keywords.items() if any(kw in q for kw in kws)]
    return "\n\n".join(relevant[:3])

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
    return {"status": "ARIA v4.2 online", "model": GEMINI_MODEL}

@app.get("/api/status")
async def status():
    return {
        "online":     True,
        "gemini":     bool(GEMINI_API_KEY),
        "model":      GEMINI_MODEL,
        "google_tts": bool(GEMINI_API_KEY),  # misma key
        "voice":      GOOGLE_TTS_VOICE,
        "kb_topics":  len(HS_KNOWLEDGE),
        "tour_steps": len(TOUR_STEPS),
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
    system  = SYSTEM_PROMPT + (f"\n\nCONTEXTO NORMATIVO RELEVANTE:\n{context}" if context else "")

    history_gemini = []
    for h in req.history[-12:]:
        role = "user" if h["role"] == "user" else "model"
        history_gemini.append({"role": role, "parts": [h["content"]]})

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system,
        )
        chat_session = model.start_chat(history=history_gemini)
        response = chat_session.send_message(req.question)
        return {
            "answer":   response.text,
            "rag_used": bool(context),
            "model":    GEMINI_MODEL,
        }
    except Exception as e:
        raise HTTPException(500, f"Gemini error: {str(e)}")

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """
    Google Cloud TTS — voz neural femenina en español.
    Usa la misma GEMINI_API_KEY (ambas son Google APIs).
    1,000,000 caracteres/mes gratis con voces WaveNet/Neural2.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY no configurada en Railway")

    voice_name = req.voice or GOOGLE_TTS_VOICE

    # Determinar languageCode desde el nombre de la voz
    lang_code = "-".join(voice_name.split("-")[:2])  # "es-US-Neural2-A" → "es-US"

    payload = {
        "input": {"text": req.text},
        "voice": {
            "languageCode": lang_code,
            "name": voice_name,
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": 0.95,
            "pitch": 2.0,          # ligeramente más agudo → más suave/kawaii
            "volumeGainDb": 1.0,
        },
    }

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GEMINI_API_KEY}"

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Google TTS error: {r.text}")

    data = r.json()
    return {
        "audio_base64": data["audioContent"],
        "format":       "mp3",
        "voice":        voice_name,
    }
