import os
import gradio as gr
import joblib
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = joblib.load("models/best_model.pkl")

CSS = """
/* ── Page & global ── */
body, .gradio-container, .main, .wrap {
    background: #111827 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #f1f5f9 !important;
}

/* ── Kill any stray white/light backgrounds Gradio injects ── */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gap,
.gradio-container fieldset,
.gradio-container .gradio-radio,
.gradio-container .gradio-slider,
.gradio-container .padded {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Header block ── */
.header-block {
    background: #1f2937 !important;
    border-radius: 16px !important;
    padding: 28px 36px 24px !important;
    margin-bottom: 14px !important;
    border: 1px solid #374151 !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5) !important;
}
.header-block > *,
.header-block .prose,
.header-block .markdown,
.header-block > div,
.header-block > div > div {
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
.header-block h1 {
    color: #f1f5f9 !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    margin: 0 0 8px !important;
    letter-spacing: -0.3px !important;
}
.header-block p {
    color: #94a3b8 !important;
    font-size: 0.92rem !important;
    margin: 0 !important;
}

/* ── Cards ── */
.card {
    background: #1f2937 !important;
    border-radius: 14px !important;
    padding: 24px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
    border: 1px solid #374151 !important;
}
.card h3 {
    color: #06b6d4 !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    margin: 0 0 18px !important;
    padding-bottom: 10px !important;
    border-bottom: 1px solid #374151 !important;
}

/* ── All label text ── */
.gradio-container label > span,
.gradio-container label > span:first-child,
.gradio-container .label-wrap span,
.gradio-container fieldset > legend span,
.gradio-container .block > label > span {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    background: transparent !important;
}

/* ── Radio option text ── */
.gradio-container .gradio-radio label span,
.gradio-container input[type=radio] ~ span {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    background: transparent !important;
}

/* ── Slider + radio: teal accent ── */
input[type=range] { accent-color: #06b6d4 !important; }
input[type=radio]  { accent-color: #06b6d4 !important; }

/* ── Slider number input ── */
.gradio-container input[type=number] {
    color: #06b6d4 !important;
    font-weight: 700 !important;
    background: #111827 !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
}

/* ── Primary button ── */
button.primary {
    background: #06b6d4 !important;
    border: none !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 14px 0 !important;
    width: 100% !important;
    margin-top: 16px !important;
    box-shadow: 0 0 20px rgba(6,182,212,0.3) !important;
    letter-spacing: 0.03em !important;
    transition: box-shadow 0.18s, opacity 0.18s !important;
    cursor: pointer !important;
}
button.primary:hover {
    box-shadow: 0 0 28px rgba(6,182,212,0.5) !important;
    opacity: 0.92 !important;
}

/* ── Output result card ── */
.output-card {
    background: #1f2937 !important;
    border-radius: 14px !important;
    padding: 16px 24px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
    border: 1px solid #374151 !important;
    margin-top: 8px !important;
    min-height: 80px !important;
}

/* ── Disclaimer ── */
.disclaimer p {
    color: #4b5563 !important;
    font-size: 0.78rem !important;
    text-align: center !important;
    margin: 0 !important;
}

/* ── Hide Gradio footer ── */
footer, .footer-wrap, .svelte-footer { display: none !important; }
"""

SEX_LABEL   = {0: "Weiblich", 1: "Männlich"}
CP_LABEL    = {0: "Typische Angina", 1: "Atypische Angina", 2: "Nicht-anginöser Schmerz", 3: "Asymptomatisch"}
FBS_LABEL   = {0: "Nein", 1: "Ja"}
RESTECG_LABEL = {0: "Normal", 1: "ST-T Anomalie", 2: "Linksventrikuläre Hypertrophie"}
EXANG_LABEL = {0: "Nein", 1: "Ja"}
SLOPE_LABEL = {0: "Aufsteigend", 1: "Flach", 2: "Absteigend"}
THAL_LABEL  = {1: "Normal", 2: "Fester Defekt", 3: "Reversibler Defekt"}


def generate_explanation(age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal,
                         risk_prob, risk_label):
    prompt = f"""Du bist ein verständlicher medizinischer KI-Assistent. \
Erkläre einem Patienten auf Deutsch, was das Modell bei seinen Werten festgestellt hat. \
Stelle keine Diagnose. Weise darauf hin, dass es sich um eine KI-gestützte Einschätzung handelt \
und ein Arzt konsultiert werden sollte.

Patientenwerte:
- Alter: {age} Jahre
- Geschlecht: {SEX_LABEL[sex]}
- Brustschmerztyp: {CP_LABEL[cp]}
- Blutdruck in Ruhe: {trestbps} mmHg
- Cholesterin: {chol} mg/dl
- Nüchternblutzucker > 120 mg/dl: {FBS_LABEL[fbs]}
- Ruhe-EKG: {RESTECG_LABEL[restecg]}
- Max. Herzfrequenz: {thalach}
- Belastungsangina: {EXANG_LABEL[exang]}
- ST-Senkung (oldpeak): {oldpeak}
- ST-Steigung: {SLOPE_LABEL[slope]}
- Gefärbte Gefässe (ca): {ca}
- Thalassämie: {THAL_LABEL[thal]}

KI-Ergebnis: {risk_label} (Risikowahrscheinlichkeit: {risk_prob:.1f}%)

Antworte ausschliesslich mit 3–5 Stichpunkten im Format:
• Stichpunkt 1
• Stichpunkt 2
• Stichpunkt 3
Kein Fliesstext, kein Titel, nur die Bullet-Points."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)
    heart_risk_probability = proba[0][1] / sum(proba[0]) * 100

    if heart_risk_probability < 40:
        accent = "#34d399"
        glow   = "rgba(52,211,153,0.18)"
        icon   = "✓"
        label  = "Geringes Risiko"
    elif heart_risk_probability <= 60:
        accent = "#f59e0b"
        glow   = "rgba(245,158,11,0.18)"
        icon   = "△"
        label  = "Grenzfall / unsichere Einschätzung"
    else:
        accent = "#f87171"
        glow   = "rgba(248,113,113,0.18)"
        icon   = "⚠"
        label  = "Herzerkrankung wahrscheinlich"

    model_label = "Herzerkrankung" if prediction == 1 else "Kein Risiko"

    html = f"""
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #1f2937;
        border-radius: 14px;
        border: 1px solid #374151;
        border-left: 4px solid {accent};
        box-shadow: 0 0 24px {glow}, 0 4px 20px rgba(0,0,0,0.4);
        padding: 24px 28px;
        margin-top: 4px;
    ">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <span style="
                width: 36px; height: 36px; border-radius: 50%;
                background: {glow};
                border: 1.5px solid {accent};
                display: flex; align-items: center; justify-content: center;
                font-size: 1rem; color: {accent}; flex-shrink: 0;
            ">{icon}</span>
            <div>
                <div style="color: #e2e8f0; font-size: 1rem; font-weight: 600;">{label}</div>
            </div>
        </div>

        <div style="display: flex; align-items: flex-end; gap: 6px; margin-bottom: 14px;">
            <span style="font-size: 3rem; font-weight: 800; color: {accent}; line-height: 1;">{heart_risk_probability:.1f}</span>
            <span style="font-size: 1.1rem; font-weight: 600; color: #94a3b8; padding-bottom: 6px;">%</span>
            <span style="font-size: 0.78rem; color: #64748b; padding-bottom: 8px; margin-left: 4px;">Risikowahrscheinlichkeit</span>
        </div>

        <div style="background: #374151; border-radius: 99px; height: 6px; width: 100%; overflow: hidden;">
            <div style="
                background: {accent};
                width: {heart_risk_probability:.1f}%;
                height: 6px;
                border-radius: 99px;
                box-shadow: 0 0 8px {accent};
            "></div>
        </div>
    </div>
    """

    explanation_text = generate_explanation(
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal,
        heart_risk_probability, label,
    )

    explanation_html = f"""
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #1f2937;
        border-radius: 14px;
        border: 1px solid #374151;
        border-left: 4px solid #06b6d4;
        padding: 22px 28px;
        margin-top: 10px;
    ">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
            <span style="font-size: 1.2rem;">🧠</span>
            <span style="color: #06b6d4; font-size: 0.72rem; font-weight: 700;
                         letter-spacing: 0.14em; text-transform: uppercase;">KI-Erklärung</span>
        </div>
        <p style="color: #cbd5e1; font-size: 1.02rem; line-height: 1.9; margin: 0 0 14px; white-space: pre-line;">
            {explanation_text}
        </p>
        <p style="color: #4b5563; font-size: 0.75rem; margin: 0; border-top: 1px solid #374151;
                  padding-top: 10px;">
            ⓘ Dies ist eine KI-gestützte Einschätzung und kein Ersatz für eine ärztliche Diagnose.
        </p>
    </div>
    """

    return html, explanation_html

with gr.Blocks(title="Heart Risk Assistant", css=CSS) as demo:

    with gr.Column(elem_classes="header-block"):
        gr.Markdown("# ❤️ Heart Risk Assistant")
        gr.Markdown("Gib die klinischen Patientenwerte ein und klicke auf **Risiko berechnen**.")

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="card"):
            gr.Markdown("### Patientendaten")
            age       = gr.Slider(1, 100, value=50, step=1, label="Alter (age)")
            sex       = gr.Radio([0, 1], value=0, label="Geschlecht (0 = Frau, 1 = Mann)")
            cp        = gr.Radio([0, 1, 2, 3], value=0,
                                 label="Brustschmerztyp (0=Typisch, 1=Atypisch, 2=Nicht-anginös, 3=Asymptomatisch)")
            trestbps  = gr.Slider(80, 200, value=120, step=1, label="Blutdruck in Ruhe (mmHg)")
            chol      = gr.Slider(100, 600, value=200, step=1, label="Cholesterin (mg/dl)")
            fbs       = gr.Radio([0, 1], value=0, label="Nüchternblutzucker > 120 mg/dl (0=Nein, 1=Ja)")
            restecg   = gr.Radio([0, 1, 2], value=0,
                                 label="Ruhe-EKG (0=Normal, 1=ST-T Anomalie, 2=Linkshypertrophie)")

        with gr.Column(elem_classes="card"):
            gr.Markdown("### Belastungsdaten")
            thalach   = gr.Slider(60, 220, value=150, step=1, label="Max. Herzfrequenz (thalach)")
            exang     = gr.Radio([0, 1], value=0, label="Belastungsangina (0=Nein, 1=Ja)")
            oldpeak   = gr.Slider(0.0, 6.2, value=0.0, step=0.1, label="ST-Senkung unter Belastung (oldpeak)")
            slope     = gr.Radio([0, 1, 2], value=1,
                                 label="ST-Steigung (0=Aufsteigend, 1=Flach, 2=Absteigend)")
            ca        = gr.Radio([0, 1, 2, 3], value=0, label="Anzahl gefärbter Gefäße (ca: 0–3)")
            thal      = gr.Radio([1, 2, 3], value=2,
                                 label="Thalassämie (1=Normal, 2=Fester Defekt, 3=Reversibler Defekt)")
            gr.Markdown("")
            btn = gr.Button("Risiko berechnen", variant="primary")

    with gr.Column(elem_classes="output-card"):
        output      = gr.HTML(label="Ergebnis")
        explanation = gr.HTML()

    with gr.Column(elem_classes="disclaimer"):
        gr.Markdown(
            "<hr><p>&#9432; Dieses Tool dient ausschliesslich Bildungszwecken und ersetzt keine medizinische Diagnose.</p>"
        )

    btn.click(
        fn=predict,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
        outputs=[output, explanation],
    )

demo.launch()
