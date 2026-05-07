import sys
import cv2
import gradio as gr
from ultralytics import YOLO
from pathlib import Path

# 1. CHARGEMENT DU MODÈLE
MODEL_PATH = "models/best.pt"

# sys.exit(1) provoque un arrêt propre avec message lisible,
# plutôt qu'un crash obscur lors de la première inférence
if not Path(MODEL_PATH).exists():
    print(f"❌ ERREUR : Le fichier '{MODEL_PATH}' est introuvable.")
    print("   → Placez 'best.pt' dans le même dossier que app.py et relancez.")
    sys.exit(1)

model = YOLO(MODEL_PATH)
print(f"✅ Modèle '{MODEL_PATH}' chargé avec succès.")

# 2. FONCTION DE DÉTECTION
def detect_defects(image, conf_threshold):
    if image is None:
        return None, "⏳ En attente d'une image ou d'une capture webcam..."

    # Gradio fournit du RGB, mais OpenCV travaille en BGR — conversion obligatoire
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model(frame, conf=conf_threshold, imgsz=640, verbose=False)
    result = results[0]

    annotated = result.plot()
    # Re-conversion BGR→RGB pour que Gradio affiche correctement l'image annotée
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    counts = {"crack": 0, "damage": 0, "erosion": 0}
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label in counts:
                counts[label] += 1

    total = sum(counts.values())
    if total == 0:
        summary = "✅ Aucun défaut détecté."
    else:
        summary = (
            f"⚠️ {total} défaut(s) détecté(s) — "
            f"Fissures: {counts['crack']} | "
            f"Dommages: {counts['damage']} | "
            f"Érosions: {counts['erosion']}"
        )
    return annotated, summary

# 3. INTERFACE GRADIO
with gr.Blocks(title="Inspection IA Nordtank") as demo:
    gr.Markdown("# 🌬️ Inspection de Pales d'Éoliennes par IA")
    gr.Markdown(
        "Chargez une **image** (upload) ou utilisez la **webcam**, "
        "ajustez le seuil de confiance, puis lancez l'analyse."
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Entrée Image ou Webcam"
            )
            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                label="Seuil de Confiance",
                info="Valeur basse = plus de détections, valeur haute = plus strict"
            )
            btn = gr.Button("🔍 LANCER L'ANALYSE", variant="primary")

        with gr.Column():
            output_img = gr.Image(type="numpy", label="Résultat de l'Analyse")
            output_text = gr.Textbox(
                label="Bilan des défauts",
                interactive=False,
                value="⏳ En attente d'une analyse..."
            )

    btn.click(
        fn=detect_defects,
        inputs=[input_img, conf_slider],
        outputs=[output_img, output_text]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
