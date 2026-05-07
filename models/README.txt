# 🌬️ Inspection de Pales d'Éoliennes par IA — Nordtank

## Structure du projet
INTERFACE/
├── app.py           # Application principale (interface Gradio + détection YOLO)
├── best.pt          # Modèle YOLO entraîné
├── requirements.txt # Dépendances Python
└── README.txt       # Ce fichier

## Prérequis
- Python 3.9 ou supérieur
- Le fichier `best.pt` (modèle entraîné) doit être placé à la racine du dossier INTERFACE/

## Installation et démarrage
Ouvrez un terminal dans le dossier INTERFACE/, puis :

### Option rapide
    pip install -r requirements.txt
    python app.py

### Option propre (environnement isolé)
    python -m venv gradio-env

    Windows :
        gradio-env\Scripts\activate

    macOS/Linux :
        source gradio-env/bin/activate

    pip install -r requirements.txt
    python app.py

L'interface s'ouvre automatiquement sur http://127.0.0.1:7860
Pour générer un lien public accessible depuis l'extérieur, modifier la dernière
ligne de app.py : demo.launch(share=True, inbrowser=True)

## Utilisation
1. Ajustez le seuil de confiance (0.4 par défaut)
2. Chargez une image (upload) ou activez la webcam
3. Cliquez sur 🔍 LANCER L'ANALYSE
4. Consultez l'image annotée et le bilan des défauts détectés

## Dépannage

❌ "Le fichier 'best.pt' est introuvable"
→ Placez best.pt dans le dossier INTERFACE/, au même niveau que app.py.

❌ L'interface ne s'ouvre pas automatiquement
→ Ouvrez manuellement http://127.0.0.1:7860 dans votre navigateur.

❌ Erreur liée à OpenCV sur Linux/serveur
→ Remplacez dans requirements.txt :
   opencv-python  →  opencv-python-headless