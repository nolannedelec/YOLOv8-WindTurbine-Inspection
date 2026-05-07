# 🌬️ Détection de Défauts sur Pales d'Éoliennes par IA Embarquée

> Projet de M1 – ISEN Brest (2026)  
> **Nolan Nedelec** & **Cassandre Poyen** 

---

## 📋 Présentation

L'inspection des pales d'éoliennes est une opération **dangereuse et coûteuse**, traditionnellement réalisée par des techniciens en hauteur ou via des systèmes de cordistes. Ce projet propose une **alternative autonome et embarquée** : un modèle IA déployé sur drone, capable de détecter en temps réel les défauts structurels des pales.

### Classes de défauts détectées

| Classe | Description |
|--------|-------------|
| `Crack` | Fissures de surface |
| `Erosion` | Érosion du bord d'attaque |
| `Damage` | Dégâts structurels divers |

---

## ⚡ Performances

| Métrique | Valeur |
|----------|--------|
| Précision (mAP50) | **82,1 %** |
| Fluidité | **30 FPS** |
| Latence totale | **31 – 33 ms** |
| Seuil de confiance recommandé | **0.35 – 0.40** |
| Cible matérielle | NVIDIA Jetson Orin Nano |

---

## 🏗️ Architecture Technique

### Modèle

- **YOLOv8m** — sélectionné après comparaison avec YOLOv11 et YOLOv26
- Dataset : **1 886 images** réelles, nettoyées et équilibrées 

### Optimisations pour l'embarqué

- Conversion **TensorRT (FP16)** pour l'accélération GPU sur Jetson
- Vectorisation **NumPy** du post-traitement pour éliminer les goulots d'étranglement CPU

---

## 📁 Structure du dépôt

```
.
├── models/                  # Poids du modèle
│   ├── best.pt               # ⚠️ Voir section Releases (~50Mo)
│   └── README.md             # Instructions pour générer le .engine
│
├── src/
│   └── app.py                # Interface Gradio (test local / webcam)
│
├── data/
│   └── data_img/         # Images de test (exemples de défauts)
│
└── requirements.txt
```

---

## 🚀 Installation & Lancement

### 1. Cloner le dépôt

```bash
git clone https://github.com/nolannedelec/<nom-du-repo>.git
cd <nom-du-repo>
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

> Dépendances : `ultralytics`, `gradio`, `numpy`, `opencv-python`

### 3. Récupérer le modèle

Le fichier `best.pt` fait ~50 Mo et **ne peut pas être stocké directement dans le dépôt**. Il est disponible dans la section **[Releases](https://github.com/nolannedelec/YOLOv8-WindTurbine-Inspection/releases)** de ce dépôt.

Placez-le dans le dossier `models/` une fois téléchargé.

### 4. Lancer l'interface Gradio (test local)

```bash
python src/app.py
```

L'interface permet de :
- Choisir la source d'entrée (image uploadée ou flux webcam)
- Ajuster dynamiquement le seuil de confiance
- Visualiser l'image annotée avec le bilan des défauts détectés

---


## 📦 requirements.txt

```
ultralytics
gradio
numpy
opencv-python
```
---

*Projet réalisé dans le cadre du Master 1 à l'**ISEN Brest** – 2026*
