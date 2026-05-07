# 🚀 Guide d'Inférence et Optimisation (NVIDIA Jetson)

Ce guide explique comment transformer le modèle entraîné (`.pt`) en un moteur de calcul haute performance (`.engine`) pour atteindre **30 FPS** sur la NVIDIA Jetson Orin Nano.

---

## 1. Préparation de la Jetson

Avant de lancer l'inférence, il est nécessaire de pousser les performances de la carte au maximum :

```bash
# Activer le mode de puissance maximale
sudo nvpmodel -m 0

# Forcer les horloges du GPU au maximum
sudo jetson_clocks
```

---

## 2. Conversion du modèle (en deux étapes)

La conversion se fait en **deux étapes séparées** : d'abord sur votre machine de développement, puis sur la Jetson. Cette approche évite d'installer Ultralytics sur la Jetson et supprime les conflits de versions fréquents.

### Étape 2a — Sur votre machine : `.pt` → `.onnx`

```bash
yolo export model=modeles/best.pt format=onnx half=True
```

Un fichier `best.onnx` sera généré dans `modeles/`. Copiez-le ensuite sur la Jetson (via `scp`, clé USB, etc.).

### Étape 2b — Sur la Jetson : `.onnx` → `.engine`

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=modeles/best.onnx \
  --saveEngine=modeles/best.engine \
  --fp16
```

| Paramètre | Rôle |
|-----------|------|
| `--onnx` | Chemin vers le fichier ONNX transféré |
| `--saveEngine` | Chemin de sortie du moteur TensorRT |
| `--fp16` | Active la précision FP16 pour accélérer l'inférence GPU |

> ⚠️ **Ce fichier `.engine` est spécifique au matériel.** Il ne fonctionnera pas sur une autre machine et ne doit pas être partagé dans le dépôt.

---

## 3. Lancement de l'inférence temps réel

Pour obtenir la fluidité maximale (30 FPS), le script `live_jetson.py` exploite la **vectorisation NumPy** pour le post-traitement des boîtes de détection, éliminant ainsi les goulots d'étranglement CPU.

```bash
python3 src/live_jetson.py --engine modeles/best.engine --source 0
```

| Argument | Description |
|----------|-------------|
| `--engine` | Chemin vers le moteur TensorRT généré |
| `--source 0` | Caméra embarquée de la Jetson (ou index du flux vidéo) |

---

## 💡 Pourquoi cette optimisation ?

| Configuration | FPS | Latence |
|---------------|-----|---------|
| Sans TensorRT (CPU/GPU brut) | ~8 FPS | ~120 ms |
| **TensorRT FP16 + NumPy(vectorisation)** | **30 FPS** | **~33 ms** |

Une latence inférieure à **35 ms** est indispensable pour qu'un drone en mouvement puisse détecter et localiser les défauts en temps réel sans flou de mouvement ni décalage d'affichage.
