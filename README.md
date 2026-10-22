# 🤟 Sign-to-Speech — Reconnaissance de gestes en temps réel

Pipeline d'intelligence artificielle permettant de traduire en direct des gestes de la main via webcam en mots de texte.

Le projet combine la vision par ordinateur (**CNN via MediaPipe**) et l'apprentissage profond séquentiel (**RNN via LSTM / TensorFlow**) pour reconnaître des dynamiques de mouvements dans le temps.

---

## 📌 Architecture du Pipeline

1. **Extraction spatiale (CNN)** : 
   - Utilisation du modèle `HandLandmarker` de **Google MediaPipe**.
   - Détection de la main et extraction en 3D de **21 points clés** (*landmarks* : $x, y, z$), soit **63 coordonnées** par image.
2. **Traitement temporel (RNN / LSTM)** : 
   - Découpage en séquences temporelles de **30 frames consécutives** (matrice de taille `30 × 63`).
   - Réseau de neurones récurrents (**LSTM à deux couches**) sous **TensorFlow / Keras** pour capturer la dynamique et la vitesse des mouvements.
3. **Inférence & Prédiction en direct** : 
   - Analyse d'un buffer glissant sur flux vidéo webcam et classification avec niveau de confiance via fonction `Softmax`.

---

## 🛠️ Stack Technique

* **Langage** : Python 3
* **Deep Learning** : TensorFlow, Keras (LSTM, Dense, Dropout)
* **Computer Vision** : MediaPipe Hand Landmarker, OpenCV
* **Data Processing** : NumPy, Scikit-Learn

---

## 📂 Structure du Répertoire

```text
├── sign_to_speech_pipeline.ipynb   # Notebook complet d'entraînement et inférence
├── README.md                       # Documentation du projet
└── requirements.txt                # Dépendances Python
