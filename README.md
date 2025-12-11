# 👗 Fashion AI - Reconnaissance de Vêtements

Application web de reconnaissance de vêtements utilisant Fashion MNIST avec PyTorch et ONNX Runtime Web.

<img width="1920" height="912" alt="image" src="https://github.com/user-attachments/assets/6b168cb4-4e10-472c-99f2-1c6931d5b796" />


## 🎯 Fonctionnalités

- ✅ **Upload d'images** : Glissez-déposez ou sélectionnez une photo
- ✅ **Webcam en direct** : Capturez des photos avec votre caméra
- ✅ **10 classes de vêtements** : T-shirt, Pantalon, Pull, Robe, Manteau, Sandale, Chemise, Basket, Sac, Bottine
- ✅ **Interface moderne** : Design girly/coloré avec animations
- ✅ **Prédictions en temps réel** : Inférence directement dans le navigateur

## 🧠 Techniques ML Utilisées

### Modèle CNN avec :
- **Dropout** (0.25 et 0.5) pour la régularisation
- **Batch Normalization** après chaque couche convolutive et FC
- **Data Augmentation** : rotation, translation, flip, zoom

### Architecture :
```
Conv2d(1→32) → BatchNorm → ReLU → MaxPool
Conv2d(32→64) → BatchNorm → ReLU → MaxPool
Conv2d(64→128) → BatchNorm → ReLU
Dropout(0.25)
FC(128*7*7 → 256) → BatchNorm → ReLU
Dropout(0.5)
FC(256 → 10)
```

## 📦 Installation

### 1. Lancer l'application web

```bash
# Serveur HTTP simple avec Python
python -m http.server 8000

# Ou avec Node.js
npx http-server -p 8000
```

### 2. Ouvrir dans le navigateur

```
http://localhost:8000
```

## 🎨 Utilisation

### Mode Upload
1. Cliquez sur "📤 Upload"
2. Glissez une image ou cliquez pour sélectionner
3. Les prédictions s'affichent automatiquement

### Mode Webcam
1. Cliquez sur "📸 Webcam"
2. Autorisez l'accès à la caméra
3. Cliquez sur "📸 Capturer" pour prendre une photo
4. Les prédictions s'affichent automatiquement

## 📊 Résultats Attendus

Avec les techniques de régularisation appliquées :

- **Précision Train** : ~95%
- **Précision Test** : ~92-93%
- **Pas de surapprentissage** grâce à Dropout + BatchNorm + Data Augmentation

## 📈 Suivi de l'entraînement avec TensorBoard

<img width="1192" height="880" alt="image" src="https://github.com/user-attachments/assets/1580e25a-83fc-4022-a0ce-690f680c5ddb" />

<img width="1248" height="958" alt="image" src="https://github.com/user-attachments/assets/40f760e3-5fc7-4044-b9ba-51c3882f3c7c" />


## 🎓 Concepts Clés

### Dropout
Désactive aléatoirement des neurones pendant l'entraînement pour éviter le surapprentissage.

### Batch Normalization
Normalise les activations de chaque couche pour stabiliser et accélérer l'entraînement.

### Data Augmentation
Transformations aléatoires des images d'entraînement :
- Rotation (±10°)
- Translation (±10%)
- Zoom (90-110%)
- Flip horizontal

### ONNX
Format d'export universel permettant l'inférence dans le navigateur avec ONNX Runtime Web.

## 🚀 Améliorations Possibles

1. **Transfer Learning** avec ResNet ou EfficientNet
2. **Détection d'objets** pour localiser plusieurs vêtements
3. **Recherche par similarité** dans une base de données
4. **Filtres de style** (couleur, matière, saison)
5. **API backend** avec FastAPI/Flask

## 📝 Notes Techniques

### Prétraitement des Images
```javascript
// 1. Redimensionner à 28x28
// 2. Convertir en niveaux de gris
// 3. Normaliser [-1, 1] : (pixel / 255 - 0.5) / 0.5
```

### Compatibilité ONNX Runtime Web
- Opset version 13
- Pas de fichiers .data externes
- Input shape dynamique pour batch processing

## 🐛 Troubleshooting

### Erreur "Model not found"
Vérifiez que `fashion_mnist_model.onnx` est dans le même dossier que `index.html`.

### Erreur webcam
Utilisez HTTPS ou localhost (permissions requises).

### Prédictions incorrectes
Le modèle est entraîné sur Fashion MNIST (images 28x28 en niveaux de gris). Les performances peuvent varier sur des photos réelles couleur.

## 📚 Ressources

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)


