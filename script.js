// ===== CONFIGURATION =====
const MODEL_PATH = 'fashion_mnist_model.onnx';
const IMAGE_SIZE = 28;

const CLASSES = [
    'T-shirt/top',
    'Pantalon',
    'Pull',
    'Robe',
    'Manteau',
    'Sandale',
    'Chemise',
    'Basket',
    'Sac',
    'Bottine'
];

const CLASS_ICONS = [
    '👕', '👖', '🧥', '👗', '🧥',
    '👡', '👔', '👟', '👜', '👢'
];

// ===== VARIABLES GLOBALES =====
let session = null;
let webcamStream = null;
let currentMode = 'upload';

// ===== ÉLÉMENTS DOM =====
const uploadModeBtn = document.getElementById('uploadModeBtn');
const webcamModeBtn = document.getElementById('webcamModeBtn');
const uploadSection = document.getElementById('uploadSection');
const webcamSection = document.getElementById('webcamSection');
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const webcam = document.getElementById('webcam');
const captureBtn = document.getElementById('captureBtn');
const loading = document.getElementById('loading');
const previewSection = document.getElementById('previewSection');
const previewCanvas = document.getElementById('previewCanvas');
const resultsContainer = document.getElementById('resultsContainer');
const resetBtn = document.getElementById('resetBtn');
const errorMessage = document.getElementById('errorMessage');

// ===== INITIALISATION =====
async function init() {
    try {
        console.log('🚀 Initialisation de ONNX Runtime...');
        session = await ort.InferenceSession.create(MODEL_PATH);
        console.log('✅ Modèle chargé avec succès !');
        console.log('📊 Entrées:', session.inputNames);
        console.log('📊 Sorties:', session.outputNames);
    } catch (error) {
        console.error('❌ Erreur de chargement du modèle:', error);
        showError('Erreur lors du chargement du modèle. Vérifiez que fashion_mnist_model.onnx est présent.');
    }
}

// ===== GESTION DES MODES =====
uploadModeBtn.addEventListener('click', () => {
    switchMode('upload');
});

webcamModeBtn.addEventListener('click', () => {
    switchMode('webcam');
});

function switchMode(mode) {
    currentMode = mode;
    
    // Mise à jour des boutons
    uploadModeBtn.classList.toggle('active', mode === 'upload');
    webcamModeBtn.classList.toggle('active', mode === 'webcam');
    
    // Mise à jour des sections
    uploadSection.classList.toggle('active', mode === 'upload');
    webcamSection.classList.toggle('active', mode === 'webcam');
    
    // Gestion de la webcam
    if (mode === 'webcam') {
        startWebcam();
    } else {
        stopWebcam();
    }
    
    // Réinitialiser l'affichage
    hidePreview();
}

// ===== GESTION UPLOAD =====
uploadZone.addEventListener('click', () => {
    fileInput.click();
});

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImageFile(file);
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImageFile(file);
    }
});

async function handleImageFile(file) {
    const reader = new FileReader();
    reader.onload = async (e) => {
        const img = new Image();
        img.onload = async () => {
            await processImage(img);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// ===== GESTION WEBCAM =====
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        webcam.srcObject = webcamStream;
        console.log('✅ Webcam démarrée');
    } catch (error) {
        console.error('❌ Erreur webcam:', error);
        showError('Impossible d\'accéder à la webcam. Vérifiez les permissions.');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
        webcam.srcObject = null;
        console.log('🛑 Webcam arrêtée');
    }
}

captureBtn.addEventListener('click', async () => {
    const canvas = document.createElement('canvas');
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0);
    
    const img = new Image();
    img.onload = async () => {
        await processImage(img);
    };
    img.src = canvas.toDataURL();
});

// ===== PRÉTRAITEMENT IMAGE =====
function preprocessImage(img) {
    const canvas = document.createElement('canvas');
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    const ctx = canvas.getContext('2d');
    
    // Fond BLANC (Fashion MNIST a fond blanc, pas noir)
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    
    // Calculer les dimensions
    const scale = Math.min(IMAGE_SIZE / img.width, IMAGE_SIZE / img.height);
    const width = img.width * scale;
    const height = img.height * scale;
    const x = (IMAGE_SIZE - width) / 2;
    const y = (IMAGE_SIZE - height) / 2;
    
    // Dessiner l'image
    ctx.drawImage(img, x, y, width, height);
    
    // Récupérer les pixels
    const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const pixels = imageData.data;
    
    // Convertir en tensor
    const inputData = new Float32Array(IMAGE_SIZE * IMAGE_SIZE);
    
    for (let i = 0; i < pixels.length; i += 4) {
        // Conversion RGB → Grayscale (formule standard)
        const gray = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
        
        // INVERSER si fond sombre (Fashion MNIST = fond clair, vêtement sombre)
        const inverted = 255 - gray;
        
        // Normalisation [-1, 1]
        inputData[i / 4] = (inverted / 255.0 - 0.5) / 0.5;
    }
    
    return inputData;
}

// ===== INFÉRENCE =====
async function processImage(img) {
    try {
        showLoading();
        hideError();
        
        // Prétraitement
        const inputData = preprocessImage(img);
        
        // Créer le tensor ONNX
        const tensor = new ort.Tensor('float32', inputData, [1, 1, IMAGE_SIZE, IMAGE_SIZE]);
        
        // Inférence
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        
        // Récupérer les logits
        const output = results.output.data;
        
        // Softmax pour obtenir les probabilités
        const probabilities = softmax(Array.from(output));
        
        // Trier par probabilité décroissante
        const predictions = probabilities
            .map((prob, idx) => ({ class: CLASSES[idx], icon: CLASS_ICONS[idx], probability: prob }))
            .sort((a, b) => b.probability - a.probability);
        
        // Afficher les résultats
        displayResults(img, predictions);
        
        hideLoading();
        
    } catch (error) {
        console.error('❌ Erreur lors de l\'inférence:', error);
        showError('Erreur lors de l\'analyse de l\'image. Veuillez réessayer.');
        hideLoading();
    }
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exp = arr.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum);
}

// ===== AFFICHAGE DES RÉSULTATS =====
function displayResults(img, predictions) {
    // Afficher l'image dans le canvas
    const ctx = previewCanvas.getContext('2d');
    
    // Taille d'affichage (plus grande pour mieux voir)
    const displaySize = 280;
    previewCanvas.width = displaySize;
    previewCanvas.height = displaySize;
    
    // Fond noir
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, displaySize, displaySize);
    
    // Calculer les dimensions pour conserver le ratio
    const scale = Math.min(displaySize / img.width, displaySize / img.height);
    const width = img.width * scale;
    const height = img.height * scale;
    const x = (displaySize - width) / 2;
    const y = (displaySize - height) / 2;
    
    // Dessiner l'image
    ctx.drawImage(img, x, y, width, height);
    
    // Afficher les prédictions (top 3)
    resultsContainer.innerHTML = '';
    predictions.slice(0, 3).forEach((pred, index) => {
        const percentage = (pred.probability * 100).toFixed(1);
        
        const item = document.createElement('div');
        item.className = 'result-item';
        item.style.animationDelay = `${index * 0.1}s`;
        
        item.innerHTML = `
            <div class="result-icon">${pred.icon}</div>
            <div class="result-content">
                <div class="result-label">${pred.class}</div>
                <div style="display: flex; align-items: center;">
                    <div class="confidence-bar" style="flex: 1;">
                        <div class="confidence-fill" style="width: ${percentage}%"></div>
                    </div>
                    <div class="result-percentage">${percentage}%</div>
                </div>
            </div>
        `;
        
        resultsContainer.appendChild(item);
    });
    
    // Afficher la section preview
    previewSection.classList.add('active');
    
    // Scroll vers les résultats
    previewSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ===== RÉINITIALISATION =====
resetBtn.addEventListener('click', () => {
    hidePreview();
    fileInput.value = '';
    if (currentMode === 'webcam') {
        startWebcam();
    }
});

function hidePreview() {
    previewSection.classList.remove('active');
}

// ===== GESTION DU LOADING =====
function showLoading() {
    loading.classList.add('active');
}

function hideLoading() {
    loading.classList.remove('active');
}

// ===== GESTION DES ERREURS =====
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('active');
}

function hideError() {
    errorMessage.classList.remove('active');
}

// ===== NETTOYAGE À LA FERMETURE =====
window.addEventListener('beforeunload', () => {
    stopWebcam();
});

init();