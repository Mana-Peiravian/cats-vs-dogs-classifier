let model = null;
let isModelLoaded = false;

function $(id) { return document.getElementById(id); }

async function loadModel() {
  const statusEl = $('model-status');
  const uploadArea = $('upload-area');

  try {
    console.log('🔄 Loading model from:', './models/tfjs_model/model.json');
    statusEl.textContent = '🔄 Downloading model files...';
    statusEl.className = 'model-status loading';

    model = await tf.loadGraphModel('./models/tfjs_model/model.json?v=20');

    console.log('GraphModel input nodes:', model.inputNodes);
    console.log('GraphModel output nodes:', model.outputNodes);

    isModelLoaded = true;

    statusEl.textContent = '✅ Real TensorFlow model loaded! Ready for predictions.';
    statusEl.className = 'model-status success';

    uploadArea.classList.remove('disabled');
    uploadArea.querySelector('p em').textContent = 'Ready to analyze your images!';
  } catch (error) {
    console.error('❌ Model loading failed:', error);
    statusEl.innerHTML = `
      <strong>❌ Model Loading Failed</strong><br>
      <small>Error: ${error.message}</small><br>
      <small>Make sure model files are uploaded to GitHub</small>
    `;
    statusEl.className = 'model-status error';
  }
}

async function runInference(tensor) {
  let inputName  = (model.inputNodes && model.inputNodes[0]) || (model.inputs && model.inputs[0]?.name);
  let outputName = (model.outputNodes && model.outputNodes[0]) || (model.outputs && model.outputs[0]?.name);

  const candidates = [
    inputName,
    'serving_default_image:0',
    'serving_default_input_1:0',
    'image:0',
    'input_1:0'
  ].filter(Boolean);

  let out;
  let lastErr;
  for (const name of candidates) {
    try {
      console.log('Trying input name:', name, 'output:', outputName);
      out = await model.executeAsync({ [name]: tensor }, outputName ? [outputName] : undefined);
      inputName = name;
      break;
    } catch (e) {
      lastErr = e;
    }
  }
  if (!out) throw lastErr || new Error('Could not execute model with any candidate input name.');

  const y = Array.isArray(out) ? out[0] : out;
  const scores = await y.data();
  const score = scores[0];

  if (Array.isArray(out)) out.forEach(t => t.dispose());
  else y.dispose();

  return score;
}

function showImagePreview(src) {
  const preview = $('imagePreview');
  const img = $('previewImg');
  img.src = src;
  preview.style.display = 'block';
  $('result').style.display = 'none';
}

function showResult(prediction) {
  const resultDiv = $('result');
  const predictionText = $('prediction');
  const confidenceText = $('confidence');
  const technicalDetails = $('technical-details');

  let label, emoji, confidence, className;

  if (prediction > 0.5) {
    label = "This is a DOG!";
    emoji = "🐶";
    confidence = (prediction * 100).toFixed(1);
    className = "dog";
  } else {
    label = "This is a CAT!";
    emoji = "🐱";
    confidence = ((1 - prediction) * 100).toFixed(1);
    className = "cat";
  }

  predictionText.textContent = `${emoji} ${label}`;
  confidenceText.textContent = `Model Confidence: ${confidence}%`;
  technicalDetails.innerHTML = `
    <strong>Technical Details:</strong><br>
    Raw prediction score: ${prediction.toFixed(4)}<br>
    Threshold: 0.5 (${prediction > 0.5 ? 'Dog' : 'Cat'} predicted)
  `;

  resultDiv.className = `result ${className}`;
  resultDiv.style.display = 'block';
}

async function classifyImage(imageData) {
  if (!isModelLoaded || !model) {
    alert('Model not loaded yet! Please wait.');
    return;
  }

  $('loading').style.display = 'block';
  $('result').style.display = 'none';

  try {
    const img = new Image();
    img.onload = async () => {
      try {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 150;
        canvas.height = 150;
        ctx.drawImage(img, 0, 0, 150, 150);

        const tensor = tf.browser.fromPixels(canvas)
          .toFloat()
          .div(255.0)
          .expandDims(0); // [1,150,150,3]

        console.log('📐 Input tensor shape:', tensor.shape);

        const score = await runInference(tensor);

        $('loading').style.display = 'none';
        showResult(score);

        // Clean up memory
        tensor.dispose();
      } catch (predError) {
        console.error('❌ Prediction error:', predError);
        $('loading').style.display = 'none';
        alert('Prediction failed: ' + predError.message);
      }
    };

    img.onerror = () => {
      console.error('❌ Image loading failed');
      $('loading').style.display = 'none';
      alert('Failed to load image');
    };

    img.src = imageData;
  } catch (error) {
    console.error('❌ Classification error:', error);
    $('loading').style.display = 'none';
    alert('Classification failed: ' + error.message);
  }
}

// --- Event wiring (no inline attributes) ---
function onUploadClick() {
  $('fileInput').click();
}

function onFileChange(e) {
  if (!isModelLoaded) {
    alert('Please wait for the model to load first!');
    return;
  }
  const file = e.target.files?.[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = ev => {
      showImagePreview(ev.target.result);
      classifyImage(ev.target.result);
    };
    reader.readAsDataURL(file);
  }
}

function onDragOver(e) {
  if (!isModelLoaded) return;
  e.preventDefault();
  $('upload-area').classList.add('dragover');
}

function onDragLeave(e) {
  $('upload-area').classList.remove('dragover');
}

function onDrop(e) {
  if (!isModelLoaded) return;
  e.preventDefault();
  $('upload-area').classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0 && files[0].type.startsWith('image/')) {
    const file = files[0];
    const reader = new FileReader();
    reader.onload = ev => {
      showImagePreview(ev.target.result);
      classifyImage(ev.target.result);
    };
    reader.readAsDataURL(file);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  // Load model
  loadModel();

  // Bind events
  const area = $('upload-area');
  area.addEventListener('click', onUploadClick);
  area.addEventListener('dragover', onDragOver);
  area.addEventListener('dragleave', onDragLeave);
  area.addEventListener('drop', onDrop);

  $('fileInput').addEventListener('change', onFileChange);
});
