from flask import Flask, request, render_template, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and processor
processor = None
model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the BLIP model and processor"""
    global processor, model
    try:
        logger.info("Loading BLIP model...")
        
        # Load processor
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        if processor is None:
            raise Exception("Failed to load processor")
        logger.info("Processor loaded successfully")
        
        # Load model
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        if model is None:
            raise Exception("Failed to load model")
        logger.info("Model loaded successfully")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        # Verify model is properly loaded
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Processor type: {type(processor)}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Set globals to None on failure
        processor = None
        model = None
        raise

def generate_caption(image_path):
    """Generate caption for an image"""
    global model, processor
    
    try:
        # Check if model and processor are loaded
        if model is None or processor is None:
            logger.error("Model or processor is None")
            load_model()  # Try to reload
            if model is None or processor is None:
                return "Error: Model not loaded properly"
        
        # Check if image file exists
        if not os.path.exists(image_path):
            return "Error: Image file not found"
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Image loaded successfully: {image.size}")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return f"Error loading image: {str(e)}"
        
        # Process image with error handling
        try:
            inputs = processor(image, return_tensors="pt")
            logger.info("Image processed by processor")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return f"Error processing image: {str(e)}"
        
        # Move inputs to same device as model
        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logger.info(f"Inputs moved to device: {device}")
        except Exception as e:
            logger.error(f"Error moving inputs to device: {str(e)}")
            return f"Error moving inputs to device: {str(e)}"
        
        # Generate caption
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50, num_beams=5)
            logger.info("Caption generated successfully")
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return f"Error during generation: {str(e)}"
        
        # Decode caption
        try:
            caption = processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Caption decoded: {caption}")
            return caption if caption else "No caption generated"
        except Exception as e:
            logger.error(f"Error decoding caption: {str(e)}")
            return f"Error decoding caption: {str(e)}"
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_caption: {str(e)}")
        return f"Unexpected error: {str(e)}"

def image_to_base64(image_path):
    """Convert image to base64 for display in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Get image format
            with Image.open(image_path) as img:
                img_format = img.format.lower()
                
            return f"data:image/{img_format};base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and generate caption"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Generate caption
            caption = generate_caption(filepath)
            
            # Convert image to base64 for display
            image_base64 = image_to_base64(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'caption': caption,
                'image': image_base64,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'processor_loaded': processor is not None,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    })

# HTML Template (embedded for simplicity)
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>8-BIT CAPTION GEN</title>
    <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Press Start 2P', 'Courier New', monospace;
            background: #C0C2B8;
            min-height: 100vh;
            padding: 40px 20px;
            color: #D5FF40;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: #000000;
            border: 4px solid #000000;
            border-radius: 0;
            box-shadow: 10px 10px 0px rgba(0,0,0,0.8);
            overflow: hidden;
            position: relative;
        }
        
        .header {
            background: #000000;
            color: #D5FF40;
            padding: 30px;
            text-align: center;
            border-bottom: 4px dashed #333;
        }
        
        .header h1 {
            font-size: 1.5em;
            margin-bottom: 15px;
            text-transform: uppercase;
            line-height: 1.5;
            text-shadow: 2px 2px 0 #333;
        }
        
        .header p {
            font-size: 0.6em;
            color: #C0C2B8;
            line-height: 1.5;
        }
        
        .content {
            padding: 40px;
            background: #000000;
        }
        
        .upload-area {
            border: 4px dashed #C0C2B8;
            border-radius: 0;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.2s ease;
            cursor: pointer;
            background: #000;
            color: #D5FF40;
        }
        
        .upload-area:hover, .upload-area.dragover {
            border-color: #D5FF40;
            background: #111;
        }
        
        .upload-icon {
            font-size: 2em;
            margin-bottom: 20px;
            color: #D5FF40;
        }
        
        .upload-text {
            font-size: 0.7em;
            color: #C0C2B8;
            margin-bottom: 20px;
            line-height: 1.8;
        }
        
        .file-input { display: none; }
        
        .btn {
            background: #D5FF40;
            color: #000000;
            border: 4px solid #D5FF40;
            padding: 15px 30px;
            border-radius: 0;
            font-size: 0.8em;
            font-family: 'Press Start 2P', monospace;
            cursor: pointer;
            text-transform: uppercase;
            box-shadow: 4px 4px 0px #C0C2B8;
            transition: all 0.1s;
        }
        
        .btn:hover {
            background: #000;
            color: #D5FF40;
        }
        
        .btn:active {
            box-shadow: 0px 0px 0px #C0C2B8;
            transform: translate(4px, 4px);
        }
        
        .checkbox-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 25px 0;
            font-size: 0.7em;
            color: #C0C2B8;
            cursor: pointer;
            user-select: none;
        }
        
        .checkbox-container input {
            cursor: pointer;
            width: 20px;
            height: 20px;
            margin-right: 15px;
            accent-color: #000;
            border: 2px solid #D5FF40;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            font-size: 0.7em;
            color: #D5FF40;
        }
        
        .blink {
            animation: blinker 1s linear infinite;
        }
        @keyframes blinker { 50% { opacity: 0; } }
        
        .result {
            display: none;
            background: #111;
            border: 4px solid #333;
            border-radius: 0;
            padding: 20px;
            margin-top: 30px;
        }
        
        .result-image {
            max-width: 100%;
            height: auto;
            border: 4px solid #C0C2B8;
            margin-bottom: 20px;
            image-rendering: auto;
        }
        
        .caption {
            background: #000;
            padding: 20px;
            border: 2px solid #D5FF40;
            font-size: 0.8em;
            line-height: 1.6;
            color: #D5FF40;
        }
        
        .error {
            background: #000;
            color: #ff4040;
            padding: 15px;
            border: 2px solid #ff4040;
            margin-top: 20px;
            font-size: 0.7em;
        }
        
        .file-info {
            background: #222;
            padding: 15px;
            border: 2px solid #C0C2B8;
            margin-top: 15px;
            font-size: 0.6em;
            color: #D5FF40;
            line-height: 1.5;
        }
        
        .crt-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%);
            background-size: 100% 4px;
            pointer-events: none;
            z-index: 999;
        }
    </style>
</head>
<body>
    <div class="crt-overlay"></div>
    <div class="container">
        <div class="header">
            <h1>IMAGE <br>CAPTION GENERATOR</h1>
            <p>> AWAITING INPUT FOR AI ANALYSIS...</p>
        </div>
        
        <div class="content">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">[+]</div>
                    <div class="upload-text">CLICK TO SELECT OR DRAG & DROP</div>
                    <input type="file" id="fileInput" class="file-input" accept="image/*">
                    <button type="button" class="btn" onclick="document.getElementById('fileInput').click()">
                        SELECT IMAGE
                    </button>
                </div>
                
                <div id="fileInfo" class="file-info" style="display: none;"></div>
                
                <div class="loading" id="loading">
                    <p class="blink">> PROCESSING DATA...</p>
                </div>
                
                <div class="result" id="result">
                    <img id="resultImage" class="result-image" alt="Uploaded image">
                    <div class="caption" id="caption"></div>
                </div>
                
                <div id="error" class="error" style="display: none;"></div>
            </form>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const error = document.getElementById('error');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
        
        function handleFileSelect(file) {
            if (!file.type.startsWith('image/')) {
                showError('ERROR: INVALID FILE TYPE.');
                return;
            }
            if (file.size > 16 * 1024 * 1024) {
                showError('ERROR: FILE SIZE EXCEEDS LIMIT.');
                return;
            }
            
            fileInfo.innerHTML = `> SELECTED: ${file.name}<br>> SIZE: ${(file.size / 1024 / 1024).toFixed(2)} MB`;
            fileInfo.style.display = 'block';
            
            uploadFile(file);
        }
        
        function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            loading.style.display = 'block';
            result.style.display = 'none';
            error.style.display = 'none';
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                
                if (data.success) {
                    document.getElementById('resultImage').src = data.image;
                    document.getElementById('caption').textContent = "> " + data.caption.toUpperCase();
                    result.style.display = 'block';
                } else {
                    showError("> ERROR: " + (data.error || 'PROCESSING FAILED.').toUpperCase());
                }
            })
            .catch(err => {
                loading.style.display = 'none';
                showError('> SYSTEM ERROR: NETWORK FAILURE.');
                console.error('Error:', err);
            });
        }
        
        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
            result.style.display = 'none';
        }
    </script>
</body>
</html>
'''

# Create templates directory and save template
templates_dir = 'templates'
os.makedirs(templates_dir, exist_ok=True)

# Write template with explicit UTF-8 encoding to handle any Unicode characters
try:
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_template)
except UnicodeEncodeError:
    # Fallback: write without problematic characters
    clean_template = html_template.encode('ascii', 'ignore').decode('ascii')
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(clean_template)

if __name__ == '__main__':
    # Load model on startup
    try:
        load_model()
        print("🚀 Starting Flask application...")
        print("📝 Model loaded successfully!")
        print("🌐 Open http://localhost:5050 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5050)
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        print("Make sure you have the required dependencies installed:")
        print("pip install flask torch transformers pillow")