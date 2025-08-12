#!/usr/bin/env python3
"""
ChangeFormer Web UI
A Flask web application for change detection using ChangeFormer
"""

import os
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import uuid
from datetime import datetime
import requests
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import ChangeFormer modules
import utils
from models.basic_model import CDEvaluator
from argparse import Namespace

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['OUTPUT_FOLDER'] = 'temp_outputs'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global model instance
model = None
device = None

def initialize_model():
    """Initialize the ChangeFormer model"""
    global model, device
    
    print("Initializing ChangeFormer model...")
    
    # Create args object
    args = Namespace(
        n_class=2,
        embed_dim=256,
        net_G='ChangeFormerV6',
        gpu_ids='-1',  # Use CPU
        checkpoint_dir='checkpoints/ChangeFormer_LEVIR',
        output_folder=app.config['OUTPUT_FOLDER']
    )
    
    # Set device
    utils.get_device(args)
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids) > 0
                        else "cpu")
    
    # Initialize model
    model = CDEvaluator(args)
    model.load_checkpoint('best_ckpt.pt')
    model.eval()
    
    print(f"Model loaded successfully on {device}")

def preprocess_image(image_path, target_size=256):
    """Preprocess image from file path for the model"""
    img = Image.open(image_path).convert('RGB')
    return preprocess_pil_image(img, target_size)

def preprocess_pil_image(img, target_size=256):
    """Preprocess PIL image for the model"""
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    # Normalize to [-1, 1] range
    img_array = (img_array.astype(np.float32) / 127.5) - 1.0
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor

def run_change_detection(image_a, image_b):
    """Run change detection on two PIL Images"""
    global model, device
    
    if model is None:
        raise Exception("Model not initialized")
    
    # Preprocess images
    img_a_tensor = preprocess_pil_image(image_a)
    img_b_tensor = preprocess_pil_image(image_b)
    
    # Move to device
    img_a_tensor = img_a_tensor.to(device)
    img_b_tensor = img_b_tensor.to(device)
    
    # Create batch
    batch = {
        'A': img_a_tensor,
        'B': img_b_tensor,
        'name': ['change_detection']
    }
    
    # Run inference
    with torch.no_grad():
        score_map = model._forward_pass(batch)
    
    # Convert prediction to image
    pred = score_map[0, 0].cpu().numpy()  # Remove batch and channel dimensions
    
    # Use the working normalization method (same as in process_image_png.py)
    pred_normalized = pred.copy()
    
    # Scale to 0-255 range
    if pred_normalized.max() > pred_normalized.min():
        pred_normalized = ((pred_normalized - pred_normalized.min()) / 
                          (pred_normalized.max() - pred_normalized.min())) * 255
    else:
        pred_normalized = pred_normalized * 255
    
    # Convert to PIL Image
    pred_img = Image.fromarray(pred_normalized.astype(np.uint8))
    
    # Create highlighted version with bounding boxes
    highlighted_img = create_highlighted_image_from_pil(pred_normalized, image_a)
    
    # Analyze changes for detailed description
    change_analysis = analyze_changes(pred_normalized)
    
    return pred_img, highlighted_img, change_analysis

def create_highlighted_image(pred_array, original_image_path, threshold=128):
    """Create a highlighted version of the original image with bounding boxes around changes"""
    
    # Load original image
    original_img = Image.open(original_image_path).convert('RGB')
    return create_highlighted_image_from_pil(pred_array, original_img, threshold)

def create_highlighted_image_from_pil(pred_array, original_img, threshold=128):
    """Create a highlighted version of the original PIL image with bounding boxes around changes"""
    
    # Resize original image
    original_img = original_img.resize((256, 256), Image.Resampling.LANCZOS)
    original_array = np.array(original_img)
    
    # Create change mask
    change_mask = pred_array > threshold
    
    # Find contours of changed regions
    import cv2
    change_mask_uint8 = change_mask.astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(change_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create highlighted image
    highlighted_array = original_array.copy()
    
    # Draw bounding boxes around significant changes
    min_area = 50  # Minimum area to consider for bounding box
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw red rectangle
            cv2.rectangle(highlighted_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add semi-transparent overlay
            overlay = highlighted_array.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), -1)
            highlighted_array = cv2.addWeighted(highlighted_array, 0.8, overlay, 0.2, 0)
    
    # Convert back to PIL Image
    highlighted_img = Image.fromarray(highlighted_array)
    
    return highlighted_img

def analyze_changes(pred_array):
    """Analyze the change detection result and provide detailed description"""
    
    # Calculate change statistics
    total_pixels = pred_array.size
    change_threshold = 128  # Threshold for considering a pixel as changed
    changed_pixels = np.sum(pred_array > change_threshold)
    change_percentage = (changed_pixels / total_pixels) * 100
    
    # Find regions with significant changes
    change_mask = pred_array > change_threshold
    
    # Calculate change intensity
    avg_change_intensity = np.mean(pred_array[change_mask]) if np.any(change_mask) else 0
    
    # Determine change severity
    if change_percentage == 0:
        severity = "No changes detected"
        description = "The two images appear to be identical or very similar. No significant changes were detected between the time periods."
    elif change_percentage < 5:
        severity = "Minimal changes"
        description = f"Very few changes detected ({change_percentage:.1f}% of the area). These might be minor variations or noise."
    elif change_percentage < 15:
        severity = "Minor changes"
        description = f"Some changes detected ({change_percentage:.1f}% of the area). These could be seasonal variations, construction, or environmental changes."
    elif change_percentage < 30:
        severity = "Moderate changes"
        description = f"Significant changes detected ({change_percentage:.1f}% of the area). This indicates notable development, construction, or environmental changes."
    else:
        severity = "Major changes"
        description = f"Extensive changes detected ({change_percentage:.1f}% of the area). This suggests major development, land use changes, or significant environmental events."
    
    # Create detailed analysis
    analysis = {
        'severity': severity,
        'description': description,
        'change_percentage': float(change_percentage),
        'changed_pixels': int(changed_pixels),
        'total_pixels': int(total_pixels),
        'avg_intensity': float(avg_change_intensity),
        'has_changes': bool(change_percentage > 0)
    }
    
    return analysis

def download_image_from_url(url):
    """Download image from URL and return PIL Image"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        raise Exception(f"Failed to download image from {url}: {str(e)}")

def upload_image_to_imgbb(image, api_key):
    """Upload image to ImgBB and return the URL"""
    try:
        # Convert PIL image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Prepare the upload
        files = {'image': ('result.png', buffer, 'image/png')}
        data = {'key': api_key}
        
        response = requests.post('https://api.imgbb.com/1/upload', files=files, data=data)
        response.raise_for_status()
        
        result = response.json()
        if result['success']:
            return result['data']['url']
        else:
            raise Exception(f"ImgBB upload failed: {result.get('error', {}).get('message', 'Unknown error')}")
    except Exception as e:
        raise Exception(f"Failed to upload to ImgBB: {str(e)}")

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/', methods=['POST'])
def index_post():
    """Redirect POST requests to upload endpoint"""
    return upload_images()

@app.route('/upload', methods=['POST'])
def upload_images():
    """Handle image upload and change detection"""
    try:
        # Check if files were uploaded
        if 'image_a' not in request.files or 'image_b' not in request.files:
            return jsonify({'error': 'Please upload both images'}), 400
        
        file_a = request.files['image_a']
        file_b = request.files['image_b']
        
        if file_a.filename == '' or file_b.filename == '':
            return jsonify({'error': 'Please select both images'}), 400
        
        # Generate unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        
        filename_a = f"image_a_{timestamp}_{unique_id}.png"
        filename_b = f"image_b_{timestamp}_{unique_id}.png"
        output_filename = f"result_{timestamp}_{unique_id}.png"
        
        # Save uploaded images
        path_a = os.path.join(app.config['UPLOAD_FOLDER'], filename_a)
        path_b = os.path.join(app.config['UPLOAD_FOLDER'], filename_b)
        
        file_a.save(path_a)
        file_b.save(path_b)
        
        # Run change detection
        result_image, highlighted_image, change_analysis = run_change_detection(path_a, path_b)
        
        # Save results
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        result_image.save(output_path)
        
        highlighted_filename = f"highlighted_{output_filename}"
        highlighted_path = os.path.join(app.config['OUTPUT_FOLDER'], highlighted_filename)
        highlighted_image.save(highlighted_path)
        
        # Convert images to base64 for display
        img_a_b64 = image_to_base64(Image.open(path_a))
        img_b_b64 = image_to_base64(Image.open(path_b))
        result_b64 = image_to_base64(result_image)
        highlighted_b64 = image_to_base64(highlighted_image)
        
        # Clean up uploaded files
        os.remove(path_a)
        os.remove(path_b)
        
        return jsonify({
            'success': True,
            'image_a': img_a_b64,
            'image_b': img_b_b64,
            'result': result_b64,
            'highlighted': highlighted_b64,
            'output_filename': output_filename,
            'highlighted_filename': highlighted_filename,
            'analysis': change_analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_urls', methods=['POST'])
def process_image_urls():
    """Handle image URLs and return ImgBB links"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        image_a_url = data.get('image_a_url')
        image_b_url = data.get('image_b_url')
        
        if not image_a_url or not image_b_url:
            return jsonify({'error': 'Both image_a_url and image_b_url are required'}), 400
        
        # Get API key from environment variable
        imgbb_api_key = os.getenv('IMGBB_API_KEY')
        if not imgbb_api_key:
            return jsonify({'error': 'IMGBB_API_KEY not found in environment variables'}), 500
        
        # Download images from URLs
        print(f"Downloading image A from: {image_a_url}")
        image_a = download_image_from_url(image_a_url)
        
        print(f"Downloading image B from: {image_b_url}")
        image_b = download_image_from_url(image_b_url)
        
        # Run change detection
        print("Running change detection...")
        result_image, highlighted_image, change_analysis = run_change_detection(image_a, image_b)
        
        # Upload result to ImgBB
        print("Uploading result to ImgBB...")
        result_url = upload_image_to_imgbb(result_image, imgbb_api_key)
        
        print("Uploading highlighted result to ImgBB...")
        highlighted_url = upload_image_to_imgbb(highlighted_image, imgbb_api_key)
        
        return jsonify({
            'success': True,
            'image_a_url': image_a_url,
            'image_b_url': image_b_url,
            'result_url': result_url,
            'highlighted_url': highlighted_url,
            'analysis': change_analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_result(filename):
    """Download the result image"""
    try:
        return send_file(
            os.path.join(app.config['OUTPUT_FOLDER'], filename),
            as_attachment=True,
            download_name=f"change_detection_result_{filename}"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'None'
    })

if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()
    
    print("Starting ChangeFormer Web UI...")
    print("Open your browser and go to: http://localhost:5000")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False) 