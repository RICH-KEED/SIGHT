import os
import sys
import requests
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import anthropic
import base64
from PIL import Image
import io
import numpy as np
import torch
import cv2
import argparse

# Add ChangeFormer to path
sys.path.append('../ChangeFormer')

# Import ChangeFormer components
try:
    from models.basic_model import CDEvaluator
    from utils import get_device
    CHANGEFORMER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ChangeFormer not available: {e}")
    CHANGEFORMER_AVAILABLE = False

load_dotenv()

class UnifiedAPI:
    def __init__(self):
        # Initialize TextAnalyzer
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        
        # Initialize ChangeFormer if available
        self.changeformer = None
        if CHANGEFORMER_AVAILABLE:
            self.changeformer = self._init_changeformer()
        
        # ImgBB API key
        self.imgbb_api_key = os.getenv('IMGBB_API_KEY')
        if not self.imgbb_api_key:
            raise ValueError("IMGBB_API_KEY not found in environment variables")
    
    def _init_changeformer(self):
        """Initialize ChangeFormer model"""
        try:
            # Create args object
            args = argparse.Namespace()
            args.gpu_ids = '-1'  # Use CPU by default, change to '0' for GPU
            args.checkpoint_root = '../ChangeFormer/checkpoints'
            args.project_name = 'ChangeFormer_LEVIR'
            args.n_class = 1  # Add missing n_class attribute
            args.net_G = 'ChangeFormerV6'  # Specify the model architecture
            args.embed_dim = 256  # Required for ChangeFormerV6
            args.checkpoint_dir = '../ChangeFormer/checkpoints/ChangeFormer_LEVIR'  # Required by CDEvaluator
            args.output_folder = '../ChangeFormer/temp_outputs'  # Required by CDEvaluator
            
            # Get device
            get_device(args)
            device = torch.device('cuda' if args.gpu_ids else 'cpu')

            # Ensure all required attrs exist
            defaults = {
                'net_G': 'ChangeFormerV6',
                'embed_dim': 256,
                'n_class': 1,
                'checkpoint_dir': '../ChangeFormer/checkpoints/ChangeFormer_LEVIR',
                'output_folder': '../ChangeFormer/temp_outputs',
            }
            for k, v in defaults.items():
                if not hasattr(args, k) or getattr(args, k) in (None, ''):
                    setattr(args, k, v)

            print(f"🔧 ChangeFormer args: {vars(args)}")
            
            # Initialize model
            model = CDEvaluator(args)
            model.load_checkpoint('best_ckpt.pt')
            model.net_G.eval()
            
            print("✅ ChangeFormer model loaded successfully!")
            return model
        except Exception as e:
            print(f"❌ Error loading ChangeFormer: {e}")
            return None
    
    def download_image_from_url(self, url):
        """Download image from URL and return PIL Image"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"Failed to download image from {url}: {e}")
    
    def upload_image_to_imgbb(self, image_pil):
        """Upload PIL image to ImgBB and return URL"""
        try:
            # Convert PIL to bytes
            img_byte_arr = io.BytesIO()
            image_pil.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Upload to ImgBB
            files = {'image': ('image.png', img_byte_arr, 'image/png')}
            data = {'key': self.imgbb_api_key}
            
            response = requests.post('https://api.imgbb.com/1/upload', files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                return result['data']['url']
            else:
                raise Exception(f"ImgBB upload failed: {response.text}")
                
        except Exception as e:
            raise Exception(f"Failed to upload image: {e}")
    
    def run_changeformer(self, image_a_url, image_b_url):
        """Run ChangeFormer on two image URLs and return result URL"""
        try:
            if not self.changeformer:
                raise Exception("ChangeFormer model not available")
            
            # Download images
            image_a = self.download_image_from_url(image_a_url)
            image_b = self.download_image_from_url(image_b_url)
            
            # Preprocess images for ChangeFormer (match web_ui)
            def preprocess_image(image):
                image = image.resize((256, 256), Image.Resampling.LANCZOS)
                img_array = np.array(image)
                img_array = (img_array.astype(np.float32) / 127.5) - 1.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                return img_tensor
            
            # Preprocess both images
            img_a_tensor = preprocess_image(image_a)
            img_b_tensor = preprocess_image(image_b)
            
            # Run inference via CDEvaluator pipeline (match web_ui)
            with torch.no_grad():
                device = next(self.changeformer.net_G.parameters()).device
                img_a_tensor = img_a_tensor.to(device)
                img_b_tensor = img_b_tensor.to(device)
                batch = { 'A': img_a_tensor, 'B': img_b_tensor, 'name': ['change_detection'] }
                score_map = self.changeformer._forward_pass(batch)
                pred = score_map[0, 0].detach().cpu().numpy()

                # Min-max normalize to 0-255 (exactly like web_ui)
                pred_normalized = pred.copy()
                if pred_normalized.max() > pred_normalized.min():
                    pred_normalized = ((pred_normalized - pred_normalized.min()) /
                                       (pred_normalized.max() - pred_normalized.min())) * 255.0
                else:
                    pred_normalized = pred_normalized * 255.0

                # Do NOT apply additional morphology here; return same as web_ui result
                result_image = Image.fromarray(pred_normalized.astype(np.uint8), mode='L')
            
            # Upload result to ImgBB
            result_url = self.upload_image_to_imgbb(result_image)
            
            return {
                'success': True,
                'result_url': result_url,
                'image_a_url': image_a_url,
                'image_b_url': image_b_url
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image_a_url': image_a_url,
                'image_b_url': image_b_url
            }
    
    def run_text_analysis(self, image_a_url, image_b_url, change_mask_url):
        """Run text analysis on three image URLs"""
        try:
            # Create messages with URLs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze these satellite images for changes. Provide a concise summary of:\n1. What changed (buildings, roads, vegetation, etc.)\n2. Scale of changes (small, medium, large areas)\n3. Type of development (urban, agricultural, infrastructure)\n4. Notable patterns or clusters\n\nBefore Image:. Also tell if the change is positive or negative."
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": image_a_url,
                            },
                        },
                        {
                            "type": "text",
                            "text": "After Image:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": image_b_url,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Change Detection Result (white areas = detected changes):"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": change_mask_url,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Keep the analysis clear and actionable for satellite image interpretation."
                        }
                    ],
                }
            ]
            
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=messages,
            )
            
            return {
                'success': True,
                'analysis': response.content[0].text,
                'image_a_url': image_a_url,
                'image_b_url': image_b_url,
                'change_mask_url': change_mask_url
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image_a_url': image_a_url,
                'image_b_url': image_b_url,
                'change_mask_url': change_mask_url
            }
    
    def run_combined(self, image_a_url, image_b_url):
        """Run both ChangeFormer and text analysis"""
        try:
            # Step 1: Run ChangeFormer
            changeformer_result = self.run_changeformer(image_a_url, image_b_url)
            
            if not changeformer_result['success']:
                return changeformer_result
            
            # Step 2: Run text analysis with the result
            text_result = self.run_text_analysis(
                image_a_url, 
                image_b_url, 
                changeformer_result['result_url']
            )
            
            if not text_result['success']:
                return text_result
            
            # Combine results
            return {
                'success': True,
                'changeformer_result_url': changeformer_result['result_url'],
                'text_analysis': text_result['analysis'],
                'image_a_url': image_a_url,
                'image_b_url': image_b_url
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image_a_url': image_a_url,
                'image_b_url': image_b_url
            }

# Initialize API
try:
    api = UnifiedAPI()
    print("✅ Unified API initialized successfully!")
    print(f"📊 ChangeFormer available: {CHANGEFORMER_AVAILABLE}")
except Exception as e:
    print(f"❌ Failed to initialize API: {e}")
    api = None

# Flask app
app = Flask(__name__)

@app.route('/changeformer', methods=['POST'])
def changeformer_endpoint():
    """Endpoint 1: ChangeFormer only - takes 2 image URLs"""
    if not api:
        return jsonify({'error': 'API not initialized'}), 500
    
    if not CHANGEFORMER_AVAILABLE:
        return jsonify({'error': 'ChangeFormer not available'}), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        image_a_url = data.get('image_a_url')
        image_b_url = data.get('image_b_url')
        
        if not image_a_url or not image_b_url:
            return jsonify({'error': 'Both image_a_url and image_b_url are required'}), 400
        
        result = api.run_changeformer(image_a_url, image_b_url)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/text-analysis', methods=['POST'])
def text_analysis_endpoint():
    """Endpoint 2: Text analysis only - takes 3 image URLs"""
    if not api:
        return jsonify({'error': 'API not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        image_a_url = data.get('image_a_url')
        image_b_url = data.get('image_b_url')
        change_mask_url = data.get('change_mask_url')
        
        if not image_a_url or not image_b_url or not change_mask_url:
            return jsonify({'error': 'All three URLs are required: image_a_url, image_b_url, change_mask_url'}), 400
        
        result = api.run_text_analysis(image_a_url, image_b_url, change_mask_url)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Provide backward-compatible alias matching TEXT-OUTPUT/ai.py
@app.route('/analyze', methods=['POST'])
def analyze_alias():
    """Alias for text-analysis; accepts the same payload with 3 URLs"""
    return text_analysis_endpoint()

@app.route('/combined', methods=['POST'])
def combined_endpoint():
    """Endpoint 3: Combined - takes 2 image URLs, runs both ChangeFormer and text analysis"""
    if not api:
        return jsonify({'error': 'API not initialized'}), 500
    
    if not CHANGEFORMER_AVAILABLE:
        return jsonify({'error': 'ChangeFormer not available for combined analysis'}), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        image_a_url = data.get('image_a_url')
        image_b_url = data.get('image_b_url')
        
        if not image_a_url or not image_b_url:
            return jsonify({'error': 'Both image_a_url and image_b_url are required'}), 400
        
        result = api.run_combined(image_a_url, image_b_url)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New: Legacy-compatible route matching prior web_ui behavior
@app.route('/process_urls', methods=['POST'])
def process_urls_endpoint():
    """Accepts two image URLs and optionally runs analysis.
    Payload:
    {
      "image_a_url": str,
      "image_b_url": str,
      "analyze": bool  # optional; default true
    }
    """
    if not api:
        return jsonify({'error': 'API not initialized'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        image_a_url = data.get('image_a_url')
        image_b_url = data.get('image_b_url')
        analyze = data.get('analyze', True)

        if not image_a_url or not image_b_url:
            return jsonify({'error': 'Both image_a_url and image_b_url are required'}), 400

        if analyze:
            result = api.run_combined(image_a_url, image_b_url)
        else:
            result = api.run_changeformer(image_a_url, image_b_url)

        status = 200 if result.get('success') else 500
        return jsonify(result), status

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = 'healthy' if api else 'unhealthy'
    return jsonify({
        'status': status, 
        'service': 'Unified ChangeFormer + Text Analysis API',
        'changeformer_available': CHANGEFORMER_AVAILABLE,
        'endpoints': {
            'changeformer': '/changeformer (POST) - 2 image URLs',
            'text_analysis': '/text_analysis (POST) - 3 image URLs', 
            'combined': '/combined (POST) - 2 image URLs'
        }
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API info"""
    return jsonify({
        'message': 'Unified ChangeFormer + Text Analysis API',
        'version': '1.0.0',
        'endpoints': {
            'health': 'GET /health',
            'changeformer': 'POST /changeformer',
            'text_analysis': 'POST /text-analysis',
            'combined': 'POST /combined'
        },
        'usage': {
            'changeformer': 'Send 2 image URLs for change detection',
            'text_analysis': 'Send 3 image URLs for AI analysis',
            'combined': 'Send 2 image URLs for both change detection + analysis'
        }
    })

if __name__ == "__main__":
    if api:
        print("🚀 Starting Unified API server...")
        print("📡 Available Endpoints:")
        print("   • Root: http://localhost:5002/")
        print("   • Health: http://localhost:5002/health")
        print("   • ChangeFormer: http://localhost:5002/changeformer")
        print("   • Text Analysis: http://localhost:5002/text-analysis")
        print("   • Combined: http://localhost:5002/combined")
        print(f"🔧 ChangeFormer Status: {'✅ Available' if CHANGEFORMER_AVAILABLE else '❌ Not Available'}")
        print("=" * 60)
        app.run(host='0.0.0.0', port=5002, debug=False)
    else:
        print("❌ Failed to initialize API. Check your configuration.")
        print("📋 Required environment variables:")
        print("   • ANTHROPIC_API_KEY")
        print("   • IMGBB_API_KEY")
        print("📁 Required files:")
        print("   • ../ChangeFormer/checkpoints/ChangeFormer_LEVIR/best_ckpt.pt") 