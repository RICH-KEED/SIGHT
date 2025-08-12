# Combined ChangeFormer + Text Analysis API

A unified API that combines ChangeFormer (change detection) and Claude AI (text analysis) for satellite image analysis.

## 🚀 Features

- **ChangeFormer**: AI-powered change detection between two satellite images
- **Text Analysis**: Detailed analysis of changes using Claude AI
- **Combined Workflow**: Automatic change detection + analysis in one request
- **URL-based**: Works with image URLs (ImgBB integration)

## 📋 Requirements

- Python 3.8+
- PyTorch
- Anthropic API key
- ImgBB API key

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the `TEXT-OUTPUT` directory:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
IMGBB_API_KEY=your_imgbb_api_key_here
```

### 3. ChangeFormer Model
Ensure the ChangeFormer model checkpoint is available at:
```
../ChangeFormer/checkpoints/ChangeFormer_LEVIR/best_ckpt.pt
```

## 🎯 API Endpoints

### Base URL: `http://localhost:5002`

### 1. ChangeFormer Only
**Endpoint**: `POST /changeformer`

**Input**: 2 image URLs
```json
{
    "image_a_url": "https://example.com/before.jpg",
    "image_b_url": "https://example.com/after.jpg"
}
```

**Output**:
```json
{
    "success": true,
    "result_url": "https://i.ibb.co/result.png",
    "image_a_url": "https://example.com/before.jpg",
    "image_b_url": "https://example.com/after.jpg"
}
```

### 2. Text Analysis Only
**Endpoint**: `POST /text-analysis`

**Input**: 3 image URLs (including change mask)
```json
{
    "image_a_url": "https://example.com/before.jpg",
    "image_b_url": "https://example.com/after.jpg",
    "change_mask_url": "https://example.com/change_mask.png"
}
```

**Output**:
```json
{
    "success": true,
    "analysis": "Detailed AI analysis of the changes...",
    "image_a_url": "https://example.com/before.jpg",
    "image_b_url": "https://example.com/after.jpg",
    "change_mask_url": "https://example.com/change_mask.png"
}
```

### 3. Combined (Recommended)
**Endpoint**: `POST /combined`

**Input**: 2 image URLs
```json
{
    "image_a_url": "https://example.com/before.jpg",
    "image_b_url": "https://example.com/after.jpg"
}
```

**Output**:
```json
{
    "success": true,
    "changeformer_result_url": "https://i.ibb.co/result.png",
    "text_analysis": "Detailed AI analysis of the changes...",
    "image_a_url": "https://example.com/before.jpg",
    "image_b_url": "https://example.com/after.jpg"
}
```

### 4. Health Check
**Endpoint**: `GET /health`

**Output**:
```json
{
    "status": "healthy",
    "service": "Combined ChangeFormer + Text Analysis API",
    "endpoints": {
        "changeformer": "/changeformer (POST) - 2 image URLs",
        "text_analysis": "/text_analysis (POST) - 3 image URLs",
        "combined": "/combined (POST) - 2 image URLs"
    }
}
```

## 🚀 Usage

### Start the API Server
```bash
cd TEXT-OUTPUT
python combined_api.py
```

### Test the API
```bash
python test_combined_api.py
```

### Example with cURL

#### ChangeFormer Only:
```bash
curl -X POST http://localhost:5002/changeformer \
  -H "Content-Type: application/json" \
  -d '{
    "image_a_url": "https://example.com/before.jpg",
    "image_b_url": "https://example.com/after.jpg"
  }'
```

#### Combined Analysis:
```bash
curl -X POST http://localhost:5002/combined \
  -H "Content-Type: application/json" \
  -d '{
    "image_a_url": "https://example.com/before.jpg",
    "image_b_url": "https://example.com/after.jpg"
  }'
```

## 📊 Response Analysis

The text analysis provides:
1. **What changed**: Buildings, roads, vegetation, etc.
2. **Scale of changes**: Small, medium, large areas
3. **Type of development**: Urban, agricultural, infrastructure
4. **Notable patterns**: Clusters and spatial distribution
5. **Change assessment**: Positive or negative impact

## 🔧 Configuration

### GPU/CPU Usage
Modify in `combined_api.py`:
```python
args.gpu_ids = '0'  # Use GPU 0, or '-1' for CPU
```

### Model Checkpoint Path
```python
args.checkpoint_root = '../ChangeFormer/checkpoints'
args.project_name = 'ChangeFormer_LEVIR'
```

## 🐛 Troubleshooting

### Common Issues:

1. **"API not initialized"**
   - Check your `.env` file has correct API keys
   - Ensure ChangeFormer checkpoint exists

2. **"ChangeFormer model not initialized"**
   - Verify checkpoint path: `../ChangeFormer/checkpoints/ChangeFormer_LEVIR/best_ckpt.pt`
   - Check PyTorch installation

3. **"ImgBB upload failed"**
   - Verify your ImgBB API key is valid
   - Check image format (PNG/JPG supported)

4. **CUDA errors**
   - Set `args.gpu_ids = '-1'` for CPU-only mode
   - Ensure PyTorch CUDA version matches your GPU drivers

## 📝 Notes

- Images are automatically resized to 256x256 for ChangeFormer processing
- Results are hosted on ImgBB for easy sharing
- The API handles image download and upload automatically
- All endpoints return JSON responses with consistent error handling

## 🔗 Related Files

- `combined_api.py`: Main API server
- `test_combined_api.py`: Test script
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (create this) 