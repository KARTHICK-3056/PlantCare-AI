# 🌿 PlantCare AI - Disease Detection System

## Overview

PlantCare AI is an advanced plant disease detection system powered by artificial intelligence and computer vision. It uses YOLO (You Only Look Once) for disease classification and Real-ESRGAN for optional image enhancement to improve detection accuracy.

### Key Features

- **AI-Powered Detection**: Identifies 15+ plant diseases across tomatoes, potatoes, and peppers
- **Image Enhancement**: Optional Real-ESRGAN 4x upscaling for improved accuracy
- **Real-Time Analysis**: Instant disease diagnosis with confidence scores
- **Treatment Recommendations**: Expert cure suggestions for each detected disease
- **User-Friendly Interface**: Beautiful, modern web interface built with Streamlit

---

## 📦 Package Contents

After extraction, you should see the following structure:

```
PlantCare-AI/
├── Real-ESRGAN/              # Image enhancement module
├── Image Segmentation_Plant Disease/  # YOLO model and data
├── real.py                   # Main application file
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Installation Guide

### Prerequisites

- **Python 3.8 or higher** (Python 3.9-3.10 recommended)
- **pip** package manager
- **NVIDIA GPU** (optional, but recommended for faster processing)
- **CUDA Toolkit** (if using GPU)

### Step 1: Extract the ZIP File

Extract the downloaded ZIP file to your desired location:

```bash
# Example location
C:\PlantCare-AI\
```

### Step 2: Install Python Dependencies

Open a terminal/command prompt in the extracted folder and run:

```bash
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes depending on your internet connection.

### Step 3: Setup Real-ESRGAN

Navigate to the Real-ESRGAN directory and install it in development mode:

```bash
cd Real-ESRGAN
python setup.py develop
cd ..
```

This command installs Real-ESRGAN and creates necessary links for the enhancement module.

### Step 4: Run the Application

From the root directory, launch the Streamlit application:

```bash
streamlit run real.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

---

## 🛠️ Troubleshooting

### Issue 1: Import Error with `rgb_to_grayscale`

**Error Message:**
```
ImportError: cannot import name 'rgb_to_grayscale' from 'torchvision.transforms.functional_tensor'
```

**Solution:**

This error occurs due to changes in torchvision's API. Follow these steps:

1. **Locate the file**: Navigate to your Python environment's site-packages folder:
   - **Anaconda**: `C:\ProgramData\anaconda3\envs\[ENV_NAME]\Lib\site-packages\basicsr\data\degradations.py`
   - **Regular Python**: `C:\Python3X\Lib\site-packages\basicsr\data\degradations.py`
   - **Virtual Environment**: `[YOUR_VENV_PATH]\Lib\site-packages\basicsr\data\degradations.py`

2. **Edit the file**: Open `degradations.py` in a text editor

3. **Find and replace**:
   
   Replace:
   ```python
   from torchvision.transforms.functional_tensor import rgb_to_grayscale
   ```
   
   With:
   ```python
   from torchvision.transforms.functional import rgb_to_grayscale
   ```

4. **Save the file** and restart the application

**Alternative Solution (if you can't find the file):**

```bash
# Find the exact path
python -c "import basicsr; print(basicsr.__file__)"

# This will show you the exact location of basicsr installation
```

### Issue 2: CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Disable image enhancement (uncheck the enhancement toggle)
- Use smaller images (resize before upload)
- Close other GPU-intensive applications
- Use CPU mode by setting `device = 'cpu'` in the code

### Issue 3: Model Files Not Found

**Error Message:**
```
FileNotFoundError: YOLO model not found
```

**Solution:**

Ensure all paths in `real.py` are correct. Update the configuration section:

```python
config = {
    "REALESRGAN_PATH": r"C:\PlantCare-AI\Real-ESRGAN",
    "MODEL_REALESRGAN_PATH": r"C:\PlantCare-AI\Real-ESRGAN\weights\RealESRGAN_x4plus.pth",
    "YOLO_MODEL_PATH": r"C:\PlantCare-AI\Image Segmentation_Plant Disease\Yolov11 Variants for PDP\best.pt",
    "HEALTHY_IMAGES_PATH": r"C:\PlantCare-AI\Image Segmentation_Plant Disease\healthy"
}
```

### Issue 4: Streamlit Command Not Found

**Error Message:**
```
'streamlit' is not recognized as an internal or external command
```

**Solution:**

```bash
# Reinstall streamlit
pip install --upgrade streamlit

# Or use python -m
python -m streamlit run real.py
```

---

## 💻 System Requirements

### Minimum Requirements

- **CPU**: Intel Core i5 or AMD equivalent
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8+

### Recommended Requirements

- **CPU**: Intel Core i7 or AMD Ryzen 7
- **RAM**: 16 GB or more
- **GPU**: NVIDIA GPU with 6GB+ VRAM (GTX 1060 or better)
- **Storage**: 10 GB free space (SSD recommended)
- **OS**: Windows 10/11 with CUDA support

---

## 📖 Usage Guide

### Basic Usage

1. **Launch the application**
   ```bash
   streamlit run real.py
   ```

2. **Upload an image**
   - Click the upload area or drag-and-drop
   - Supported formats: JPG, JPEG, PNG
   - Recommended: Clear leaf images with visible symptoms

3. **Optional: Enable Enhancement**
   - Check the "Enable Image Enhancement" box
   - This uses Real-ESRGAN for 4x upscaling
   - Takes longer but may improve accuracy

4. **Analyze**
   - Click "🔬 Analyze Plant"
   - Wait for results (5-30 seconds depending on enhancement)

5. **View Results**
   - Disease name and confidence score
   - Treatment recommendations
   - Visual comparison with healthy plant

### Tips for Best Results

- ✅ Use clear, well-lit images
- ✅ Focus on leaves showing symptoms
- ✅ Avoid blurry or low-resolution images
- ✅ Capture single leaves when possible
- ✅ Enable enhancement for low-quality images

---

## 🧪 Detected Diseases

### Tomato (9 diseases)
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus

### Potato (2 diseases)
- Early Blight
- Late Blight

### Pepper (1 disease)
- Bacterial Spot

---

## 🔧 Advanced Configuration

### GPU Configuration

To force GPU or CPU usage, modify the device configuration in `real.py`:

```python
# Force CPU
device = torch.device('cpu')

# Force GPU
device = torch.device('cuda')

# Auto-detect (recommended)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Port Configuration

To change the default port (8501):

```bash
streamlit run real.py --server.port 8080
```

### Memory Optimization

For systems with limited RAM:

```bash
streamlit run real.py --server.maxUploadSize 5
```

---

## 📝 Environment Setup (Alternative)

### Using Anaconda

```bash
# Create new environment
conda create -n plantcare python=3.9

# Activate environment
conda activate plantcare

# Install dependencies
pip install -r requirements.txt

# Setup Real-ESRGAN
cd Real-ESRGAN
python setup.py develop
cd ..

# Run application
streamlit run real.py
```

### Using Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup Real-ESRGAN
cd Real-ESRGAN
python setup.py develop
cd ..

# Run application
streamlit run real.py
```

---

## 🐛 Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Module not found | Missing dependencies | Run `pip install -r requirements.txt` |
| CUDA error | GPU driver issues | Update NVIDIA drivers or use CPU mode |
| Port already in use | Another app using port 8501 | Use `--server.port 8080` flag |
| Model not loading | Incorrect paths | Update paths in configuration section |
| Out of memory | Large image/GPU limit | Resize image or disable enhancement |

---

## 📧 Support & Contact

For issues, questions, or support, please contact the authors:

### Authors

**Vijaya Karthick**
- 📧 Email: vkr3056@gmail.com
- 📸 Instagram: [@karthickxviii](https://instagram.com/karthickxviii)
- 💻 GitHub: [KARTHICK-3056](https://github.com/KARTHICK-3056)

**Divyesh Hari**
- 📧 Email: divyesh02208@gmail.com
- 💻 GitHub: [DIVYESH-HARI](https://github.com/DIVYESH-HARI)

**Vishnu Vardhan**
- 📧 Email: skvishnu2006@gmail.com

---

## 🙏 Acknowledgments

- **YOLO**: Ultralytics for the object detection framework
- **Real-ESRGAN**: Xintao Wang et al. for image enhancement
- **Streamlit**: For the amazing web framework
- **Contributors**: All developers who contributed to this project

---

## 🔄 Version History

- **v1.0.0** (2025-01-01): Initial release
  - YOLO-based disease detection
  - Real-ESRGAN image enhancement
  - 15+ disease classifications
  - Treatment recommendations

---

**Made with ❤️ for healthier crops**