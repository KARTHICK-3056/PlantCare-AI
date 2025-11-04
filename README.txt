================================================================================
                    PLANTCARE AI - DISEASE DETECTION SYSTEM
                          Installation & Setup Guide
================================================================================

OVERVIEW
--------
PlantCare AI is an advanced plant disease detection system that uses:
- YOLO (You Only Look Once) for disease classification
- Real-ESRGAN for optional image enhancement
- Streamlit for the web interface

Detects 15+ diseases across tomatoes, potatoes, and peppers with expert
treatment recommendations.

================================================================================
SYSTEM REQUIREMENTS
================================================================================

MINIMUM:
- Python 3.12 or higher
- 8 GB RAM
- 5 GB free disk space
- Windows 10/11, Linux, or macOS

RECOMMENDED:
- Python 3.12
- 16 GB RAM
- NVIDIA GPU with 6GB+ VRAM
- 10 GB free disk space (SSD preferred)

================================================================================
INSTALLATION STEPS
================================================================================

STEP 1: EXTRACT THE ZIP FILE
-----------------------------
Extract the downloaded ZIP file to your desired location.

Example:
    C:\PlantCare-AI\

After extraction, you should see:
    - Real-ESRGAN/                    (Image enhancement module)
    - Image Segmentation_Plant Disease/  (YOLO model and data)
    - real.py                         (Main application file)
    - requirements.txt                (Python dependencies)
    - README.txt                      (This file)


STEP 2: INSTALL DEPENDENCIES
-----------------------------
Open Command Prompt or Terminal in the extracted folder and run:

    pip install -r requirements.txt

This will install all required Python packages.
NOTE: This may take 5-10 minutes depending on your internet connection.


STEP 3: SETUP REAL-ESRGAN
--------------------------
Navigate to the Real-ESRGAN directory and install it:

    cd Real-ESRGAN
    python setup.py develop
    cd ..

This installs Real-ESRGAN in development mode.


STEP 4: RUN THE APPLICATION
----------------------------
From the root directory, launch the application:

    streamlit run real.py

The application will open automatically in your web browser at:
    http://localhost:8501

================================================================================
TROUBLESHOOTING
================================================================================

ISSUE 1: ImportError with rgb_to_grayscale
-------------------------------------------
ERROR MESSAGE:
    ImportError: cannot import name 'rgb_to_grayscale' from 
    'torchvision.transforms.functional_tensor'

SOLUTION:
1. Locate your Python environment's site-packages folder:
   
   FOR ANACONDA:
   C:\ProgramData\anaconda3\envs\[YOUR_ENV_NAME]\Lib\site-packages\
   
   FOR REGULAR PYTHON:
   C:\Python3X\Lib\site-packages\
   
   FOR VIRTUAL ENVIRONMENT:
   [YOUR_VENV_PATH]\Lib\site-packages\

2. Navigate to: basicsr\data\degradations.py

3. Open the file in a text editor (Notepad, VSCode, etc.)

4. Find the line:
   from torchvision.transforms.functional_tensor import rgb_to_grayscale

5. Replace it with:
   from torchvision.transforms.functional import rgb_to_grayscale

6. Save the file and restart the application

ALTERNATIVE (Find exact path):
    python -c "import basicsr; print(basicsr.__file__)"


ISSUE 2: Streamlit Command Not Found
-------------------------------------
ERROR MESSAGE:
    'streamlit' is not recognized as an internal or external command

SOLUTION:
    pip install --upgrade streamlit
    
OR use:
    python -m streamlit run real.py


ISSUE 3: Model Files Not Found
-------------------------------
ERROR MESSAGE:
    FileNotFoundError: YOLO model not found

SOLUTION:
Ensure all paths in real.py are correct. Open real.py and verify the
configuration section matches your installation path:

    config = {
        "REALESRGAN_PATH": r"C:\PlantCare-AI\Real-ESRGAN",
        "MODEL_REALESRGAN_PATH": r"C:\PlantCare-AI\Real-ESRGAN\weights\RealESRGAN_x4plus.pth",
        "YOLO_MODEL_PATH": r"C:\PlantCare-AI\Image Segmentation_Plant Disease\Yolov11 Variants for PDP\best.pt",
        "HEALTHY_IMAGES_PATH": r"C:\PlantCare-AI\Image Segmentation_Plant Disease\healthy"
    }


ISSUE 4: CUDA Out of Memory
----------------------------
ERROR MESSAGE:
    RuntimeError: CUDA out of memory

SOLUTIONS:
- Disable image enhancement (uncheck the enhancement toggle in the app)
- Upload smaller images (resize before upload)
- Close other GPU-intensive applications
- Use CPU mode instead of GPU


ISSUE 5: Port Already in Use
-----------------------------
ERROR MESSAGE:
    Address already in use

SOLUTION:
Use a different port:
    streamlit run real.py --server.port 8080

================================================================================
USAGE GUIDE
================================================================================

BASIC USAGE:
1. Launch: streamlit run real.py
2. Upload an image (JPG, JPEG, or PNG)
3. Optional: Enable enhancement for low-quality images
4. Click "Analyze Plant"
5. View results and treatment recommendations

TIPS FOR BEST RESULTS:
- Use clear, well-lit images
- Focus on leaves showing symptoms
- Avoid blurry or low-resolution images
- Enable enhancement only for poor quality images

================================================================================
DETECTED DISEASES
================================================================================

TOMATO (9 diseases):
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites (Two-spotted spider mite)
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus

POTATO (2 diseases):
- Early Blight
- Late Blight

PEPPER (1 disease):
- Bacterial Spot

================================================================================
ALTERNATIVE SETUP WITH ANACONDA
================================================================================

CREATE ENVIRONMENT:
    conda create -n plantcare python=3.12
    conda activate plantcare

INSTALL DEPENDENCIES:
    pip install -r requirements.txt

SETUP REAL-ESRGAN:
    cd Real-ESRGAN
    python setup.py develop
    cd ..

RUN APPLICATION:
    streamlit run real.py

================================================================================
ALTERNATIVE SETUP WITH VIRTUAL ENVIRONMENT
================================================================================

CREATE VIRTUAL ENVIRONMENT:
    python -m venv venv

ACTIVATE (Windows):
    venv\Scripts\activate

ACTIVATE (Linux/Mac):
    source venv/bin/activate

INSTALL DEPENDENCIES:
    pip install -r requirements.txt

SETUP REAL-ESRGAN:
    cd Real-ESRGAN
    python setup.py develop
    cd ..

RUN APPLICATION:
    streamlit run real.py

================================================================================
ADVANCED CONFIGURATION
================================================================================

CHANGE PORT:
    streamlit run real.py --server.port 8080

LIMIT UPLOAD SIZE (in MB):
    streamlit run real.py --server.maxUploadSize 5

FORCE CPU MODE:
    Edit real.py and change:
    device = torch.device('cpu')

FORCE GPU MODE:
    Edit real.py and change:
    device = torch.device('cuda')

================================================================================
COMMON ERROR SOLUTIONS
================================================================================

ERROR                          | SOLUTION
-------------------------------|------------------------------------------
Module not found               | Run: pip install -r requirements.txt
CUDA error                    D| Update NVIDIA drivers or use CPU mode
Port already in use            | Use: --server.port 8080
Model not loading              | Check and update paths in real.py
Out of memory                  | Disable enhancement or resize images
Streamlit not found            | Run: pip install --upgrade streamlit
Import error (rgb_to_grayscale)| Edit degradations.py as shown above

================================================================================
SUPPORT & CONTACT
================================================================================

For issues, questions, or support, please contact the authors:

AUTHORS:
Vijaya Karthick
- Email: vkr3056@gmail.com
- Instagram: @karthickxviii
- GitHub: https://github.com/KARTHICK-3056

Divyesh Hari
- Email: divyesh02208@gmail.com

Vishnu Vardhan
- Email: skvishnu2006@gmail.com

================================================================================
VERSION INFORMATION
================================================================================

Version: 1.0.0
Release Date: January 1, 2025

FEATURES:
- YOLO-based disease detection
- Real-ESRGAN 4x image enhancement
- 15+ disease classifications
- Expert treatment recommendations
- Modern web interface with Streamlit

================================================================================
LICENSE
================================================================================

This project is licensed under the MIT License.
See LICENSE file for complete details.

================================================================================
ACKNOWLEDGMENTS
================================================================================

- Ultralytics: YOLO object detection framework
- Xintao Wang et al.: Real-ESRGAN image enhancement
- Streamlit: Web application framework
- All contributors to this project

================================================================================

Made with care for healthier crops.

================================================================================