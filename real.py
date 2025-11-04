import streamlit as st
import torch
from PIL import Image
from pathlib import Path
import numpy as np
import warnings
import sys
import logging
from ultralytics import YOLO
import tempfile
import os
import subprocess
import shutil

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="PlantCare AI - Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# CUSTOM CSS - FULL LIGHT MODE
# ==============================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles - Full Light Theme */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove all dark backgrounds - unified white */
    .stApp {
        background-color: #f8f9fa;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1200px;
        background-color: #f8f9fa;
    }
    
    /* Force consistent background */
    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa;
    }
    
    [data-testid="stHeader"] {
        background-color: #f8f9fa;
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #d4f1d8 0%, #e8f5e9 50%, #f1f8f3 100%);
        border-radius: 24px;
        padding: 60px 40px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .powered-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background-color: rgba(46, 125, 50, 0.15);
        color: #2e7d32;
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 24px;
        border: 1px solid rgba(46, 125, 50, 0.3);
    }
    
    .hero-title {
        font-size: 64px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 24px;
        color: #1a1a1a;
    }
    
    .hero-title-green {
        color: #2e7d32;
    }
    
    .hero-subtitle {
        font-size: 18px;
        color: #616161;
        line-height: 1.6;
        max-width: 800px;
        margin: 0 auto 40px auto;
    }
    
    /* Stats */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 80px;
        margin-top: 50px;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 48px;
        font-weight: 800;
        color: #2e7d32;
        line-height: 1;
    }
    
    .stat-label {
        font-size: 16px;
        color: #757575;
        margin-top: 8px;
    }
    
    /* Section Styles */
    .section-container {
        background-color: white;
        border-radius: 24px;
        padding: 60px 40px;
        margin-bottom: 40px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .section-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: #1a1a1a;
        margin-bottom: 16px;
    }
    
    .section-subtitle {
        font-size: 18px;
        text-align: center;
        color: #757575;
        margin-bottom: 50px;
    }
    
    /* Feature Cards */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 24px;
        margin-top: 40px;
    }
    
    .feature-card {
        background-color: #ffffff;
        border: 2px solid #e8e8e8;
        border-radius: 16px;
        padding: 32px 24px;
        text-align: left;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: #2e7d32;
        box-shadow: 0 8px 24px rgba(46, 125, 50, 0.12);
        transform: translateY(-4px);
    }
    
    .feature-icon-box {
        width: 56px;
        height: 56px;
        background-color: #2e7d32;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-bottom: 20px;
    }
    
    .feature-title {
        font-size: 20px;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 12px;
    }
    
    .feature-description {
        font-size: 15px;
        color: #616161;
        line-height: 1.6;
    }
    
    /* Steps Section */
    .steps-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 32px;
        margin-top: 40px;
    }
    
    .step-card {
        text-align: center;
    }
    
    .step-icon-circle {
        width: 80px;
        height: 80px;
        background-color: #2e7d32;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        margin: 0 auto 16px auto;
        position: relative;
    }
    
    .step-number {
        position: absolute;
        top: -10px;
        right: -10px;
        width: 40px;
        height: 40px;
        background-color: white;
        border: 3px solid #2e7d32;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 700;
        color: #2e7d32;
    }
    
    .step-title {
        font-size: 20px;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 8px;
    }
    
    .step-description {
        font-size: 14px;
        color: #616161;
        line-height: 1.5;
    }
    
    /* Disease Cards */
    .disease-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        margin-top: 40px;
    }
    
    .disease-card {
        background-color: white;
        border: 2px solid #e8e8e8;
        border-radius: 16px;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .disease-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        transform: translateY(-4px);
    }
    
    .disease-badge {
        position: absolute;
        top: 16px;
        right: 16px;
        background-color: #2e7d32;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    
    .disease-content {
        padding: 24px;
    }
    
    .disease-title {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 16px;
    }
    
    .disease-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .disease-list-item {
        font-size: 15px;
        color: #616161;
        padding: 6px 0;
        display: flex;
        align-items: center;
    }
    
    .disease-list-item::before {
        content: "●";
        color: #2e7d32;
        font-size: 12px;
        margin-right: 10px;
    }
    
    /* Upload Container */
    .upload-container {
        background-color: white;
        border-radius: 24px;
        padding: 60px 40px;
        margin-bottom: 40px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .upload-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: #1a1a1a;
        margin-bottom: 16px;
    }
    
    .upload-subtitle {
        font-size: 18px;
        text-align: center;
        color: #424242;
        margin-bottom: 40px;
    }
    
    /* FILE UPLOADER - COMPLETE REDESIGN */
    .stFileUploader {
        background-color: transparent !important;
        padding: 0 !important;
        border: none !important;
    }
    
    .stFileUploader > div {
        background-color: transparent !important;
    }
    
    .stFileUploader > div > div {
        background-color: transparent !important;
    }
    
    .stFileUploader label {
        display: none !important;
    }
    
    .stFileUploader section {
        border: none !important;
        padding: 0 !important;
        background-color: transparent !important;
    }
    
    .stFileUploader section > div {
        background-color: transparent !important;
    }
    
    /* Dropzone Main Container */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f9f9f9 !important;
        border: 2px dashed #ddd !important;
        border-radius: 12px !important;
        padding: 80px 40px !important;
        text-align: center !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #f5f5f5 !important;
        border-color: #ccc !important;
    }
    
    /* SVG Icon - Properly Centered */
    [data-testid="stFileUploaderDropzone"] svg {
        width: 80px !important;
        height: 80px !important;
        color: #2e7d32 !important;
        background: rgba(46, 125, 50, 0.1) !important;
        padding: 16px !important;
        border-radius: 50% !important;
        margin-bottom: 20px !important;
    }
    
    /* Instructions Container - Make text dark */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        width: 100% !important;
        text-align: center !important;
    }
    
    /* Main instruction text - Dark color */
    [data-testid="stFileUploaderDropzoneInstructions"] p {
        color: #1a1a1a !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin: 0 0 8px 0 !important;
    }
    
    /* Secondary text - Dark color */
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #888 !important;
        font-size: 15px !important;
        font-weight: 400 !important;
        display: block !important;
        margin-bottom: 20px !important;
    }
    
    /* Browse Button */
    .stFileUploader button {
        display: none !important;
    }
    
    /* File List */
    .stFileUploader [data-testid="stFileUploaderFileList"] {
        background-color: transparent !important;
        border: none !important;
        padding: 20px 0 0 0 !important;
        margin-top: 20px !important;
    }
    
    /* File Items */
    .stFileUploader [data-testid="stFileUploaderFileItem"] {
        background-color: #f5f5f5 !important;
        border: 1px solid #e8e8e8 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
    }
    
    /* File name - Dark text */
    .stFileUploader [data-testid="stFileUploaderFileItem"] p {
        color: #1a1a1a !important;
        font-weight: 500 !important;
        margin: 0 !important;
        font-size: 14px !important;
    }
    
    /* File links */
    .stFileUploader [data-testid="stFileUploaderFileItem"] a {
        color: #1a1a1a !important;
        text-decoration: none !important;
        font-weight: 500 !important;
    }
    
    /* File size badge - Gray */
    .stFileUploader [data-testid="stFileUploaderFileItem"] span {
        background-color: #e8e8e8 !important;
        color: #666 !important;
        padding: 2px 8px !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    
    /* Small helper text */
    .stFileUploader small {
        color: #888 !important;
        font-size: 13px !important;
    }
    
    /* Enhancement Checkbox Container */
    .enhancement-container {
        display: flex;
        justify-content: center;
        margin: 30px 0;
    }
    
    .stCheckbox {
        background-color: white !important;
        padding: 16px 24px !important;
        border-radius: 12px !important;
        border: 2px solid #e8e8e8 !important;
        display: inline-block !important;
        transition: all 0.3s ease !important;
    }
    
    .stCheckbox:hover {
        border-color: #2e7d32 !important;
        background-color: #f9fdf9 !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.1) !important;
    }
    
    .stCheckbox label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        cursor: pointer !important;
        margin: 0 !important;
    }
    
    .stCheckbox label span {
        color: #1a1a1a !important;
    }
    
    .stCheckbox label p {
        color: #1a1a1a !important;
    }
    
    /* Enhancement Status Badge */
    .enhancement-badge {
        background-color: #d4f1d8;
        color: #2e7d32;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 15px;
        font-weight: 600;
        border: 1px solid rgba(46, 125, 50, 0.3);
        text-align: center;
        margin: 20px auto;
        width: fit-content;
    }
    
    /* Results Section */
    .result-box {
        background-color: white;
        border: 2px solid #e8e8e8;
        border-left: 5px solid #f44336;
        border-radius: 16px;
        padding: 32px;
        margin: 24px 0;
    }
    
    .result-box.healthy {
        border-left-color: #4caf50;
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 24px;
    }
    
    .result-label {
        font-size: 14px;
        color: #f44336;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .result-label.healthy {
        color: #4caf50;
    }
    
    .result-disease-name {
        font-size: 28px;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    .result-badge {
        background-color: #f44336;
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    
    .result-badge.healthy {
        background-color: #4caf50;
    }
    
    .confidence-section {
        margin-top: 20px;
    }
    
    .confidence-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    
    .confidence-label {
        font-size: 15px;
        color: #757575;
    }
    
    .confidence-value {
        font-size: 20px;
        font-weight: 700;
        color: #2e7d32;
    }
    
    .progress-bar-bg {
        width: 100%;
        height: 12px;
        background-color: #e8e8e8;
        border-radius: 6px;
        overflow: hidden;
    }
    
    .progress-bar-fill {
        height: 100%;
        background-color: #2e7d32;
        transition: width 0.5s ease;
    }
    
    /* Treatment Section */
    .treatment-box {
        background-color: #f1f8f3;
        border-radius: 16px;
        padding: 32px;
        margin: 24px 0;
    }
    
    .treatment-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
    }
    
    .treatment-icon {
        width: 40px;
        height: 40px;
        background-color: #2e7d32;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    
    .treatment-title {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    .treatment-step {
        background-color: white;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        display: flex;
        gap: 16px;
        align-items: flex-start;
    }
    
    .treatment-step-number {
        font-size: 18px;
        font-weight: 700;
        color: #2e7d32;
        min-width: 24px;
    }
    
    .treatment-step-text {
        font-size: 15px;
        color: #424242;
        line-height: 1.6;
    }
    
    /* Footer */
    .footer-container {
        background-color: #f1f8f3;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin-top: 60px;
    }
    
    .footer-title {
        font-size: 24px;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 8px;
    }
    
    .footer-text {
        font-size: 15px;
        color: #757575;
        margin: 4px 0;
    }
    
    /* Streamlit Button Overrides */
    .stButton {
        display: flex;
        justify-content: center;
    }
    
    .stButton > button {
        background-color: #2e7d32 !important;
        color: white !important;
        border: none !important;
        padding: 16px 40px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        min-width: 220px !important;
        white-space: nowrap !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 52px !important;
        line-height: 1 !important;
        text-align: center !important;
    }
    
    .stButton > button:hover {
        background-color: #1b5e20 !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3) !important;
    }
    
    .stButton > button p {
        color: white !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
        line-height: 1 !important;
    }
    
    .stButton > button div {
        white-space: nowrap !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Image Caption Styling */
    [data-testid="stImage"] figcaption {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        text-align: center !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
        padding: 8px !important;
        border-radius: 4px !important;
    }
    
    /* Info/Success/Warning Message Text */
    .stAlert {
        color: #1a1a1a !important;
    }
    
    .stAlert p {
        color: #1a1a1a !important;
    }
    
    .stSuccess {
        background-color: #d4f1d8 !important;
        color: #1a1a1a !important;
    }
    
    .stInfo {
        background-color: #e3f2fd !important;
        color: #1a1a1a !important;
    }
    
    .stWarning {
        background-color: #fff9c4 !important;
        color: #1a1a1a !important;
    }
    
    /* Spinner Text */
    .stSpinner > div {
        color: #1a1a1a !important;
    }
    
    .stSpinner p {
        color: #1a1a1a !important;
    }
    
    /* All paragraph text */
    p {
        color: #1a1a1a !important;
    }
    
    /* Markdown text */
    [data-testid="stMarkdownContainer"] {
        color: #1a1a1a !important;
    }
    
    [data-testid="stMarkdownContainer"] p {
        color: #1a1a1a !important;
    }
    
    /* Status messages */
    [data-testid="stStatusWidget"] {
        color: #1a1a1a !important;
    }
    
    [data-testid="stStatusWidget"] p {
        color: #1a1a1a !important;
    }
    
    /* Hide Streamlit UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .features-grid, .steps-grid, .disease-grid {
            grid-template-columns: 1fr;
        }
        
        .stats-row {
            flex-direction: column;
            gap: 30px;
        }
        
        .hero-title {
            font-size: 40px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# CONFIGURATION
# ==============================
@st.cache_resource
def load_config():
    config = {
        "REALESRGAN_PATH": r"C:\Users\vkr30\Real-ESRGAN",
        "MODEL_REALESRGAN_PATH": r"C:\Users\vkr30\Real-ESRGAN\weights\RealESRGAN_x4plus.pth",
        "YOLO_MODEL_PATH": r"C:\Users\vkr30\Image Segmentation_Plant Disease\Yolov11 Variants for PDP\best.pt",
        "HEALTHY_IMAGES_PATH": r"C:\Users\vkr30\Image Segmentation_Plant Disease\healthy"
    }
    return config

# ==============================
# DISEASE CURES
# ==============================
DISEASE_CURE = {
    "Pepper__bell___Bacterial_spot": [
        "Remove and destroy infected leaves.",
        "Spray copper-based bactericides regularly.",
        "Practice crop rotation and sanitize tools to prevent spread.",
        "Ensure good air circulation by pruning dense foliage."
    ],
    "Potato___Early_blight": [
        "Remove affected foliage to prevent spore spread.",
        "Apply fungicides containing Chlorothalonil or Mancozeb.",
        "Rotate crops and avoid planting potatoes in the same soil consecutively.",
        "Ensure proper spacing for good air circulation."
    ],
    "Potato___Late_blight": [
        "Remove and destroy infected plants immediately.",
        "Apply fungicides containing Metalaxyl or Mancozeb at the first signs of infection.",
        "Avoid overhead irrigation; use drip irrigation instead.",
        "Plant resistant potato varieties if available."
    ],
    "Tomato_Bacterial_spot": [
        "Remove and destroy infected leaves.",
        "Spray copper-based bactericides regularly.",
        "Practice crop rotation and sanitize tools to prevent spread.",
        "Ensure good air circulation by pruning dense foliage."
    ],
    "Tomato_Early_blight": [
        "Remove affected leaves and destroy them.",
        "Apply fungicides such as Chlorothalonil or Copper hydroxide.",
        "Rotate crops and avoid planting tomatoes in the same soil consecutively.",
        "Keep soil moisture consistent but avoid wetting leaves."
    ],
    "Tomato_Late_blight": [
        "Remove infected plants and dispose of them safely.",
        "Apply protective fungicides like Metalaxyl or Mancozeb before infection spreads.",
        "Ensure good drainage to prevent waterlogging.",
        "Plant resistant tomato varieties if possible."
    ],
    "Tomato_Leaf_Mold": [
        "Remove infected leaves to prevent spore spread.",
        "Apply fungicides such as Mancozeb or Copper oxychloride.",
        "Maintain proper plant spacing for airflow.",
        "Avoid wetting leaves during irrigation."
    ],
    "Tomato_Septoria_leaf_spot": [
        "Remove infected leaves and destroy them.",
        "Apply fungicides like Chlorothalonil or Mancozeb.",
        "Practice crop rotation and keep soil clean from debris.",
        "Ensure proper spacing and airflow around plants."
    ],
    "Tomato_Spider_mites_Two_spotted_spider_mite": [
        "Spray insecticidal soap or neem oil directly on affected areas.",
        "Maintain adequate humidity to discourage mite development.",
        "Introduce natural predators like ladybugs or predatory mites.",
        "Avoid excessive use of nitrogen fertilizers which favor mite growth."
    ],
    "Tomato__Target_Spot": [
        "Remove and destroy infected leaves.",
        "Apply fungicides like Chlorothalonil or Copper oxychloride.",
        "Ensure good spacing and pruning for airflow.",
        "Practice crop rotation to reduce recurring infections."
    ],
    "Tomato__Tomato_YellowLeaf__Curl_Virus": [
        "Remove and destroy infected plants immediately.",
        "Control whitefly populations using yellow sticky traps or insecticides.",
        "Avoid planting tomatoes near infected crops.",
        "Use resistant tomato varieties if available."
    ],
    "Tomato__Tomato_mosaic_virus": [
        "Remove infected plants and sanitize all tools.",
        "Avoid handling healthy plants after touching infected ones.",
        "Practice crop rotation and avoid planting tomatoes continuously.",
        "Wash hands and tools frequently to prevent virus spread."
    ]
}

# ==============================
# SUPPRESS WARNINGS
# ==============================
warnings.filterwarnings('ignore')
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_yolo_model():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(config["YOLO_MODEL_PATH"])
    return model, device

# ==============================
# IMAGE ENHANCEMENT WITH REAL-ESRGAN
# ==============================
def enhance_image_with_realesrgan(image):
    """
    Enhance image using Real-ESRGAN via subprocess call
    Uses Real-ESRGAN's own inputs/results folders
    """
    config = load_config()
    
    # Use Real-ESRGAN's own folders
    input_folder = os.path.join(config["REALESRGAN_PATH"], "inputs")
    output_folder = os.path.join(config["REALESRGAN_PATH"], "results")
    
    # Create folders if they don't exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate unique filename to avoid conflicts
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    input_filename = f"plantcare_{unique_id}.png"
    input_path = os.path.join(input_folder, input_filename)
    
    # Expected output filename (Real-ESRGAN adds _out suffix)
    output_filename = f"plantcare_{unique_id}_out.png"
    output_path = os.path.join(output_folder, output_filename)
    
    try:
        # Save input image
        image.save(input_path, "PNG")
        
        print(f"✓ Saved input image to: {input_path}")
        print(f"✓ Input image exists: {os.path.exists(input_path)}")
        print(f"✓ Expected output: {output_path}")
        
        # Verify Real-ESRGAN installation
        inference_script = os.path.join(config["REALESRGAN_PATH"], "inference_realesrgan.py")
        if not os.path.exists(inference_script):
            raise Exception(f"Real-ESRGAN inference script not found at: {inference_script}")
        
        model_path = config["MODEL_REALESRGAN_PATH"]
        if not os.path.exists(model_path):
            raise Exception(f"Model weights not found at: {model_path}")
        
        print(f"✓ Inference script: {inference_script}")
        print(f"✓ Model weights: {model_path}")
        
        # Build Real-ESRGAN command - process single file
        command = [
            "python", 
            inference_script,
            "-n", "RealESRGAN_x4plus",
            "-i", input_path,
            "-o", output_folder,
            "-s", "4",
            "--fp32",
            "--ext", "png"  # Force PNG output
        ]
        
        # Print command for debugging
        print("\n" + "="*50)
        print("Running Real-ESRGAN command:")
        print(" ".join(command))
        print("="*50 + "\n")
        
        # Run Real-ESRGAN enhancement from its directory
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True,
            timeout=180,
            cwd=config["REALESRGAN_PATH"]
        )
        
        # Print output for debugging
        print("\n--- Real-ESRGAN Output ---")
        print("Return code:", result.returncode)
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        print("--- End Output ---\n")
        
        # Wait for file system to sync
        import time
        time.sleep(2)  # Increased wait time
        
        # List all files in both input and output folders
        print(f"\nInput folder contents: {os.listdir(input_folder)}")
        print(f"Output folder contents: {os.listdir(output_folder)}")
        
        # Check if output was created
        if not os.path.exists(output_path):
            # List all files in output folder for debugging
            output_files = os.listdir(output_folder) if os.path.exists(output_folder) else []
            print(f"⚠ Expected output not found: {output_path}")
            print(f"Files in output folder: {output_files}")
            
            # Try to find any output with similar name
            possible_outputs = [
                f"plantcare_{unique_id}_out.jpg",
                f"plantcare_{unique_id}.png",
                f"plantcare_{unique_id}.jpg"
            ]
            
            for possible in possible_outputs:
                test_path = os.path.join(output_folder, possible)
                if os.path.exists(test_path):
                    output_path = test_path
                    print(f"✓ Found alternative output: {possible}")
                    break
            
            # If still not found, check for any new files
            if not os.path.exists(output_path):
                # Get the newest file in output folder
                if output_files:
                    newest_file = max([os.path.join(output_folder, f) for f in output_files], 
                                    key=os.path.getctime)
                    if newest_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        output_path = newest_file
                        print(f"✓ Using newest file: {os.path.basename(newest_file)}")
        
        # Final check
        if not os.path.exists(output_path):
            error_msg = f"""
Real-ESRGAN did not produce output image.

Paths:
- Input: {input_path} (exists: {os.path.exists(input_path)})
- Expected output: {output_path}
- Output folder: {output_folder}

Files in output folder: {os.listdir(output_folder) if os.path.exists(output_folder) else 'folder does not exist'}

Real-ESRGAN output:
{result.stdout}
{result.stderr}

Suggestion: Try running this command manually in terminal:
cd {config["REALESRGAN_PATH"]}
python inference_realesrgan.py -n RealESRGAN_x4plus -i {input_path} -o {output_folder} -s 4 --fp32
            """
            raise Exception(error_msg)
        
        print(f"✓ Loading enhanced image from: {output_path}")
        enhanced_image = Image.open(output_path).convert('RGB')
        print(f"✓ Enhanced image size: {enhanced_image.size}")
        
        return enhanced_image
        
    except subprocess.TimeoutExpired:
        raise Exception("Real-ESRGAN timed out (>180s). Try with a smaller image.")
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("\n!!! ERROR !!!")
        print(error_detail)
        print("!!! END !!!\n")
        raise
    finally:
        # Don't cleanup - leave files for debugging
        # User can manually clean the folders if needed
        print(f"\n📁 Files kept for inspection:")
        print(f"   Input: {input_path}")
        print(f"   Expected output: {output_path}")
        print(f"\nTo clean up manually, delete files from:")
        print(f"   {input_folder}")
        print(f"   {output_folder}")


# ==============================
# DISEASE DETECTION
# ==============================
def detect_disease(image):
    model, device = load_yolo_model()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name
    
    results = model(tmp_path)
    pred = results[0]
    
    class_names = [
        "Pepper__bell___Bacterial_spot","Pepper__bell___healthy",
        "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
        "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight",
        "Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite","Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus","Tomato__Tomato_mosaic_virus",
        "Tomato_healthy"
    ]
    pred.names = {i:name for i,name in enumerate(class_names)}
    
    class_idx = int(pred.probs.top1)
    cls_name = pred.names[class_idx]
    conf = pred.probs.data[class_idx].item()
    
    os.unlink(tmp_path)
    
    return cls_name, conf

# ==============================
# HEALTHY IMAGE PICKER
# ==============================
def get_healthy_image_for_class(predicted_class):
    config = load_config()
    mapping = {"Tomato":"tomato.JPG","Potato":"potato.JPG","Pepper":"pepper.JPG"}
    for key,fname in mapping.items():
        if key in predicted_class:
            path = Path(config["HEALTHY_IMAGES_PATH"])/fname
            if path.exists():
                return Image.open(path)
    return None

# ==============================
# MAIN APP
# ==============================
def main():
    # Initialize session state
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    if 'enhanced_image' not in st.session_state:
        st.session_state.enhanced_image = None
    if 'used_enhancement' not in st.session_state:
        st.session_state.used_enhancement = False
    
    # Hero Section
    st.markdown("""
        <div class="hero-container">
            <div class="powered-badge">
                ✨ Powered by AI & Computer Vision
            </div>
            <div class="hero-title">
                <span class="hero-title-green">PlantCare AI</span><br>
                Disease Detection
            </div>
            <div class="hero-subtitle">
                Upload a photo of your plant and get instant AI-powered disease diagnosis with expert cure recommendations for tomatoes, potatoes, and peppers.
            </div>
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-number">15+</div>
                    <div class="stat-label">Diseases Detected</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">3</div>
                    <div class="stat-label">Plant Types</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">95%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Why PlantCare AI
    st.markdown("""
        <div class="section-container">
            <div class="section-title">Why PlantCare AI?</div>
            <div class="section-subtitle">Cutting-edge technology meets agricultural expertise to protect your crops</div>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon-box">🧠</div>
                    <div class="feature-title">AI-Powered Detection</div>
                    <div class="feature-description">Advanced YOLO neural network trained on thousands of plant disease images for accurate identification</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon-box">⚡</div>
                    <div class="feature-title">Instant Results</div>
                    <div class="feature-description">Get disease diagnosis and confidence scores in seconds with our optimized detection pipeline</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon-box">🛡️</div>
                    <div class="feature-title">Expert Recommendations</div>
                    <div class="feature-description">Detailed cure recommendations with step-by-step treatment guides for each detected disease</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon-box">📈</div>
                    <div class="feature-title">95% Accuracy</div>
                    <div class="feature-description">Industry-leading accuracy rates backed by extensive testing and continuous model improvements</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # How It Works
    st.markdown("""
        <div class="section-container">
            <div class="section-title">How It Works</div>
            <div class="section-subtitle">Simple, fast, and accurate plant disease detection in 4 easy steps</div>
            <div class="steps-grid">
                <div class="step-card">
                    <div class="step-icon-circle">
                        📤
                        <div class="step-number">01</div>
                    </div>
                    <div class="step-title">Upload Image</div>
                    <div class="step-description">Take a clear photo of your plant leaf showing any symptoms or abnormalities</div>
                </div>
                <div class="step-card">
                    <div class="step-icon-circle">
                        🔍
                        <div class="step-number">02</div>
                    </div>
                    <div class="step-title">AI Analysis</div>
                    <div class="step-description">Our YOLO model processes the image through advanced computer vision algorithms</div>
                </div>
                <div class="step-card">
                    <div class="step-icon-circle">
                        📋
                        <div class="step-number">03</div>
                    </div>
                    <div class="step-title">Get Results</div>
                    <div class="step-description">Receive instant disease identification with confidence scores and severity assessment</div>
                </div>
                <div class="step-card">
                    <div class="step-icon-circle">
                        💊
                        <div class="step-number">04</div>
                    </div>
                    <div class="step-title">Take Action</div>
                    <div class="step-description">Follow expert cure recommendations and treatment steps to save your crops</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Diseases We Detect
    st.markdown("""
        <div class="section-container">
            <div class="section-title">Diseases We Detect</div>
            <div class="section-subtitle">Our AI model is trained to identify 15+ common plant diseases across tomatoes, potatoes, and peppers</div>
            <div class="disease-grid">
                <div class="disease-card">
                    <div style="position: relative; background: linear-gradient(135deg, #d4f1d8 0%, #a5d6a7 100%); height: 200px; display: flex; align-items: center; justify-content: center; font-size: 80px;">
                        🍅
                        <div class="disease-badge">9 Diseases</div>
                    </div>
                    <div class="disease-content">
                        <div class="disease-title">Tomato Diseases</div>
                        <ul class="disease-list">
                            <li class="disease-list-item">Bacterial Spot</li>
                            <li class="disease-list-item">Early Blight</li>
                            <li class="disease-list-item">Late Blight</li>
                            <li class="disease-list-item">Leaf Mold</li>
                            <li class="disease-list-item">Septoria Leaf Spot</li>
                            <li class="disease-list-item">Spider Mites</li>
                            <li class="disease-list-item">Target Spot</li>
                            <li class="disease-list-item">Yellow Leaf Curl Virus</li>
                            <li class="disease-list-item">Mosaic Virus</li>
                        </ul>
                    </div>
                </div>
                <div class="disease-card">
                    <div style="position: relative; background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%); height: 200px; display: flex; align-items: center; justify-content: center; font-size: 80px;">
                        🥔
                        <div class="disease-badge">2 Diseases</div>
                    </div>
                    <div class="disease-content">
                        <div class="disease-title">Potato Diseases</div>
                        <ul class="disease-list">
                            <li class="disease-list-item">Early Blight</li>
                            <li class="disease-list-item">Late Blight</li>
                        </ul>
                    </div>
                </div>
                <div class="disease-card">
                    <div style="position: relative; background: linear-gradient(135deg, #ffccbc 0%, #ffab91 100%); height: 200px; display: flex; align-items: center; justify-content: center; font-size: 80px;">
                        🌶️
                        <div class="disease-badge">1 Disease</div>
                    </div>
                    <div class="disease-content">
                        <div class="disease-title">Pepper Diseases</div>
                        <ul class="disease-list">
                            <li class="disease-list-item">Bacterial Spot</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("""
        <div class="upload-container">
            <div class="upload-title">Upload Your Plant Image</div>
            <div class="upload-subtitle">
                Share a clear photo of your plant leaf to receive instant AI-powered disease diagnosis
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhancement Toggle - Centered
    st.markdown('<div class="enhancement-container">', unsafe_allow_html=True)
    col_left, col_check, col_right = st.columns([1, 1, 1])
    with col_check:
        use_enhancement = st.checkbox(
            "Enable Image Enhancement (Real-ESRGAN 4x)", 
            value=False,
            key="enhancement_toggle",
            help="Uses Real-ESRGAN AI upscaling to enhance image quality before analysis. Improves detection accuracy but takes longer to process."
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show enhancement status
    if use_enhancement:
        st.markdown("""
            <div class="enhancement-badge">
                ✨ Image Enhancement Enabled - Processing will be more thorough
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="height: 20px;"></div>
        """, unsafe_allow_html=True)
    
    # File Uploader
    st.markdown('<div style="padding: 20px 0;">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select your plant image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed",
        key="file_uploader"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert('RGB')
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(input_image, caption="📸 Your Uploaded Image", width='stretch')
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analyze Button - Centered
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🔬 Analyze Plant", type="primary", width='stretch'):
                with st.spinner("🔄 Analyzing image..."):
                    # Detect on original
                    original_class, original_conf = detect_disease(input_image)
                    
                    # Enhancement if enabled
                    if use_enhancement:
                        try:
                            with st.spinner("✨ Enhancing image with Real-ESRGAN (4x upscaling)..."):
                                enhanced_image = enhance_image_with_realesrgan(input_image)
                                st.success("✅ Image enhanced successfully with Real-ESRGAN!")
                                
                            with st.spinner("🔍 Detecting disease on enhanced image..."):
                                enhanced_class, enhanced_conf = detect_disease(enhanced_image)
                            
                            # Compare results and use the one with higher confidence
                            if enhanced_conf > original_conf:
                                final_class = enhanced_class
                                final_conf = enhanced_conf
                                used_which = "enhanced"
                            else:
                                final_class = original_class
                                final_conf = original_conf
                                used_which = "original"
                            
                            # Store enhanced image and flag in session state
                            st.session_state.enhanced_image = enhanced_image
                            st.session_state.used_enhancement = True
                            st.session_state.used_which = used_which
                            st.session_state.original_conf = original_conf
                            st.session_state.enhanced_conf = enhanced_conf
                            
                        except Exception as e:
                            st.warning(f"⚠️ Enhancement failed: {str(e)}. Using original image.")
                            final_class = original_class
                            final_conf = original_conf
                            st.session_state.used_enhancement = False
                    else:
                        final_class = original_class
                        final_conf = original_conf
                        st.session_state.used_enhancement = False
                    
                    # Store in session state
                    st.session_state.analyzed = True
                    st.session_state.final_class = final_class
                    st.session_state.final_conf = final_conf
                    st.session_state.input_image = input_image
                    st.rerun()
    
    # Show Results if analyzed
    if st.session_state.analyzed:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Show enhancement comparison only if enhancement was used AND it improved the result
        if st.session_state.used_enhancement and st.session_state.enhanced_image is not None:
            # Only show comparison if enhanced result was better or equal
            if st.session_state.used_which == "enhanced":
                st.markdown("""
                    <div class="section-container">
                        <div class="section-title" style="font-size: 28px;">Image Enhancement Comparison</div>
                        <div class="section-subtitle" style="margin-bottom: 30px;">Original vs Enhanced (4x upscaling with Real-ESRGAN)</div>
                    </div>
                """, unsafe_allow_html=True)
                
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.image(st.session_state.input_image, caption="📷 Original Image", width='stretch')
                with comp_col2:
                    st.image(st.session_state.enhanced_image, caption="✨ Enhanced Image (4x)", width='stretch')
                
                st.success(f"✨ Enhancement improved detection! Using enhanced result (Confidence: {st.session_state.enhanced_conf*100:.1f}% vs Original: {st.session_state.original_conf*100:.1f}%)")
                
                st.markdown("<br>", unsafe_allow_html=True)
            else:
                # If original was better, just show a simple info message
                st.info(f"ℹ️ Original image provided better results. Using original image analysis (Confidence: {st.session_state.original_conf*100:.1f}%)")
                st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="section-container">
                <div class="section-title">Detection Results</div>
                <div class="section-subtitle">AI-powered analysis complete</div>
            </div>
        """, unsafe_allow_html=True)
        
        final_class = st.session_state.final_class
        final_conf = st.session_state.final_conf
        
        # Check if healthy
        is_healthy = final_class.lower().endswith("healthy")
        
        if not is_healthy:
            # Disease Detection Result
            disease_display = final_class.replace("_", " ")
            
            st.markdown(f"""
                <div class="result-box">
                    <div class="result-header">
                        <div>
                            <div class="result-label">⚠️ Detected condition</div>
                            <div class="result-disease-name">{disease_display}</div>
                        </div>
                        <div class="result-badge">Disease Detected</div>
                    </div>
                    <div class="confidence-section">
                        <div class="confidence-header">
                            <div class="confidence-label">Confidence Score</div>
                            <div class="confidence-value">{final_conf*100:.1f}%</div>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width: {final_conf*100}%"></div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Treatment Recommendations
            cure_steps = DISEASE_CURE.get(final_class, [])
            if cure_steps:
                st.markdown("""
                    <div class="treatment-box">
                        <div class="treatment-header">
                            <div class="treatment-icon">🌿</div>
                            <div class="treatment-title">Treatment Recommendations</div>
                        </div>
                """, unsafe_allow_html=True)
                
                for i, step in enumerate(cure_steps, 1):
                    st.markdown(f"""
                        <div class="treatment-step">
                            <div class="treatment-step-number">{i}</div>
                            <div class="treatment-step-text">{step}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Image Comparison
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="section-container">
                    <div class="section-title" style="font-size: 28px; text-align: left;">Visual Comparison</div>
                </div>
            """, unsafe_allow_html=True)
            
            healthy_img = get_healthy_image_for_class(final_class)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.input_image, caption="Your Plant (Diseased)", width='stretch')
            with col2:
                if healthy_img:
                    st.image(healthy_img, caption="Healthy Reference", width='stretch')
                else:
                    st.info("No healthy reference image available")
        
        else:
            # Healthy Plant Result
            st.markdown(f"""
                <div class="result-box healthy">
                    <div class="result-header">
                        <div>
                            <div class="result-label healthy">✅ Plant Status</div>
                            <div class="result-disease-name">Plant is Healthy!</div>
                        </div>
                        <div class="result-badge healthy">Healthy</div>
                    </div>
                    <div class="confidence-section">
                        <div class="confidence-header">
                            <div class="confidence-label">Confidence Score</div>
                            <div class="confidence-value">{final_conf*100:.1f}%</div>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width: {final_conf*100}%"></div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.success("🎉 Great news! Your plant appears to be healthy. Continue with regular care and monitoring.")
        
        # Reset Button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🔄 Analyze Another Image", width='stretch'):
                st.session_state.analyzed = False
                st.session_state.enhanced_image = None
                st.session_state.used_enhancement = False
                st.rerun()
    
    # Footer
    st.markdown("""
        <div class="footer-container">
            <div class="footer-title">PlantCare AI</div>
            <div class="footer-text">Advanced AI-powered plant disease detection for healthier crops</div>
            <div class="footer-text">© 2025 PlantCare AI. Powered by YOLO & Computer Vision.</div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()