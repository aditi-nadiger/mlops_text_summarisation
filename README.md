# MLOPS Text Summarization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-20.10.0-blue?logo=docker)
![Render](https://img.shields.io/badge/Render-Cloud%20Hosting-46E3B7?logo=render)
![spaCy](https://img.shields.io/badge/spaCy-3.5.0-09A3D5)


A powerful text summarization tool that extracts key information from various sources using TF-IDF algorithm, deployed with CI/CD pipelines and MLOps best practices.

## 🛠️ Project Structure

```text
MLOPS_TEXT_SUMMARISATION/
├── .github/
│   └── workflows/
│       ├── build-push-docker.yml    # Docker build/push automation
│       └── render-cd.yml            # Render deployment automation
├── Logs/
│   └── app.log                     # Application logs
├── Scripts/
│   ├── app.py                      # Main application with logging
│   └── Text-Summarizer.py          # Original implementation
├── venv/                           # Virtual environment
├── .gitignore
├── Dockerfile                      # Container configuration
├── LICENSE
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```
## 🔧 CI/CD Pipeline

1. **On push to main branch**:
   - Automatically builds Docker image
   - Pushes to Docker Hub

2. **Manual/Scheduled Deployment**:
   - Triggers Render.com deployment via webhook
   - Can be scheduled (every 10 minutes) or manually triggered

## 🌟 Features

- **Multiple Input Sources**:
  - Direct text input
  - Upload TXT/PDF files
  - Wikipedia URL scraping
- **Advanced NLP Processing**:
  - spaCy for sentence segmentation
  - NLTK for lemmatization
  - TF-IDF algorithm for key sentence extraction
- **Deployment Ready**:
  - Docker containerization
  - GitHub Actions CI/CD pipelines
  - Render.com deployment
- **Comprehensive Logging**:
  - Detailed application logs
  - Error tracking
- **Performance Metrics**:
  - Original vs. summary word count
  - Quality threshold adjustment

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (for container deployment)

### Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aditi-nadiger/mlops_text_summarisation.git
   cd MLOPS_TEXT_SUMMARISATION

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # For Linux/Mac/Git Bash:
   source venv/bin/activate
   # For Windows:
   venv\Scripts\activate

3. Install dependencies:
   ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

4. Run the application:

   ```bash
    streamlit run Scripts/app.py

