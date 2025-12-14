# Human Activity Recognition Web Application

A full-stack machine learning web application for recognizing human activities from smartphone accelerometer data. This project implements multiple ML algorithms (KNN, SVM, Logistic Regression, Random Forest, Decision Tree) with a FastAPI backend and interactive web frontend.

## 🎓 University Project

This is a complete implementation of a Human Activity Recognition system using the UCI HAR Dataset, featuring both exploratory data analysis (Jupyter notebook) and a production-ready web application.

## 🌟 Features

- **Interactive Web Interface**: Modern, responsive UI for training models and making predictions
- **Multiple ML Models**: Compare performance across 5 different classification algorithms
- **Real-time Training**: Train models directly from the web interface
- **Live Predictions**: Make activity predictions using any trained model
- **Performance Metrics**: View accuracy, precision, recall, and F1-score for all models
- **RESTful API**: Complete FastAPI backend with documented endpoints

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 2GB free disk space (for dataset)
- Internet connection (for initial dataset download)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

The application requires the UCI HAR Dataset. Download it using:

```bash
# On Linux/Mac
wget https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
unzip human_activity_recognition_using_smartphones.zip
unzip "UCI HAR Dataset.zip"

# On Windows (PowerShell)
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip" -OutFile "dataset.zip"
Expand-Archive -Path "dataset.zip" -DestinationPath "."
Expand-Archive -Path "UCI HAR Dataset.zip" -DestinationPath "."
```

Alternatively, manually download from: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Ensure the extracted folder is named `UCI HAR Dataset` in the project root.

### 3. Run the Application

```bash
# From the project root directory
python backend/main.py
```

Or using uvicorn directly:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the Application

Open your web browser and navigate to:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## 📁 Project Structure

```
Human_Activity_Recognition/
│
├── backend/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   └── saved_models/            # Trained models (created after training)
│   └── utils/
│       ├── __init__.py
│       ├── data_processor.py        # Data loading and preprocessing
│       └── model_trainer.py         # Model training and evaluation
│
├── templates/
│   └── index.html                   # Frontend web interface
│
├── static/                          # Static files (if any)
│
├── UCI HAR Dataset/                 # Dataset (download separately)
│   ├── train/
│   ├── test/
│   ├── activity_labels.txt
│   └── features.txt
│
├── Human_Activity_Recognition.ipynb # Original Jupyter notebook
├── requirements.txt                 # Python dependencies
├── CLAUDE.md                        # Development guide
└── README.md                        # This file
```

## 🎯 How to Use

### Step 1: Load Dataset
1. Open the web interface at http://localhost:8000
2. Click "Load Dataset" button
3. Wait for the dataset to load (shows info about samples and features)

### Step 2: Train Models
1. Click "Train All Models" button
2. Wait for training to complete (progress bar shows status)
3. View training completion message

### Step 3: View Performance Metrics
1. Click "View Metrics" button
2. Compare performance across all models
3. Best model is highlighted with a badge

### Step 4: Make Predictions
1. Select a model from the dropdown
2. Click "Predict Activity" button
3. View predicted activity and confidence score

## 🔌 API Endpoints

### Dataset Operations
- `GET /api/health` - Health check and dataset availability
- `GET /api/dataset/info` - Get dataset information
- `POST /api/dataset/load` - Load the dataset
- `GET /api/activities` - Get list of activity labels

### Model Operations
- `POST /api/train` - Train all models or specific models
- `GET /api/train/status` - Get training status
- `GET /api/models` - Get available/trained models
- `GET /api/models/best` - Get best performing model
- `GET /api/models/{model_name}/metrics` - Get metrics for specific model

### Predictions
- `POST /api/predict` - Make activity predictions
- `GET /api/metrics/summary` - Get summary of all metrics

### Example API Usage

```python
import requests

# Load dataset
response = requests.post("http://localhost:8000/api/dataset/load")
print(response.json())

# Train models
response = requests.post("http://localhost:8000/api/train")
print(response.json())

# Make prediction
features = [[0.5] * 561]  # Example features
response = requests.post(
    "http://localhost:8000/api/predict",
    json={"features": features, "model_name": "Random Forest"}
)
print(response.json())
```

## 📊 Dataset Information

- **Name**: UCI Human Activity Recognition Using Smartphones Dataset
- **Samples**: 10,299 total (7,352 training + 2,947 test)
- **Features**: 561 time and frequency domain variables
- **Classes**: 6 activities
  - WALKING
  - WALKING_UPSTAIRS
  - WALKING_DOWNSTAIRS
  - SITTING
  - STANDING
  - LAYING
- **Subjects**: 30 volunteers (age 19-48)

## 🤖 Machine Learning Models

The application implements and compares 5 classification algorithms:

1. **K-Nearest Neighbors (KNN)**
   - Fast predictions
   - No training phase
   - Good for small datasets

2. **Support Vector Machine (SVM)**
   - Effective in high dimensional spaces
   - Memory efficient
   - Slower training time

3. **Logistic Regression**
   - Fast and simple
   - Interpretable results
   - Good baseline model

4. **Random Forest**
   - Ensemble method
   - Handles overfitting well
   - Feature importance available

5. **Decision Tree**
   - Easy to interpret
   - Fast predictions
   - Prone to overfitting

## 📈 Expected Performance

Based on validation set (1,471 samples):

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | ~98.1% | ~98.1% |
| Logistic Regression | ~98.1% | ~98.1% |
| SVM | ~97.3% | ~97.3% |
| KNN | ~96.6% | ~96.6% |
| Decision Tree | ~94.2% | ~94.2% |

## 🛠️ Development

### Running in Development Mode

```bash
# With auto-reload
uvicorn backend.main:app --reload

# With specific host and port
uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload
```

### Using the Jupyter Notebook

```bash
jupyter notebook Human_Activity_Recognition.ipynb
```

The notebook contains the complete data analysis and model development process.

## 🐛 Troubleshooting

### Dataset Not Found Error
- Ensure `UCI HAR Dataset` folder exists in project root
- Check folder structure matches expected format
- Verify all required files are present (X_train.txt, y_train.txt, etc.)

### Import Errors
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)
- Use virtual environment

### Port Already in Use
```bash
# Use a different port
uvicorn backend.main:app --port 8080
```

### Models Not Training
- Ensure dataset is loaded first
- Check available memory (>2GB recommended)
- Review console logs for specific errors

## 📝 Assignment Requirements Met

This project fulfills all university project requirements:

✅ Dataset loading and preprocessing
✅ Missing value handling
✅ Data normalization (pre-normalized in dataset)
✅ Train/validation/test split
✅ Implementation of required algorithms:
  - KNN
  - SVM
  - Logistic Regression
  - Random Forest
  - Decision Tree
✅ Evaluation metrics (Accuracy, F1, Precision, Recall)
✅ Model comparison and analysis
✅ Visualization of results
✅ Complete documentation

## 🎨 Frontend Features

- Responsive design (mobile-friendly)
- Real-time status updates
- Interactive model training
- Dynamic metrics display
- Activity predictions with confidence scores
- Modern gradient UI design
- Progress indicators
- Error handling and user feedback

## 🔒 Notes

- Dataset is not included in repository (must be downloaded separately)
- Trained models are saved in `backend/models/saved_models/`
- First training may take 2-5 minutes depending on hardware
- SVM training is the slowest (enable probability for predict_proba)

## 📚 Technologies Used

### Backend
- **FastAPI**: Modern web framework for APIs
- **Uvicorn**: ASGI server
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Joblib**: Model persistence

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling with gradients and animations
- **Vanilla JavaScript**: Interactivity and API calls
- **Fetch API**: HTTP requests

## 👥 Contributors

University Project - Fall 2025

## 📄 License

This project is created for educational purposes.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the HAR dataset
- Scikit-learn for ML algorithms
- FastAPI for the excellent web framework
