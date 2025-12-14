# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Human Activity Recognition (HAR) machine learning project using the UCI Human Activity Recognition Using Smartphones Dataset. The project includes:

1. **Jupyter Notebook**: Complete data analysis and ML model development
2. **Full-Stack Web Application**: FastAPI backend + HTML/CSS/JS frontend for interactive model training and predictions

The project implements and compares multiple classification algorithms to predict human activities (walking, walking upstairs, walking downstairs, sitting, standing, laying) from smartphone accelerometer and gyroscope data.

## Dataset Information

- **Dataset**: UCI Human Activity Recognition Using Smartphones Dataset
- **Source**: https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
- **Structure**:
  - Training set: 7352 samples with 561 features
  - Test set: 2947 samples with 561 features
  - 6 activity classes
  - Features are pre-normalized (range -1 to 1)
- **Note**: Dataset contains 42 duplicate feature names that need to be resolved by appending numerical suffixes

## Running the Application

### Option 1: Web Application (Recommended)

**Quick Start:**
```bash
# Windows
run_app.bat

# Linux/Mac
chmod +x run_app.sh
./run_app.sh
```

**Manual Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python backend/main.py

# Or with uvicorn
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser.

### Option 2: Jupyter Notebook

For data analysis and development:

1. Open the notebook:
   ```bash
   jupyter notebook Human_Activity_Recognition.ipynb
   ```

2. Or use JupyterLab:
   ```bash
   jupyter lab Human_Activity_Recognition.ipynb
   ```

3. Run all cells sequentially from top to bottom, as later cells depend on variables/data from earlier cells.

## Project Workflow

The notebook follows this structure:

1. **Dataset Acquisition**: Downloads and extracts the UCI HAR dataset (requires `wget` and `unzip`)
2. **Data Loading**: Loads training/test features (X), labels (y), activity labels, and feature names
3. **Data Exploration**: Checks for missing values, examines distributions, visualizes activity counts
4. **Preprocessing**: Resolves duplicate column names by appending numerical suffixes
5. **Data Splitting**: Creates validation set from training data (80/20 split with stratification)
6. **Model Training**: Trains multiple classifiers:
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Logistic Regression
   - Random Forest
   - Decision Tree
7. **Evaluation**: Calculates accuracy, precision, recall, and F1-score for each model
8. **Visualization**: Generates performance comparison plots and confusion matrices

## Key Implementation Details

### Data Loading Pattern

Features and labels are loaded separately and must be aligned:
```python
# Use sep=r'\s+' (raw string) to avoid SyntaxWarning
X_train = pd.read_csv(f'{dataset_path}train/X_train.txt', sep=r'\s+', header=None)
y_train = pd.read_csv(f'{dataset_path}train/y_train.txt', sep=r'\s+', header=None)
```

### Handling Duplicate Column Names

The dataset has 42 duplicate feature names (mostly `fBodyAcc*bandsEnergy*` features). These must be resolved before model training:
```python
def get_unique_columns(columns):
    seen = collections.defaultdict(int)
    unique_columns = []
    for col in columns:
        if seen[col] > 0:
            unique_columns.append(f"{col}_{seen[col]}")
        else:
            unique_columns.append(col)
        seen[col] += 1
    return unique_columns
```

### Model Training Pattern

All scikit-learn models require flattened (1D) label arrays:
```python
y_train_flat = y_train.values.ravel()
model.fit(X_train, y_train_flat)
```

### Visualization Best Practices

When using seaborn countplot with palette, explicitly set hue to avoid FutureWarnings:
```python
sns.countplot(data=data, x='column', hue='column', palette='viridis', legend=False)
```

## Expected Performance

Based on validation set results:
- **Best performers**: Logistic Regression and Random Forest (~98.1% accuracy)
- **Strong performer**: SVM (~97.3% accuracy)
- **Good performer**: KNN (~96.6% accuracy)
- **Baseline**: Decision Tree (~94.2% accuracy)

## File Structure

```
Human_Activity_Recognition/
│
├── backend/                          # FastAPI backend
│   ├── __init__.py
│   ├── main.py                       # FastAPI application with all endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── saved_models/             # Trained models (created after training)
│   └── utils/
│       ├── __init__.py
│       ├── data_processor.py         # HARDataProcessor class for data loading
│       └── model_trainer.py          # HARModelTrainer class for ML models
│
├── templates/
│   └── index.html                    # Frontend web interface (HTML/CSS/JS)
│
├── static/                           # Static files directory
│
├── UCI HAR Dataset/                  # Downloaded dataset (not in repo)
│   ├── train/
│   │   ├── X_train.txt               # Training features (7352 samples)
│   │   ├── y_train.txt               # Training labels
│   │   └── subject_train.txt
│   ├── test/
│   │   ├── X_test.txt                # Test features (2947 samples)
│   │   ├── y_test.txt                # Test labels
│   │   └── subject_test.txt
│   ├── activity_labels.txt           # Activity ID to name mapping
│   ├── features.txt                  # 561 feature names
│   └── features_info.txt
│
├── Human_Activity_Recognition.ipynb  # Original Jupyter notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # Complete documentation
├── CLAUDE.md                         # This file
├── .gitignore                        # Git ignore rules
├── run_app.bat                       # Windows startup script
├── run_app.sh                        # Linux/Mac startup script
└── AI Semester Project Fall 2025.pdf # Project requirements
```

## Architecture

### Backend (FastAPI)

**Main Components:**

1. **`backend/main.py`**: FastAPI application with endpoints for:
   - Dataset loading and info (`/api/dataset/*`)
   - Model training (`/api/train`)
   - Predictions (`/api/predict`)
   - Metrics retrieval (`/api/metrics/*`)
   - Model management (`/api/models/*`)

2. **`backend/utils/data_processor.py`**: `HARDataProcessor` class
   - Loads UCI HAR dataset files
   - Handles duplicate column names (42 duplicates)
   - Creates train/validation split (80/20)
   - Provides dataset information

3. **`backend/utils/model_trainer.py`**: `HARModelTrainer` class
   - Trains 5 ML models (KNN, SVM, LogReg, RF, DT)
   - Calculates metrics (accuracy, precision, recall, F1)
   - Generates confusion matrices
   - Saves/loads models using joblib

### Frontend (HTML/CSS/JavaScript)

**Single-page application** (`templates/index.html`) with:
- Modern gradient UI design
- Real-time training progress
- Interactive predictions
- Metrics visualization
- Responsive layout

### API Endpoints Quick Reference

```python
GET  /                          # Web interface
GET  /api/health                # Health check
POST /api/dataset/load          # Load dataset
GET  /api/dataset/info          # Dataset info
POST /api/train                 # Train models
GET  /api/train/status          # Training status
POST /api/predict               # Make predictions
GET  /api/models                # List models
GET  /api/models/best           # Best model
GET  /api/metrics/summary       # All metrics
```

## Dependencies

### Core Dependencies
- **fastapi**: Web framework (backend)
- **uvicorn**: ASGI server
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: ML algorithms (KNN, SVM, LogReg, RF, DT)
- **joblib**: Model persistence

### Development Dependencies
- **matplotlib**: Visualization (notebook)
- **seaborn**: Statistical plots (notebook)
- **jupyter**: Notebook environment

## Common Development Tasks

### Adding a New ML Model

1. Add training method to `backend/utils/model_trainer.py`:
```python
def train_new_model(self, X_train, y_train, **kwargs):
    y_train_flat = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
    model = NewModelClass(**kwargs)
    model.fit(X_train, y_train_flat)
    self.models['New Model'] = model
    return model
```

2. Update `train_all_models()` method to include the new model
3. Add the model option to frontend dropdown in `templates/index.html`

### Adding a New API Endpoint

1. Add route to `backend/main.py`:
```python
@app.get("/api/new-endpoint")
async def new_endpoint():
    # Your logic here
    return {"result": "data"}
```

2. Update frontend to call the endpoint:
```javascript
const response = await fetch('/api/new-endpoint');
const data = await response.json();
```

### Modifying the Frontend

The frontend is a single HTML file with embedded CSS and JavaScript:
- **CSS**: Located in `<style>` tag in `<head>`
- **JavaScript**: Located in `<script>` tag before `</body>`
- **HTML**: Main content in `<body>`

To add a new card/section:
```html
<div class="card">
    <h2>New Section</h2>
    <!-- Your content -->
</div>
```

## Common Issues

### Backend Issues

1. **Dataset not found**: Ensure `UCI HAR Dataset/` folder exists in project root
2. **Port already in use**: Change port in `backend/main.py` or use `--port 8080`
3. **Import errors**: Check Python path and module structure
4. **Model training fails**: Verify dataset loaded and labels are flattened

### Data Processing Issues

1. **FutureWarning with delim_whitespace**: Use `sep=r'\s+'` instead of `delim_whitespace=True`
2. **Duplicate column names**: Handled automatically by `get_unique_columns()` in data_processor.py
3. **SyntaxWarning with '\s+'**: Use raw string `r'\s+'` to avoid escape sequence warnings
4. **Model fitting errors**: Ensure labels are flattened using `.ravel()` or `.values.ravel()`

### Frontend Issues

1. **CORS errors**: CORS middleware is enabled in backend for all origins
2. **API calls fail**: Check backend is running and URLs are correct
3. **Predictions not working**: Ensure models are trained first (train button)

## Testing

### Manual API Testing

Use the interactive API documentation at `/docs`:
```bash
# Start server
python backend/main.py

# Open browser
http://localhost:8000/docs
```

### Testing with cURL

```bash
# Health check
curl http://localhost:8000/api/health

# Load dataset
curl -X POST http://localhost:8000/api/dataset/load

# Train models
curl -X POST http://localhost:8000/api/train

# Make prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.5, 0.3, ...]], "model_name": "Random Forest"}'
```

## Performance Notes

- **Dataset loading**: ~2-5 seconds
- **Model training**: 2-5 minutes total (SVM is slowest)
  - KNN: instant (lazy learning)
  - Logistic Regression: ~10 seconds
  - Random Forest: ~30 seconds
  - Decision Tree: ~5 seconds
  - SVM: ~1-3 minutes (with probability=True)
- **Predictions**: <1 second for all models
- **Model file sizes**: ~50-200MB total when saved
