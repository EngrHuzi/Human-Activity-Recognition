# Human Activity Recognition - Project Summary

## What This Project Does

This is a complete full-stack web application that uses machine learning to recognize human activities (walking, sitting, standing, etc.) from smartphone sensor data. Perfect for university projects demonstrating ML skills!

## Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **Scikit-learn**: 5 different ML algorithms (KNN, SVM, Logistic Regression, Random Forest, Decision Tree)
- **Pandas & NumPy**: Data processing
- **Uvicorn**: ASGI web server

### Frontend
- **Pure HTML/CSS/JavaScript**: No frameworks needed
- **Modern UI**: Gradient design, responsive layout
- **Interactive**: Real-time model training and predictions

### Data
- **UCI HAR Dataset**: 10,299 samples of smartphone accelerometer data
- **561 Features**: Time and frequency domain variables
- **6 Activities**: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying

## Project Structure

```
├── backend/                    # Python backend
│   ├── main.py                # FastAPI server (all endpoints)
│   └── utils/
│       ├── data_processor.py  # Dataset loading & preprocessing
│       └── model_trainer.py   # ML model training & evaluation
│
├── templates/
│   └── index.html             # Complete frontend (HTML+CSS+JS)
│
├── requirements.txt           # Python dependencies
├── README.md                  # Full documentation
├── QUICK_START.md             # 5-minute setup guide
├── CLAUDE.md                  # Developer guide
└── run_app.bat/sh             # Startup scripts
```

## Key Features

### 1. Data Management
- Automatic dataset loading
- Handles duplicate column names (42 duplicates)
- Train/validation/test split (80/20)
- Real-time dataset info display

### 2. Machine Learning
- **5 Classification Models**:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest
  - Decision Tree

- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrices

### 3. Web Interface
- Load dataset with one click
- Train all models automatically
- Compare model performance
- Make real-time predictions
- View metrics and best model

### 4. RESTful API
- Complete API with 11+ endpoints
- Interactive documentation at `/docs`
- JSON responses
- Error handling

## Performance

| Model | Expected Accuracy |
|-------|------------------|
| Random Forest | ~98.1% |
| Logistic Regression | ~98.1% |
| SVM | ~97.3% |
| KNN | ~96.6% |
| Decision Tree | ~94.2% |

## How to Run

### Quick Method
```bash
# Windows
run_app.bat

# Linux/Mac
./run_app.sh
```

### Manual Method
```bash
pip install -r requirements.txt
python backend/main.py
```

Then open: http://localhost:8000

## File Sizes

- **Code**: ~50 KB (highly modular)
- **Dataset**: ~60 MB (download separately)
- **Models**: ~50-200 MB (generated after training)
- **Total Project**: ~300-400 MB after setup

## University Project Checklist

✅ Data loading and preprocessing
✅ Exploratory data analysis (notebook)
✅ Multiple ML algorithms (5 models)
✅ Model comparison and evaluation
✅ Complete metrics (Accuracy, Precision, Recall, F1)
✅ Visualization of results
✅ Web application interface
✅ RESTful API
✅ Complete documentation
✅ Easy to demonstrate
✅ Professional presentation

## Demonstration Flow

1. **Start Application** (10 seconds)
   - Run `run_app.bat`
   - Show web interface

2. **Load Dataset** (5 seconds)
   - Click "Load Dataset"
   - Show 10,299 samples, 561 features, 6 classes

3. **Train Models** (2-3 minutes)
   - Click "Train All Models"
   - Show progress bar
   - Display training completion

4. **View Results** (30 seconds)
   - Click "View Metrics"
   - Show table with all model performances
   - Highlight best model (Random Forest/Logistic Regression)

5. **Make Predictions** (15 seconds)
   - Select model from dropdown
   - Click "Predict Activity"
   - Show prediction with confidence

6. **Show API** (30 seconds)
   - Open `/docs` in browser
   - Demonstrate interactive API documentation
   - Make a test API call

**Total Demo Time**: ~5 minutes

## Advantages Over Notebook-Only Projects

### Traditional Notebook
- Static analysis
- Run cells manually
- No interactivity
- Hard to demonstrate
- Not production-ready

### This Full-Stack Application
- ✅ Interactive web interface
- ✅ One-click training
- ✅ Real-time predictions
- ✅ Easy to demonstrate
- ✅ Production-ready code
- ✅ RESTful API
- ✅ Professional presentation

## Code Quality

- **Modular**: Separate classes for data processing and model training
- **Documented**: Comprehensive docstrings and comments
- **Type-Safe**: Pydantic models for API validation
- **Error Handling**: Try-catch blocks and HTTP status codes
- **RESTful**: Standard API design patterns
- **Responsive**: Works on desktop and mobile

## Learning Outcomes Demonstrated

1. **Machine Learning**: Multiple algorithms, evaluation, comparison
2. **Data Science**: Data preprocessing, feature engineering
3. **Backend Development**: FastAPI, RESTful APIs
4. **Frontend Development**: HTML, CSS, JavaScript
5. **Full-Stack Integration**: Frontend-backend communication
6. **Software Engineering**: Modular code, documentation
7. **Project Management**: Complete project structure

## Future Enhancements (Optional)

- Add more ML models (Neural Networks, XGBoost)
- Real-time data from smartphone sensors
- User authentication and history
- Model retraining interface
- Export predictions to CSV
- Deployment to cloud (Heroku, AWS)
- Docker containerization
- Unit tests and CI/CD

## Troubleshooting

### Dataset Not Found
Download from: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

### Port In Use
Change port in `backend/main.py` or use:
```bash
uvicorn backend.main:app --port 8080
```

### Import Errors
```bash
pip install -r requirements.txt --force-reinstall
```

## Support Files

- **README.md**: Complete technical documentation
- **QUICK_START.md**: 5-minute setup guide
- **CLAUDE.md**: Developer reference guide
- **PROJECT_SUMMARY.md**: This file

## Credits

- **Dataset**: UCI Machine Learning Repository
- **Framework**: FastAPI
- **ML Library**: Scikit-learn
- **Project Type**: University Assignment - Fall 2025

## License

Educational use - University Project

---

**Ready to impress your professors?** 🎓

This project demonstrates:
- ✅ Full-stack development skills
- ✅ Machine learning expertise
- ✅ Professional code organization
- ✅ Complete documentation
- ✅ Production-ready application

Good luck with your presentation! 🚀
