"""
FastAPI backend for Human Activity Recognition
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.data_processor import HARDataProcessor
from backend.utils.model_trainer import HARModelTrainer

# Initialize FastAPI app
app = FastAPI(
    title="Human Activity Recognition API",
    description="API for training and predicting human activities from smartphone sensor data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global instances
data_processor = HARDataProcessor()
model_trainer = HARModelTrainer()

# Track training status
training_status = {
    "is_training": False,
    "progress": 0,
    "current_model": None,
    "completed": False,
    "error": None
}


# Pydantic models
class TrainRequest(BaseModel):
    models: Optional[List[str]] = None  # If None, train all models


class PredictRequest(BaseModel):
    features: List[List[float]]
    model_name: str = "Random Forest"


class DataInfo(BaseModel):
    train_samples: int
    val_samples: int
    test_samples: int
    train_features: int
    num_classes: int
    class_names: List[str]


class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class TrainingResponse(BaseModel):
    message: str
    models_trained: List[str]
    metrics: Dict[str, ModelMetrics]


# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page"""
    html_file = os.path.join("templates", "index.html")
    if os.path.exists(html_file):
        return FileResponse(html_file)
    return HTMLResponse(content="<h1>Human Activity Recognition API</h1><p>Frontend not found. API is running at /docs</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "dataset_available": data_processor.check_dataset_exists()
    }


@app.get("/api/dataset/info")
async def get_dataset_info():
    """Get information about the loaded dataset"""
    try:
        if not data_processor.check_dataset_exists():
            raise HTTPException(
                status_code=404,
                detail="Dataset not found. Please ensure 'UCI HAR Dataset' folder exists."
            )

        if data_processor.X_train_new is None:
            # Load data if not already loaded
            data_processor.load_all_data()

        info = data_processor.get_data_info()
        return info

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dataset/load")
async def load_dataset():
    """Load the dataset"""
    try:
        if not data_processor.check_dataset_exists():
            raise HTTPException(
                status_code=404,
                detail="Dataset not found at 'UCI HAR Dataset/'. Please download it first."
            )

        data = data_processor.load_all_data()

        return {
            "message": "Dataset loaded successfully",
            "info": data_processor.get_data_info()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_models(request: TrainRequest = None):
    """Train machine learning models"""
    global training_status

    try:
        # Check if already training
        if training_status["is_training"]:
            return {"message": "Training already in progress", "status": training_status}

        # Load data if not loaded
        if data_processor.X_train_new is None:
            if not data_processor.check_dataset_exists():
                raise HTTPException(
                    status_code=404,
                    detail="Dataset not found. Please download UCI HAR Dataset first."
                )
            data_processor.load_all_data()

        # Reset training status
        training_status["is_training"] = True
        training_status["progress"] = 0
        training_status["completed"] = False
        training_status["error"] = None

        # Train models
        X_train = data_processor.X_train_new
        y_train = data_processor.y_train_new

        if request and request.models:
            # Train specific models
            for i, model_name in enumerate(request.models):
                training_status["current_model"] = model_name
                training_status["progress"] = int((i / len(request.models)) * 100)

                if model_name == "KNN":
                    model_trainer.train_knn(X_train, y_train)
                elif model_name == "SVM":
                    model_trainer.train_svm(X_train, y_train)
                elif model_name == "Logistic Regression":
                    model_trainer.train_logistic_regression(X_train, y_train)
                elif model_name == "Random Forest":
                    model_trainer.train_random_forest(X_train, y_train)
                elif model_name == "Decision Tree":
                    model_trainer.train_decision_tree(X_train, y_train)

            training_status["progress"] = 100
        else:
            # Train all models
            model_trainer.train_all_models(X_train, y_train)
            training_status["progress"] = 100

        # Evaluate on validation set
        metrics = model_trainer.evaluate(data_processor.X_val, data_processor.y_val)

        # Save models
        model_trainer.save_all_models()

        training_status["is_training"] = False
        training_status["completed"] = True
        training_status["current_model"] = None

        return {
            "message": "Models trained successfully",
            "models_trained": list(model_trainer.models.keys()),
            "metrics": metrics,
            "confusion_matrices": model_trainer.confusion_matrices
        }

    except Exception as e:
        training_status["is_training"] = False
        training_status["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/train/status")
async def get_training_status():
    """Get current training status"""
    return training_status


@app.get("/api/models")
async def get_available_models():
    """Get list of available/trained models"""
    return {
        "trained_models": list(model_trainer.models.keys()),
        "available_models": ["KNN", "SVM", "Logistic Regression", "Random Forest", "Decision Tree"]
    }


@app.get("/api/models/{model_name}/metrics")
async def get_model_metrics(model_name: str):
    """Get metrics for a specific model"""
    if model_name not in model_trainer.metrics:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for model: {model_name}. Train the model first."
        )

    return {
        "model": model_name,
        "metrics": model_trainer.metrics[model_name],
        "confusion_matrix": model_trainer.confusion_matrices.get(model_name)
    }


@app.get("/api/models/best")
async def get_best_model():
    """Get the best performing model"""
    try:
        best_name, best_model, best_metrics = model_trainer.get_best_model()
        return {
            "model_name": best_name,
            "metrics": best_metrics
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/predict")
async def predict(request: PredictRequest):
    """Make predictions using a trained model"""
    try:
        # Check if model is trained
        if request.model_name not in model_trainer.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_name} not trained. Please train it first."
            )

        # Convert input to DataFrame
        X = pd.DataFrame(request.features)

        # Make prediction
        predictions, probabilities = model_trainer.predict(X, request.model_name)

        # Get activity names
        if data_processor.activity_labels is None:
            data_processor.load_activity_labels()

        activity_map = dict(zip(
            data_processor.activity_labels['activity_id'],
            data_processor.activity_labels['activity_name']
        ))

        # Convert predictions to activity names
        predicted_activities = [activity_map[int(pred)] for pred in predictions]

        # Prepare response
        response = {
            "model_used": request.model_name,
            "predictions": predicted_activities,
            "prediction_ids": predictions.tolist(),
        }

        if probabilities is not None:
            response["probabilities"] = probabilities.tolist()
            response["confidence"] = [float(max(prob)) for prob in probabilities]

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get summary of all model metrics"""
    if not model_trainer.metrics:
        raise HTTPException(
            status_code=404,
            detail="No models trained yet. Please train models first."
        )

    return model_trainer.get_metrics_summary()


@app.get("/api/activities")
async def get_activities():
    """Get list of activity labels"""
    try:
        if data_processor.activity_labels is None:
            if not data_processor.check_dataset_exists():
                raise HTTPException(status_code=404, detail="Dataset not found")
            data_processor.load_activity_labels()

        return {
            "activities": data_processor.activity_labels.to_dict('records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
