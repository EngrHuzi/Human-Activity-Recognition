# Quick Start Guide

Get your Human Activity Recognition web application running in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- Command line/terminal access
- 2GB free disk space

## Step 1: Install Dependencies (1 minute)

Open terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

## Step 2: Download Dataset (2 minutes)

### Windows (PowerShell):
```powershell
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip" -OutFile "dataset.zip"
Expand-Archive -Path "dataset.zip" -DestinationPath "."
Expand-Archive -Path "UCI HAR Dataset.zip" -DestinationPath "."
Remove-Item dataset.zip
```

### Linux/Mac:
```bash
wget https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
unzip human_activity_recognition_using_smartphones.zip
unzip "UCI HAR Dataset.zip"
rm *.zip
```

## Step 3: Run the Application (30 seconds)

### Quick Method:

**Windows:**
```bash
run_app.bat
```

**Linux/Mac:**
```bash
chmod +x run_app.sh
./run_app.sh
```

### Manual Method:

```bash
python backend/main.py
```

## Step 4: Open Browser

Navigate to: **http://localhost:8000**

## Step 5: Use the Application

1. Click **"Load Dataset"** button
2. Click **"Train All Models"** button (wait 2-5 minutes)
3. Click **"View Metrics"** to see performance
4. Click **"Predict Activity"** to make predictions!

## Troubleshooting

### "Dataset not found" error
- Make sure `UCI HAR Dataset` folder exists in the project root
- Check that it contains `train/` and `test/` folders

### "Port already in use" error
- Close other applications using port 8000
- Or change port: `uvicorn backend.main:app --port 8080`

### Import errors
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Check Python version: `python --version` (needs 3.8+)

## Next Steps

- View API documentation at http://localhost:8000/docs
- Read README.md for detailed information
- Check CLAUDE.md for development guidance

## Support

For issues, check:
1. README.md - Complete documentation
2. CLAUDE.md - Technical details
3. `/api/health` endpoint - System status
