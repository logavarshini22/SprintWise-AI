# SprintWise AI – AI-Driven Sprint Progress Analytics

## What this running model does
- Reads sprint/project data from CSV
- Calculates sprint completion percentage
- Calculates velocity
- Tracks bugs and blocked tasks
- Calculates a sprint risk score
- Uses a small Random Forest model for AI-based risk classification
- Generates AI-style recommendations
- Displays charts and an interactive dashboard
- Allows the analyzed data to be downloaded

## Run in VS Code

### 1. Open the project folder
Open `SprintWise_AI` in VS Code.

### 2. Create a virtual environment (optional but recommended)
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install packages
```bash
pip install -r requirements.txt
```

### 4. Start the application
```bash
streamlit run app.py
```

The browser will open the SprintWise AI dashboard.

## CSV format
Use these columns:
- Sprint
- Planned_Story_Points
- Completed_Story_Points
- Planned_Tasks
- Completed_Tasks
- Bugs
- Blocked_Tasks

A sample file is included as `sample_sprint_data.csv`.

## Project flow
Sprint data → Data preprocessing → Metrics calculation → ML risk prediction → Dashboard → Recommendations

## Important note
The included ML model is a prototype suitable for an academic demonstration. For a production system, train and validate the model on a much larger historical sprint dataset.
