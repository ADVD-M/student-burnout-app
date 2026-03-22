# Student Burnout Risk Prediction Web App

An AI-powered tool for predicting student burnout risk and providing personalized mental wellness support.

## Overview
This application uses a Random Forest classification model to predict burnout risk as Low, Medium, or High based on academic and lifestyle data. It includes a risk prediction form, a results dashboard (metrics and feature importance), and an AI wellness chatbot for stress management.

Disclaimer: This tool is for informational purposes only and is not a substitute for professional mental health advice.

# View the Live App
[Link to App](https://student-burnout-pred.streamlit.app/)

## Project Structure
- app/: Streamlit frontend (Home and sub-pages)
- chatbot/: AI response engine
- data/: Dataset storage
- models/: Saved model artifacts and metrics
- src/: ML backend (loader, engineering, trainer, inference)
- utils/: Shared helpers
- train.py: End-to-end training script
- requirements.txt: Dependency list

## Quick Start

1. Clone the project:
git clone https://github.com/yourusername/student_burnout_app.git
cd student_burnout_app

2. Set up virtual environment:
python -m venv venv

3. Install dependencies:
pip install -r requirements.txt

4. Place dataset:
Ensure mentalhealth_dataset.csv is in the data/ folder.

5. Train model:
python train.py

6. Launch app:
streamlit run app/Home.py

## Tech Stack
- ML: scikit-learn (Random Forest)
- Frontend: Streamlit
- Visuals: Plotly
- Data: pandas, numpy
