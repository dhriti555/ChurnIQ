# ChurnIQ

An elegant multi-page Flask-based machine learning app that predicts telecom customer churn using a Decision Tree Classifier trained on the Kaggle Telco Customer Churn dataset.

## Pages

| Page | Description |
|------|-------------|
| **Home** | Landing page with hero, feature cards, and key stats |
| **Predict** | Interactive form — enter 19 customer attributes and get a churn risk score |
| **Insights** | Visual summary of dataset patterns (contracts, services, risk signals) |
| **Dataset** | Overview of the Telco Customer Churn data and preprocessing steps |
| **How it Works** | Step-by-step explanation of the prediction pipeline |
| **About** | Project background, tech stack, and design philosophy |

## Features
- Clean, modern, responsive dark UI across all pages
- Shared navigation and footer for a cohesive experience
- Real-time churn prediction with confidence score
- 19 customer attributes (demographics, services, billing, tenure)
- Pre-trained model included (`model.sav`)

## Tech Stack
- **Backend:** Flask
- **ML:** scikit-learn, pandas, numpy
- **Frontend:** HTML5 + modern CSS (no framework needed)

## Setup

```bash
# 1. Clone / extract the project
cd ChurnIQ

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Project Structure
```
ChurnIQ/
├── app.py                    # Flask routes
├── model.sav                 # Trained ML model
├── first_telc.csv            # Reference frame for encoding
├── Telco-Customer-Churn.csv  # Original dataset
├── requirements.txt
├── templates/
│   ├── base.html             # Shared layout (nav + footer)
│   ├── index.html            # Landing
│   ├── predict.html          # Prediction form
│   ├── insights.html
│   ├── dataset.html
│   ├── how_it_works.html
│   └── about.html
└── static/
    └── styles.css
```

## Dataset
[Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## License
MIT
