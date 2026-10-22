"""ChurnIQ — Telecom Customer Churn Prediction (Flask)"""

import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Reference dataframe used for consistent one-hot encoding
df_1 = pd.read_csv("first_telc.csv")

# Load trained model once at startup
with open("model.sav", "rb") as f:
    model = pickle.load(f)

EXPECTED_FEATURES = model.feature_names_in_

FORM_FIELDS = [
    "SeniorCitizen", "MonthlyCharges", "TotalCharges", "gender",
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "tenure",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    values = [request.form.get(f"query{i}", "") for i in range(1, 20)]

    new_df = pd.DataFrame([values], columns=FORM_FIELDS)
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Bin tenure into 12-month groups
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2["tenure_group"] = pd.cut(
        df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels
    )
    df_2.drop(columns=["tenure"], inplace=True)

    dummies = pd.get_dummies(df_2[[
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group",
    ]])

    dummies = dummies.loc[:, ~dummies.columns.duplicated()]
    dummies = dummies.reindex(columns=EXPECTED_FEATURES, fill_value=0)

    prediction = model.predict(dummies.tail(1))[0]
    probability = model.predict_proba(dummies.tail(1))[0][1]

    if prediction == 1:
        headline = "High Churn Risk"
        message = "This customer is likely to churn."
        status = "risk"
    else:
        headline = "Customer Likely to Stay"
        message = "This customer is likely to continue with the service."
        status = "safe"

    confidence = round(float(probability) * 100, 2)

    form_state = {f"query{i}": values[i - 1] for i in range(1, 20)}

    return render_template(
        "predict.html",
        headline=headline,
        message=message,
        confidence=confidence,
        status=status,
        **form_state,
    )


@app.route("/insights")
def insights():
    return render_template("insights.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
