import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load CSV file
df = pd.read_csv("loan_data.csv")

# Select input and output columns
X = df[["CIBIL_Score", "Loan_Amount"]]
y = df["Eligible"]

# Create and train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

model.fit(X, y)

# Predict loan eligibility
df["Prediction"] = model.predict(X)

# Separate eligible and ineligible applicants
eligible = df[df["Prediction"] == 1]
not_eligible = df[df["Prediction"] == 0]

# Plot graph
plt.figure(figsize=(10, 6))

plt.scatter(
    eligible["CIBIL_Score"],
    eligible["Loan_Amount"],
    color="green",
    label="Eligible"
)

plt.scatter(
    not_eligible["CIBIL_Score"],
    not_eligible["Loan_Amount"],
    color="red",
    marker="x",
    label="Not Eligible"
)

plt.xlabel("CIBIL Score")
plt.ylabel("Loan Amount Required (₹)")
plt.title("Loan Eligibility Prediction Using Logistic Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()