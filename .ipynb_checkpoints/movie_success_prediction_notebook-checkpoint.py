"""
Movie Success Prediction (TMDB/IMDb style)

Notebook-ready script that:
1) Loads movie data from CSV
2) Cleans and preprocesses features
3) Trains Logistic Regression and Random Forest classifiers
4) Evaluates with accuracy, precision, recall, confusion matrix
5) Plots confusion matrix and Random Forest feature importance
6) Shows five incorrect predictions
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# %%
# -------------------------
# 1) Load dataset from CSV
# -------------------------
# Update this to your CSV file path if needed.
CSV_PATH = "movies.csv"

df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset shape: {df.shape}")
df.head()


# %%
# ----------------------------------------------------------
# 2) Normalize expected column names for TMDB/IMDb variations
# ----------------------------------------------------------
column_aliases = {
    "budget": ["budget", "production_budget"],
    "runtime": ["runtime", "duration", "movie_duration"],
    "popularity": ["popularity", "popularity_score"],
    "vote_count": ["vote_count", "num_votes", "votes"],
    "vote_average": ["vote_average", "rating", "average_rating"],
    "genre": ["genre", "genres", "genre_names"],
}


def pick_existing_column(columns, candidates):
    """Return the first existing column from candidates, otherwise None."""
    for col in candidates:
        if col in columns:
            return col
    return None


selected_cols = {}
for key, aliases in column_aliases.items():
    found = pick_existing_column(df.columns, aliases)
    if found is None:
        raise ValueError(
            f"Required field '{key}' not found. "
            f"Checked aliases: {aliases}"
        )
    selected_cols[key] = found

print("Using columns:", selected_cols)


# %%
# -------------------------------------------------
# 3) Keep relevant columns and drop obvious extras
# -------------------------------------------------
work_df = df[
    [
        selected_cols["budget"],
        selected_cols["runtime"],
        selected_cols["popularity"],
        selected_cols["vote_count"],
        selected_cols["vote_average"],
        selected_cols["genre"],
    ]
].copy()

# Rename to canonical names for easier downstream logic.
work_df.columns = ["budget", "runtime", "popularity", "vote_count", "vote_average", "genre"]


# %%
# -------------------------------------
# 4) Handle missing values and cleaning
# -------------------------------------
# Ensure numeric columns are truly numeric; coerce invalid values to NaN.
numeric_cols = ["budget", "runtime", "popularity", "vote_count", "vote_average"]
for col in numeric_cols:
    work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

# Fill numeric missing values with median.
for col in numeric_cols:
    work_df[col] = work_df[col].fillna(work_df[col].median())

# Fill missing genre with a placeholder.
work_df["genre"] = work_df["genre"].fillna("Unknown").astype(str)

# Clean genre string format (handles common list-like and delimiter variants).
work_df["genre"] = (
    work_df["genre"]
    .str.replace(r"[\[\]\"']", "", regex=True)
    .str.replace(r"\s*,\s*", "|", regex=True)
    .str.replace(r"\s*\|\s*", "|", regex=True)
    .str.strip()
)

work_df.head()


# %%
# ----------------------------------------------------------
# 5) Define success target: vote_average >= 7 (binary class)
# ----------------------------------------------------------
work_df["success"] = (work_df["vote_average"] >= 7).astype(int)

print("Target distribution:")
print(work_df["success"].value_counts(normalize=True).rename("proportion"))


# %%
# --------------------------------------------------------
# 6) One-hot encode categorical genre feature into numeric
# --------------------------------------------------------
genre_dummies = work_df["genre"].str.get_dummies(sep="|")

X = pd.concat(
    [work_df[["budget", "runtime", "popularity", "vote_count"]], genre_dummies],
    axis=1,
)
y = work_df["success"]

print(f"Feature matrix shape: {X.shape}")
print(f"Number of genre dummy columns: {genre_dummies.shape[1]}")


# %%
# -----------------------------
# 7) Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)


# %%
# ---------------------------------------------
# 8) Train Logistic Regression and Random Forest
# ---------------------------------------------
log_reg = LogisticRegression(max_iter=2000, random_state=42)
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

log_reg.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


# %%
# ---------------------------------------------
# 9) Evaluate both models with key metrics
# ---------------------------------------------
def evaluate_model(model_name, y_true, y_pred):
    """Print accuracy, precision, recall, and return confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{model_name} Performance")
    print("-" * (len(model_name) + 12))
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)
    return cm


log_preds = log_reg.predict(X_test)
rf_preds = rf_model.predict(X_test)

cm_log = evaluate_model("Logistic Regression", y_test, log_preds)
cm_rf = evaluate_model("Random Forest", y_test, rf_preds)


# %%
# ---------------------------------------------
# 10) Plot confusion matrix (Random Forest)
# ---------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Successful (0)", "Successful (1)"],
    yticklabels=["Not Successful (0)", "Successful (1)"],
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


# %%
# ---------------------------------------------
# 11) Plot Random Forest feature importances
# ---------------------------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_n = 15
top_features = importances.sort_values(ascending=False).head(top_n)

plt.figure(figsize=(9, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title(f"Top {top_n} Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# %%
# ---------------------------------------------------------
# 12) Output 5 incorrect predictions (actual vs predicted)
# ---------------------------------------------------------
results_df = X_test.copy()
results_df["actual"] = y_test.values
results_df["predicted"] = rf_preds

incorrect = results_df[results_df["actual"] != results_df["predicted"]].copy()

print("\nFive incorrect predictions (Random Forest):")
if incorrect.empty:
    print("No incorrect predictions found in the test set.")
else:
    display_cols = ["actual", "predicted", "budget", "runtime", "popularity", "vote_count"]
    print(incorrect[display_cols].head(5))

