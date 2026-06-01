import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
import pandas as pd
import dagshub

dagshub.init(repo_owner="narjesfarhat", repo_name="spam_project-week1", mlflow=True)
mlflow.set_experiment("iris-experiment")

# Load dataset (free, no download needed)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "naive_bayes": MultinomialNB(alpha=1.0),
    "logistic_regression": LogisticRegression(C=1.0, max_iter=200),
    "linear_svc": LinearSVC(C=0.5),
}

mlflow.set_experiment("spam-classifier-comparison")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)

        # Log params
        mlflow.log_param("model_type", name)
        mlflow.log_param("max_features", 3000)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"{name}: acc={acc:.3f}, f1={f1:.3f}")
