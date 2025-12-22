import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Load Data (winequality_preprocessing.csv)
df = pd.read_csv('Membangun_model/winequality_preprocessing.csv')
X = df.drop('quality', axis=1)
# Ubah target menjadi biner: 1 jika kualitas > 6, 0 jika <= 6
y = (df['quality'] > 6).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)
input_example = X_train[0:5]

# 2. Setup Eksperimen
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Eksperimen Model RandomForest Sederhana")

# 3. Training dan Logging (tanpa tuning hyperparameter)
with mlflow.start_run(run_name="RF_default_params"):
	# Log informasi dataset
	dataset_path = "Membangun_model/winequality_preprocessing.csv"
	mlflow.log_param("dataset_name", "winequality_preprocessing.csv")
	mlflow.log_param("dataset_path", dataset_path)
	mlflow.log_param("dataset_shape", str(X.shape))
	mlflow.log_artifact(dataset_path)

	# --- TRAINING ---
	model = RandomForestClassifier(
		n_estimators=100,
		max_depth=None,
		min_samples_split=2,
		min_samples_leaf=1,
		random_state=42
	)
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)

	# --- MANUAL LOGGING ---
	mlflow.log_param("n_estimators", 100)
	mlflow.log_param("max_depth", None)
	mlflow.log_param("min_samples_split", 2)
	mlflow.log_param("min_samples_leaf", 1)
	mlflow.log_param("criterion", "gini")

	acc = accuracy_score(y_test, predictions)
	f1 = f1_score(y_test, predictions, average='macro')
	precision = precision_score(y_test, predictions, average='macro')
	recall = recall_score(y_test, predictions, average='macro')

	mlflow.log_metric("accuracy", acc)
	mlflow.log_metric("f1_score", f1)
	mlflow.log_metric("precision", precision)
	mlflow.log_metric("recall", recall)

	mlflow.sklearn.log_model(model, "model_random_forest", input_example=input_example)

	print(f"Run selesai: Acc={acc}")