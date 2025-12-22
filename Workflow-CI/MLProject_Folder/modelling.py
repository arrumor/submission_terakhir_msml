
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd


# 1. Load Data (winequality_preprocessing.csv)
df = pd.read_csv('winequality_preprocessing.csv')
X = df.drop('quality', axis=1)
# Ubah target menjadi biner: 1 jika kualitas > 6, 0 jika <= 6
y = (df['quality'] > 6).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
input_example = X_train[0:5]

# 2. Setup Eksperimen
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Eksperimen Model Hyperparameter Tuning")

# 3. Tentukan Hyperparameter Space (Skenario Tuning)
n_estimators_list = [10, 50, 100]
max_depth_list = [3, 5, None]
min_samples_split_list = [2, 5]
min_samples_leaf_list = [1, 2]

# 4. Loop Tuning
for n in n_estimators_list:
    for depth in max_depth_list:
        for min_split in min_samples_split_list:
            for min_leaf in min_samples_leaf_list:
                # Mulai Run MLflow
                run_name = f"RF_n{n}_d{depth}_split{min_split}_leaf{min_leaf}"
                with mlflow.start_run(run_name=run_name):
                    # Log informasi dataset
                    dataset_path = "winequality_preprocessing.csv"
                    mlflow.log_param("dataset_name", "winequality_preprocessing.csv")
                    mlflow.log_param("dataset_path", dataset_path)
                    mlflow.log_artifact(dataset_path)

                    # --- A. TRAINING ---
                    model = RandomForestClassifier(
                        n_estimators=n,
                        max_depth=depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    # --- B. MANUAL LOGGING (Pengganti Autolog) ---

                    # 1. Log Parameters (Input konfigurasi)
                    mlflow.log_param("n_estimators", n)
                    mlflow.log_param("max_depth", depth)
                    mlflow.log_param("min_samples_split", min_split)
                    mlflow.log_param("min_samples_leaf", min_leaf)
                    mlflow.log_param("criterion", "gini") # Log param lain yang default

                    # 2. Log Metrics (Output performa - samakan dengan standar autolog)
                    acc = accuracy_score(y_test, predictions)
                    f1 = f1_score(y_test, predictions, average='macro')
                    precision = precision_score(y_test, predictions, average='macro')
                    recall = recall_score(y_test, predictions, average='macro')

                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)

                    # 3. Log Model (Simpan modelnya)
                    mlflow.sklearn.log_model(model, "model_random_forest", input_example=input_example)

                    print(f"Run selesai: n={n}, depth={depth}, split={min_split}, leaf={min_leaf} -> Acc={acc}")