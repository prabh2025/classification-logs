import mlflow
import mlflow.sklearn
import joblib
import dagshub
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ✅ Connect to DagsHub (free MLflow server)
dagshub.init(
    repo_owner='prabh2025',
    repo_name='classification-logs',
    mlflow=True
)

mlflow.set_experiment("log_classification")

def train():
    # Training data based on your existing log types
    train_logs = [
    # Security Alerts
    ("IP 192.168.1.1 blocked due to attack", "Security Alert"),
    ("Multiple login failures on user 456", "Security Alert"),
    ("Admin access escalation detected", "Security Alert"),
    ("Unauthorized access attempt detected", "Security Alert"),
    ("Suspicious activity detected on server", "Security Alert"),
    ("Brute force attack detected on port 22", "Security Alert"),
    ("User account locked after failed attempts", "Security Alert"),
    ("IP 10.0.0.1 flagged for suspicious traffic", "Security Alert"),
    ("Malware detected on system drive", "Security Alert"),
    ("Firewall blocked incoming connection", "Security Alert"),

    # User Actions
    ("User 123 logged in.", "User Action"),
    ("Account ID 789 created by Admin", "User Action"),
    ("User User265 logged out.", "User Action"),
    ("User 456 changed their password", "User Action"),
    ("User 789 updated profile settings", "User Action"),
    ("Admin user created new account", "User Action"),
    ("User 321 deleted their account", "User Action"),
    ("User 654 logged in from new device", "User Action"),
    ("Password reset requested by user 987", "User Action"),
    ("User 111 granted admin privileges", "User Action"),

    # System Notifications
    ("Backup completed successfully.", "System Notification"),
    ("File data.csv uploaded by User123", "System Notification"),
    ("System updated to version 2.0", "System Notification"),
    ("Disk cleanup completed successfully.", "System Notification"),
    ("System reboot initiated by user Admin", "System Notification"),
    ("Server maintenance completed", "System Notification"),
    ("Database backup finished successfully", "System Notification"),
    ("System health check passed", "System Notification"),
    ("Scheduled task completed successfully", "System Notification"),
    ("Storage capacity reached 80 percent", "System Notification"),

    # HTTP Requests
    ("GET /v2/servers HTTP/1.1 RCODE 200", "HTTP Request"),
    ("POST /api/login HTTP/1.1 RCODE 401", "HTTP Request"),
    ("GET /v2/users HTTP/1.1 RCODE 404", "HTTP Request"),
    ("DELETE /api/user/123 HTTP/1.1 RCODE 200", "HTTP Request"),
    ("PUT /api/settings HTTP/1.1 RCODE 500", "HTTP Request"),
    ("GET /health HTTP/1.1 RCODE 200", "HTTP Request"),
    ("POST /api/data HTTP/1.1 RCODE 201", "HTTP Request"),
    ("GET /v2/reports HTTP/1.1 RCODE 403", "HTTP Request"),
    ("PATCH /api/user HTTP/1.1 RCODE 200", "HTTP Request"),
    ("GET /api/logs HTTP/1.1 RCODE 200", "HTTP Request"),
]

    texts = [t for t, _ in train_logs]
    labels = [l for _, l in train_logs]

    # Generate BERT embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    X = embedder.encode(texts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="bert_logreg_v1"):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log to MLflow
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("embedder", "all-MiniLM-L6-v2")
        mlflow.log_param("C", 1.0)
        mlflow.log_metric("accuracy", acc)

        # Save model
        joblib.dump(clf, "models/log_classifier.joblib")
        mlflow.sklearn.log_model(
            clf, "model",
            registered_model_name="log_classifier"
        )

        print(f"✅ Training complete | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()