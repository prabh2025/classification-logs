import mlflow
import time
import logging
import sys
import os

sys.path.append(os.path.dirname(__file__))

from processor_regex import classify_with_regex
from preprocessor_bert import classify_with_bert
from preprocessor_llm import classify_with_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_log(source, log_msg):
    start = time.time()

    if source == "LegacyCRM":
        label = classify_with_llm(log_msg)
        method = "llm"
    else:
        label = classify_with_regex(log_msg)
        method = "regex"
        if not label:
            label = classify_with_bert(log_msg)
            method = "bert"

    latency = (time.time() - start) * 1000

    # ✅ Log every prediction to MLflow
    with mlflow.start_run(run_name=f"{method}_prediction", nested=True):
        mlflow.log_param("source", source)
        mlflow.log_param("method", method)
        mlflow.log_param("label", label or "Unclassified")
        mlflow.log_metric("latency_ms", latency)

    logger.info(f"[{method.upper()}] {source} → {label} ({latency:.1f}ms)")
    return label

def classify(logs):
    return [classify_log(source, msg) for source, msg in logs]

def classify_csv(input_file):
    import pandas as pd
    df = pd.read_csv(input_file)
    df["target_label"] = classify(
        list(zip(df["source"], df["log_message"]))
    )
    output_file = "resources/output.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved to {output_file}")
    return output_file

if __name__ == '__main__':
    mlflow.set_experiment("log_classification")
    classify_csv("resources/test.csv")