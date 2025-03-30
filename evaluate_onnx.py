import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import onnxruntime as ort
from onnxruntime_extensions import get_library_path
from sklearn.metrics import classification_report
from scipy.special import softmax


def load_and_process_data():
    # Load the metadata
    metadata = pd.read_csv("metadata_merged.csv")

    # Create text-label pairs using the same format as training
    text_label = pd.DataFrame(columns=["text", "label"])

    # Define label mapping (same as training)
    label_dict = {"normal": 0, "fraud_investment": 1, "fraud_loan": 2}

    # Process each row in metadata
    for (
        type_,
        label,
        pkg_name,
        permissions,
        activities,
        services,
        receivers,
        providers,
        unique_files,
    ) in metadata.values:
        # Create the same three text formats as in training
        texts = [
            json.dumps(
                {
                    "label": label,
                    "pkg_name": pkg_name,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            json.dumps(
                {
                    "label": label,
                    "pkg_name": pkg_name,
                    "manifest": {
                        "permissions": permissions,
                        "activities": activities,
                        "services": services,
                        "receivers": receivers,
                        "providers": providers,
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            json.dumps(
                {
                    "label": label,
                    "pkg_name": pkg_name,
                    "unique_files": unique_files,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]

        # Add each text format with its label
        for text in texts:
            text_label.loc[len(text_label)] = [
                text,
                label_dict[type_],
            ]

    return text_label


def evaluate_onnx_model():
    # Load and process data
    print("Loading and processing data...")
    text_label = load_and_process_data()

    # Initialize ONNX runtime session
    print("Initializing ONNX runtime...")
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())

    session = ort.InferenceSession(
        "combined_malware_classifier.onnx",
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Prepare for evaluation
    all_preds = []
    all_labels = []

    # Evaluate each sample
    print("Evaluating model...")
    for text, label in tqdm(text_label.values):
        # Prepare input
        input_feed = {"input_text": np.array([text[:512]])}

        # Run inference
        outputs = session.run(None, input_feed)
        logits = outputs[1][0]

        # Get prediction
        probabilities = softmax(logits)
        pred = np.argmax(probabilities)

        all_preds.append(pred)
        all_labels.append(label)

    # Print classification report
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["normal", "fraud_investment", "fraud_loan"],
        )
    )


if __name__ == "__main__":
    evaluate_onnx_model()
