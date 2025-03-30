import torch
from onnxruntime_extensions import gen_processing_models
from onnxruntime_extensions import get_library_path
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
import onnxruntime as ort
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from scipy.special import softmax


def export_model():
    # Step 1: Load the Huggingface Roberta tokenizer and model
    input_text = "A test text!"
    model_type = "roberta-base"

    model = RobertaForSequenceClassification.from_pretrained(model_type, num_labels=3)
    model.load_state_dict(
        torch.load("best_roberta_model_ultimate.pt", map_location=torch.device("cpu"))
    )
    model.eval()

    tokenizer = RobertaTokenizer.from_pretrained(model_type)

    # Step 2: Export the tokenizer to ONNX using gen_processing_models
    onnx_tokenizer_path = "tokenizer.onnx"

    # Generate the tokenizer ONNX model with proper configuration
    tokenizer_onnx_model, _ = gen_processing_models(
        tokenizer,
        pre_kwargs={
            "max_length": 512,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        },
    )
    assert tokenizer_onnx_model is not None

    # Update the tokenizer model's IR version to match ONNX opset 14
    tokenizer_onnx_model.ir_version = 8

    # Save the tokenizer ONNX model
    with open(onnx_tokenizer_path, "wb") as f:
        f.write(tokenizer_onnx_model.SerializeToString())

    # Step 3: Export the Huggingface Roberta model to ONNX
    onnx_model_path = "malware_classifier.onnx"
    dummy_input = tokenizer(
        "This is a dummy input",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    # Export the model to ONNX with proper dynamic axes and opset version 14
    torch.onnx.export(
        model,  # model to be exported
        (
            dummy_input["input_ids"],
            dummy_input["attention_mask"],
        ),  # model input (dummy input)
        onnx_model_path,  # where to save the ONNX model
        input_names=["input_ids", "attention_mask_input"],  # input tensor name
        output_names=["logits"],  # output tensor names
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},  # dynamic axes
            "attention_mask_input": {
                0: "batch_size",
                1: "sequence_length",
            },  # dynamic axes
            "logits": {0: "batch_size"},
        },
        opset_version=14,  # Use opset version 14 for scaled_dot_product_attention support
    )

    # Step 4: Merge the tokenizer and model ONNX files into one
    onnx_combined_model_path = "combined_malware_classifier.onnx"

    # Load the tokenizer and model ONNX files
    tokenizer_onnx_model = onnx.load(onnx_tokenizer_path)
    model_onnx_model = onnx.load(onnx_model_path)

    # Update model's IR version to match tokenizer
    model_onnx_model.ir_version = 8
    # Quantize the model
    quantize_dynamic(
        model_onnx_model,
        onnx_model_path,
        weight_type=QuantType.QInt8,
    )
    model_onnx_model = onnx.load(onnx_model_path)
    # onnx.save(combined_model, onnx_combined_model_path)
    # onnx.save(model_onnx_model, onnx_model_path)

    # Inspect the ONNX models to find the correct input/output names
    print(
        "Tokenizer Model Inputs:",
        [node.name for node in tokenizer_onnx_model.graph.input],
    )
    print(
        "Tokenizer Model Outputs:",
        [node.name for node in tokenizer_onnx_model.graph.output],
    )
    print("Model Inputs:", [node.name for node in model_onnx_model.graph.input])
    print("Model Outputs:", [node.name for node in model_onnx_model.graph.output])
    print("Tokenizer IR version:", tokenizer_onnx_model.ir_version)
    print("Model IR version:", model_onnx_model.ir_version)

    # Merge the tokenizer and model ONNX files
    combined_model = onnx.compose.merge_models(
        tokenizer_onnx_model,
        model_onnx_model,
        io_map=[("input_ids", "input_ids"), ("attention_mask", "attention_mask_input")],
    )

    # Save the combined model
    onnx.save(combined_model, onnx_combined_model_path)

    # Step 5: Test the combined ONNX model
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())

    session = ort.InferenceSession(
        onnx_combined_model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Prepare input text
    input_feed = {"input_text": np.array([input_text])}

    # Run the model
    outputs = session.run(None, input_feed)
    logits = outputs[1][0]

    # Convert logits to probabilities using softmax
    probabilities = softmax(logits)

    # Print both logits and probabilities
    print("\nModel Outputs:")
    print("Logits:", logits)
    print("Probabilities:", probabilities)
    print("\nClass predictions:")
    for i, (logit, prob) in enumerate(zip(logits, probabilities)):
        print(f"Class {i}: Logit = {logit:.4f}, Probability = {prob:.4f}")


if __name__ == "__main__":
    export_model()
