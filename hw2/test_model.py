"""
Code for Problem 1 of HW 2.
"""
import pickle

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import preprocess_dataset


def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    # Load accuracy metric
    accuracy_metric = evaluate.load('accuracy')
    # Function to compute accuracy
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # Load the fine-tuned model from the given directory
    model = BertForSequenceClassification.from_pretrained(directory)


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of Trainable Parameters: {trainable_params}")

    # Define evaluation arguments (ensuring no training happens)
    args = TrainingArguments(
        output_dir="./test_checkpoints",
        do_train=False,
        do_eval=True,  # Only evaluation mode
        per_device_eval_batch_size=32

    )

    tester = Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics
    )

    return tester


if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up tester
    tester = init_tester("checkpoints/run-1/checkpoint-10000")

    # Test
    results = tester.predict(imdb["test"])
    with open("test_results.p", "wb") as f:
        pickle.dump(results, f)
