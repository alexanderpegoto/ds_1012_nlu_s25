"""
Code for Problem 1 of HW 2.
"""
import pickle
from typing import Any, Dict

import evaluate
import numpy as np
import optuna
from datasets import Dataset, load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction




def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) \
        -> Dataset:
    """
    Problem 1d: Implement this function.

    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.

    :param dataset: A dataset
    :param tokenizer: A tokenizer
    :return: The dataset, prepreprocessed using the tokenizer

    Requirements:
    1. All input texts should be tokenized.
    2.BERT models have a maximum input length, and all inputs need to be truncated to this length.
    3.Inputs shorter than the maximum input length should be padded to this length.
    4.The pre-processed inputs do not need to be in the form of PyTorch tensors.

    """
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length = 512, # trying to use tokenizer.model_max_length but error
            padding='max_length',
            truncation = True)

    return dataset.map(tokenize_function, batched=True)

def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:
    """
    Problem 2a: Implement this function.

    This function should be passed to your Trainer's model_init keyword
    argument. It will be used by the Trainer to initialize a new model
    for each hyperparameter tuning trial. Your implementation of this
    function should support training with BitFit by freezing all non-
    bias parameters of the initialized model.

    :param trial: This parameter is required by the Trainer, but it will
        not be used for this problem. Please ignore it
    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be loaded
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A newly initialized pre-trained Transformer classifier
    """
    model = BertForSequenceClassification.from_pretrained(model_name,num_labels=2)

    if use_bitfit:
        for name, param in model.bert.named_parameters():
            if 'bias' not in name:
                param.requires_grad = False

    print("\nTrainable Parameters")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    return model

def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
                 use_bitfit: bool = False) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    """
    accuracy_metric = evaluate.load('accuracy')

    print(f"BitFit: {use_bitfit}")


    # Function to compute accuracy helped by AI
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)


    args = TrainingArguments(
        output_dir = 'checkpoints',
        num_train_epochs = 4,
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch'
    )


    trainer = Trainer(
        model = None,
        model_init = lambda: init_model(None, model_name, use_bitfit=use_bitfit),
        args=args,
        train_dataset = train_data,
        eval_dataset = val_data,
        compute_metrics = compute_metrics
    )

    return trainer




def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Problem 2c: Implement this function.

    Returns keyword arguments passed to Trainer.hyperparameter_search.
    Your hyperparameter search must satisfy the criteria listed in the
    problem set.

    :return: Keyword arguments for Trainer.hyperparameter_search
    """
    search_space = {
            "batch_size": [8, 16, 32, 64, 128],
            "learning_rate": [5e-5, 1e-4, 3e-4, 3e-5],
        }

    sampler = optuna.samplers.GridSampler(search_space)

    # Function that suggests hyperparameters from the search space
    def hp_space(trial: optuna.trial.Trial) -> Dict[str, float]:
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", search_space["learning_rate"]),
            "per_device_train_batch_size": trial.suggest_categorical("batch_size", search_space["batch_size"])
        }


    return {
        "direction": "maximize",
        "backend": "optuna",
        "hp_space": hp_space,
        "sampler": sampler
    }




if __name__ == "__main__":  # Use this script to train your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    # Set up trainer
    trainer = init_trainer(model_name, imdb["train"], imdb["val"],
                           use_bitfit=False)

    # Train and save the best hyperparameters
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open("train_results.p", "wb") as f:
        pickle.dump(best, f)
