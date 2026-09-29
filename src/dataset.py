from datasets import load_dataset, interleave_datasets, Dataset
import os

def get_language_datasets(config, split="train"):
    """
    Load English and Russian datasets as specified in config.
    Returns a list of datasets ready for interleaving.
    """
    # English dataset
    eng_dataset = load_dataset(
        config.data.english_dataset,
        name=config.data.english_config,
        split=split,
        streaming=config.data.streaming,
    )
    # Russian dataset
    rus_dataset = load_dataset(
        config.data.russian_dataset,
        name=config.data.russian_config,
        split=split,
        streaming=config.data.streaming,
    )
    return [eng_dataset, rus_dataset]

def mix_datasets(datasets, probabilities=None, seed=42):
    """
    Interleave multiple datasets with given probabilities.
    If probabilities is None, equal weighting.
    """
    if probabilities is None:
        probabilities = [1.0 / len(datasets)] * len(datasets)
    return interleave_datasets(datasets, probabilities=probabilities, seed=seed, stop_on_empty=False)

def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenize a batch of examples.
    Expects examples to have a "text" field.
    Returns dict with input_ids and attention_mask.
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_tensors=None,
    )
    return {"input_ids": tokenized["input_ids"]}

def prepare_dataloader(dataset, tokenizer, batch_size, max_length, shuffle=True):
    """
    Convert a streaming dataset to a PyTorch DataLoader.
    Note: For simplicity, we assume the dataset yields dict with "text".
    """
    import torch
    from torch.utils.data import IterableDataset, DataLoader

    class TokenizedIterableDataset(IterableDataset):
        def __init__(self, base_dataset, tokenizer, max_length):
            self.base_dataset = base_dataset
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __iter__(self):
            for example in self.base_dataset:
                # example is dict with "text"
                ids = self.tokenizer.encode(example["text"])
                # truncate or pad
                if len(ids) > self.max_length:
                    ids = ids[:self.max_length]
                # create attention mask (1 for real tokens, 0 for padding)
                attention_mask = [1] * len(ids)
                # pad if needed
                pad_len = self.max_length - len(ids)
                if pad_len > 0:
                    ids = ids + [self.tokenizer.token_to_id("[PAD]")] * pad_len
                    attention_mask = attention_mask + [0] * pad_len
                yield {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }

    tokenized_ds = TokenizedIterableDataset(dataset, tokenizer, max_length)
    loader = DataLoader(tokenized_ds, batch_size=batch_size, shuffle=shuffle)
    return loader