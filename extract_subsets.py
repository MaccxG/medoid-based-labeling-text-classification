from datasets import load_dataset, concatenate_datasets, DatasetDict
import numpy as np
import math

dataset = load_dataset("ag_news")

print(dataset["train"])
print(dataset["test"])


def extract_examples_per_label(dataset, n):
    labels = dataset.unique("label")    # Original label as a numeric value
    examples_per_class = n // len(labels) + (n % len(labels) != 0)
    parts = []
    for label in labels:
        sub = dataset.filter(lambda x: x["label"] == label).select(range(examples_per_class))
        parts.append(sub)
    
    return concatenate_datasets(parts)


def save_dataset(dataset_dict, base_name, split_name, num_examples):
    path = f"./data/{base_name}_{split_name}_{num_examples}"
    dataset_dict.save_to_disk(path)
    print(f"Saved to: {path}")


dataset_base_name = "AgNews"

# ------------------------------
# TRAIN SET
# ------------------------------

train_ds = dataset["train"].shuffle(seed=42)
test_ds = dataset["test"]

# Define subset sizes using log-spaced values between 10 and 10,000
sizes = np.logspace(math.log10(10), math.log10(10000), num=4, dtype=int)

for n in sizes:
    if n == 10:
        # Sample at least n examples per label, required to make k-medoids work with a small number of samples
        train_set = extract_examples_per_label(train_ds, n)
    else:
        splits = train_ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
        train_full = splits["train"]
        train_set = train_full.shuffle(seed=42).select(range(n))

    train_dataset_dict = DatasetDict({
        "train": train_set
    })
    print(train_dataset_dict)
    # Save to disk
    save_dataset(train_dataset_dict, dataset_base_name, "train", n)


# ------------------------------
# TEST SET
# ------------------------------

test_set = test_ds.shuffle(seed=42).select(range(1000))
test_dataset_dict = DatasetDict({"test": test_set})
print(test_dataset_dict)
# Save to disk
save_dataset(test_dataset_dict, dataset_base_name, "test", len(test_set))