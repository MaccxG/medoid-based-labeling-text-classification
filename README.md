# medoid-based-labeling-text-classification
The notebook `medoid_labels.ipynb` presents a **supervised topic labeling** approach based on **k-medoids clustering** within each class, in order to derive **descriptive labels** from text data. Cluster medoids are used as class labels, with the goal of improving the performance of **text embedding models** in text classification tasks.

The core idea is to leverage a labeled dataset and apply k-medoids clustering within each class to identify representative samples: the **medoids**. A medoid can be interpreted as the most central element of a cluster in the embedding space and thus serves as a semantic representative of the corresponding class.

These medoids are then used as **data-driven descriptive labels**, replacing the original class labels with more semantically informative representations.

To evaluate the quality of the induced labels, we measure their effectiveness in a downstream **text classification task** based on sentence embeddings. In particular, we assess whether the induced labels improve the performance of pretrained text embedding models in the text classification task.

**Multiple embedding models are compared** in order to analyze how different representation spaces influence both the quality of the induced labels and the resulting classification performance.

The **F1-score** is used as the primary evaluation metric and is recorded to enable a comparison across models.

## Dataset
- AG News

## Models
- `all-MiniLM-L6-v2`
- `all-mpnet-base-v2`
- `sentence-t5-base`
- `BAAI/bge-base-en`

## Repository structure
```text
.
├── extract_subsets.py     # Script to create dataset subsets with logarithmic growth
├── medoids_labels.ipynb         # Clustering, label induction, and evaluation pipeline
└── README.md
