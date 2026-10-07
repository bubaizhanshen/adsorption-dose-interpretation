"""Independent NumPy inference for the released selected-input transformer."""

import numpy as np
from scipy.special import erf, softmax

def scalar_layer(weights, name, variable):
    return weights[f"{name}/{name}/{variable}:0"]


def norm(x, weights, name):
    mean = x.mean(axis=-1, keepdims=True)
    variance = np.square(x - mean).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(variance + 1e-6)
            * scalar_layer(weights, name, "gamma") + scalar_layer(weights, name, "beta"))


def numpy_prediction(data, numeric, categories, vocabulary, weights):
    """Independent inference path, using the inspected source's exact operations."""
    num = data[numeric].to_numpy(float)
    num = np.maximum(num[:, :, None]
                     * scalar_layer(weights, "numerical_embeddings", "NumEmbeddingWeights")[:, 0, :]
                     + scalar_layer(weights, "numerical_embeddings", "NumEmbeddingBias"), 0)
    cat = []
    for i, name in enumerate(categories):
        lookup = {label: j for j, label in enumerate(vocabulary[name])}
        codes = np.array([lookup[label] for label in data[name]], dtype=int)
        layer = "embedding" + (f"_{i}" if i else "")
        cat.append(weights[f"cat_embeddings/cat_embeddings/{layer}/embeddings:0"][codes])
    x = np.concatenate([num, np.stack(cat, axis=1)], axis=1)
    for stage in range(4):
        layer = lambda base, i: base + (f"_{i}" if i else "")
        x = norm(x, weights, layer("layer_normalization", 2 * stage))
        att = layer("multi_head_attention", stage)
        projections = []
        for name in ("query", "key", "value"):
            kernel = weights[f"{att}/{att}/{name}/kernel:0"]
            bias = weights[f"{att}/{att}/{name}/bias:0"]
            projections.append(np.einsum("bte,ehd->bthd", x, kernel) + bias)
        q, k, v = projections
        scores = np.einsum("bthd,bshd->bhts", q, k) / 4.
        context = np.einsum("bhts,bshd->bthd", softmax(scores, axis=-1), v)
        attention = (np.einsum("bthd,hdo->bto", context,
                               weights[f"{att}/{att}/attention_output/kernel:0"])
                     + weights[f"{att}/{att}/attention_output/bias:0"])
        x = x + attention
        seq = layer("sequential", stage)
        dense = layer("dense", 2 * stage)
        hidden = x @ weights[f"{seq}/{dense}/kernel:0"] + weights[f"{seq}/{dense}/bias:0"]
        hidden = .5 * hidden * (1 + erf(hidden / np.sqrt(2)))
        dense = layer("dense", 2 * stage + 1)
        hidden = hidden @ weights[f"{seq}/{dense}/kernel:0"] + weights[f"{seq}/{dense}/bias:0"]
        x = norm(x + hidden, weights, layer("layer_normalization", 2 * stage + 1))
    x = norm(x[:, 0, :], weights, "layer_normalization_8")
    x = x @ scalar_layer(weights, "dense_8", "kernel") + scalar_layer(weights, "dense_8", "bias")
    return (x @ scalar_layer(weights, "Dense_out", "kernel")
            + scalar_layer(weights, "Dense_out", "bias")).reshape(-1)

