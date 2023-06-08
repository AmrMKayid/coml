import os
import csv
import torch
import random
import numpy as np
import pandas as pd
import altair as alt

from typing import List, Tuple, Dict, Union
from sklearn.decomposition import PCA
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity


def seed_everything(seed: int = 2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def passed():
    from IPython.display import HTML

    pups = [
        "2m78jPG",
        "pn1e9TO",
        "MQCIwzT",
        "udLK6FS",
        "ZNem5o3",
        "DS2IZ6K",
        "aydRUz8",
        "MVUdQYK",
        "kLvno0p",
        "wScLiVz",
        "Z0TII8i",
        "F1SChho",
        "9hRi2jN",
        "lvzRF3W",
        "fqHxOGI",
        "1xeUYme",
        "6tVqKyM",
        "CCxZ6Wr",
        "lMW0OPQ",
        "wHVpHVG",
        "Wj2PGRl",
        "HlaTE8H",
        "k5jALH0",
        "3V37Hqr",
        "Eq2uMTA",
        "Vy9JShx",
        "g9I2ZmK",
        "Nu4RH7f",
        "sWp0Dqd",
        "bRKfspn",
        "qawCMl5",
        "2F6j2B4",
        "fiJxCVA",
        "pCAIlxD",
        "zJx2skh",
        "2Gdl1u7",
        "aJJAY4c",
        "ros6RLC",
        "DKLBJh7",
        "eyxH0Wc",
        "rJEkEw4",
    ]
    return HTML(
        """
    <video alt="test" controls autoplay=1>
        <source src="https://openpuppies.com/mp4/%s.mp4"  type="video/mp4">
    </video>
    """
        % (random.sample(pups, 1)[0])
    )


def get_pc(
    embeddings: np.ndarray,
    n: int = 2,
) -> np.ndarray:
    "Function to return the principal components of the embeddings"
    pca = PCA(n_components=n)
    embeds_transform = pca.fit_transform(embeddings)
    return embeds_transform


def generate_chart(
    df: pd.DataFrame,
    xcol: str = "x",
    ycol: str = "y",
    lbl="on",
    color="basic",
    title="Embeddings Visualization",
) -> alt.Chart:
    "Function to generate the 2D plot of the embeddings"
    chart = (
        alt.Chart(df)
        .mark_circle(size=500)
        .encode(
            x=alt.X(
                xcol,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(labels=False, ticks=False, domain=False),
            ),
            y=alt.Y(
                ycol,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(labels=False, ticks=False, domain=False),
            ),
            color=alt.value("#333293") if color == "basic" else color,
            tooltip=["texts", "x", "y"],
        )
    )

    if lbl == "on":
        text = chart.mark_text(
            align="left", baseline="middle", dx=15, size=13, color="black"
        ).encode(text="texts", color=alt.value("black"))
    else:
        text = chart.mark_text(align="left", baseline="middle", dx=10).encode()

    result = (
        (chart + text)
        .configure(background="#FDF7F0")
        .properties(width=800, height=500, title=title)
        .configure_legend(orient="bottom", titleFontSize=18, labelFontSize=18)
    )

    return result


def visualize_embeddings(df: pd.DataFrame) -> alt.Chart:
    embeds_pc2 = get_pc(np.array(df["text_embeddings"].tolist()), 2)

    df_viz = pd.concat([df["texts"], pd.DataFrame(embeds_pc2)], axis=1)
    df_viz.columns = ["texts", "x", "y"]

    return generate_chart(df_viz)


def get_avg_embeddings(texts: List[str], model, tokenizer) -> np.ndarray:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=32,
    )
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    embeddings = torch.sum(
        last_hidden_states * torch.unsqueeze(inputs["attention_mask"], -1), 1
    ) / torch.sum(inputs["attention_mask"], -1, keepdim=True)
    return embeddings.detach().numpy()


def get_cls_embeddings(texts: List[str], model, tokenizer) -> np.ndarray:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=32,
    )
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    embeddings = last_hidden_states[:, 0, :]
    return embeddings.detach().numpy()


def test_embeddings(texts, embeddings, model, tokenizer):
    avg_embeddings = get_avg_embeddings(texts, model, tokenizer)
    cls_embeddings = get_cls_embeddings(texts, model, tokenizer)

    is_avg_embeddings = np.allclose(avg_embeddings, embeddings, atol=1e-3)
    is_cls_embeddings = np.allclose(cls_embeddings, embeddings, atol=1e-3)

    if is_avg_embeddings or is_cls_embeddings:
        print("YAYYY!! Test passed! 🎉🎉🎉\n\n\n")
        return passed()
    else:
        print("Oh no! Test failed! 😢")
        return None


def get_similarity(target, candidates, top_k: int = 3) -> Dict[int, float]:
    # Turn list into array
    candidates = np.array(candidates)
    target = np.expand_dims(np.array(target), axis=0)

    # Calculate cosine similarity
    sim = cosine_similarity(target, candidates)
    sim = np.squeeze(sim).tolist()

    sort_index = np.argsort(sim)[::-1][:top_k]
    sort_score = [sim[i] for i in sort_index][:top_k]
    similarity_scores = dict(zip(sort_index, sort_score))

    # Return similarity scores
    return similarity_scores


def test_similarity(target, candidates, similarity, top_k: int = 3):
    actual_similarity = get_similarity(target, candidates, top_k)
    if actual_similarity == similarity:
        print("YAYYY!! Test passed! 🎉🎉🎉\n\n\n")
        return passed()
    else:
        print("Oh no! Test failed! 😢")
        print(
            f"Similarity scores do not match! Expected: {similarity}, Actual: {actual_similarity}"
        )
        return None
