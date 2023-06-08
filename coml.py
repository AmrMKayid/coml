import csv
import torch
import random
import numpy as np
import pandas as pd
import altair as alt

from typing import List, Tuple, Dict, Union
from sklearn.decomposition import PCA
from transformers import DistilBertTokenizer, DistilBertModel


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
        print("YAYYY!! Test passed! 🎉🎉🎉")
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
    else:
        print("Oh no! Test failed! 😢")
        assert np.allclose(avg_embeddings, embeddings, atol=1e-3) and np.allclose(
            cls_embeddings, embeddings, atol=1e-3
        ), f"Your embeddings {embeddings} are not correct! It should be {avg_embeddings} or {cls_embeddings}"
