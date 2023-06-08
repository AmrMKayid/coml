import csv
import torch
import numpy as np
import pandas as pd
import altair as alt

from typing import List, Tuple, Dict, Union


from sklearn.decomposition import PCA


def get_pc(embeddings: np.ndarray, n: int = 2,) -> np.ndarray:
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
    embeds_pc2 = get_pc(np.array(df['text_embeddings'].tolist()), 2)

    df_viz = pd.concat([df["texts"], pd.DataFrame(embeds_pc2)], axis=1)
    df_viz.columns = ['texts', "x", "y"]

    return generate_chart(df_viz)
