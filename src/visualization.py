"""Plotly figures and safe image serialization for UI downloads."""
from __future__ import annotations
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
import plotly.express as px


def comparison_chart(frame: pd.DataFrame, metric: str):
    return px.bar(frame, x="algorithm", y=metric, color="algorithm", template="plotly_white", title=metric.replace("_", " ").title())


def png_bytes(image: np.ndarray) -> bytes:
    output = image if image.ndim == 2 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", output)
    if not ok:
        raise ValueError("Unable to encode image for download.")
    return encoded.tobytes()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")
