"""
CheXbert label precomputation utilities.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import kagglehub

from data.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


def _get_chexbert_labeler():
    try:
        from f1chexbert import F1CheXbert
        scorer = F1CheXbert()
        for attr in ["labeler", "chexbert", "model"]:
            obj = getattr(scorer, attr, None)
            if obj is not None and (hasattr(obj, "label") or hasattr(obj, "get_label")):
                return obj
        if hasattr(scorer, "label") or hasattr(scorer, "get_label"):
            return scorer
    except Exception as e:
        raise RuntimeError(f"CheXbert labeler unavailable: {e}") from e
    raise RuntimeError("CheXbert labeler not found in f1chexbert")


def compute_chexbert_labels(
    data_root: Path,
    reports_csv: str,
    output_path: Path,
    batch_size: int = 32,
) -> Path:
    """
    Compute CheXbert labels for all reports and save to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports_df = pd.read_csv(data_root / reports_csv)
    preprocessor = TextPreprocessor()
    labeler = _get_chexbert_labeler()

    labels: Dict[str, List[int]] = {}
    texts: List[str] = []
    uids: List[str] = []

    for _, row in reports_df.iterrows():
        uid = str(row["uid"])
        findings = row.get("findings", "")
        impression = row.get("impression", "")
        report_text = preprocessor.format_structured_output(
            findings=findings if pd.notna(findings) else "",
            impression=impression if pd.notna(impression) else "",
        )
        texts.append(report_text)
        uids.append(uid)

    logger.info(f"Computing CheXbert labels for {len(texts)} reports...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        if hasattr(labeler, "label"):
            batch_labels = labeler.label(batch_texts)
        elif hasattr(labeler, "get_label"):
            batch_labels = [labeler.get_label(t) for t in batch_texts]
        else:
            batch_labels = labeler(batch_texts)
        for uid, lbl in zip(uids[i:i + batch_size], batch_labels):
            labels[uid] = [int(x) for x in lbl]

    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)
    logger.info(f"Saved CheXbert labels to {output_path}")
    return output_path


def compute_chexbert_labels_from_config(
    data_config: dict,
    output_path: Optional[str] = None,
    batch_size: int = 32,
) -> Path:
    dataset = data_config.get("dataset", {})
    data_root = Path(kagglehub.dataset_download(dataset.get("kaggle_dataset")))
    reports_csv = dataset.get("reports_csv", "indiana_reports.csv")
    output_path = Path(output_path or "outputs/chexbert_labels.json")
    return compute_chexbert_labels(data_root, reports_csv, output_path, batch_size=batch_size)
