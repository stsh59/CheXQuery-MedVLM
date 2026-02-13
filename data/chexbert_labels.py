"""
CheXbert label precomputation utilities for MIMIC-CXR.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

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
    balanced_csv: str,
    output_path: Path,
    batch_size: int = 32,
) -> Path:
    """
    Compute CheXbert labels for all studies in the balanced CSV and save to JSON.

    The balanced CSV already contains parsed ``findings`` and ``impression``
    columns so we do not need to re-read report text files.

    Args:
        balanced_csv: Path to the balanced subset CSV.
        output_path: Where to save the JSON label file.
        batch_size: Batch size for the CheXbert labeler.

    Returns:
        Path to the written JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(balanced_csv)
    preprocessor = TextPreprocessor()
    labeler = _get_chexbert_labeler()

    labels: Dict[str, List[int]] = {}
    texts: List[str] = []
    study_ids: List[str] = []

    for _, row in df.iterrows():
        sid = str(int(row["study_id"]))
        if sid in labels:
            # Already processed (shouldn't happen with balanced CSV,
            # but guard against duplicate rows)
            continue
        findings = row.get("findings", "")
        impression = row.get("impression", "")
        report_text = preprocessor.format_structured_output(
            findings=findings if pd.notna(findings) else "",
            impression=impression if pd.notna(impression) else "",
        )
        texts.append(report_text)
        study_ids.append(sid)

    logger.info(f"Computing CheXbert labels for {len(texts)} studies ...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        if hasattr(labeler, "label"):
            batch_labels = labeler.label(batch_texts)
        elif hasattr(labeler, "get_label"):
            batch_labels = [labeler.get_label(t) for t in batch_texts]
        else:
            batch_labels = labeler(batch_texts)
        for sid, lbl in zip(study_ids[i : i + batch_size], batch_labels):
            labels[sid] = [int(x) for x in lbl]

    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)
    logger.info(f"Saved CheXbert labels to {output_path} ({len(labels)} studies)")
    return output_path


def compute_chexbert_labels_from_config(
    data_config: dict,
    output_path: Optional[str] = None,
    batch_size: int = 32,
) -> Path:
    """
    Config-driven wrapper for ``compute_chexbert_labels``.

    Reads the balanced CSV path from the data configuration dictionary.

    Args:
        data_config: Parsed data_config.yaml dictionary.
        output_path: Override for output JSON path.
        batch_size: Batch size for labeling.

    Returns:
        Path to the written JSON file.
    """
    dataset = data_config.get("dataset", {})
    balanced_csv = dataset.get("balanced_csv", "outputs/mimic_cxr_balanced.csv")
    output_path = Path(output_path or "outputs/chexbert_labels.json")
    return compute_chexbert_labels(balanced_csv, output_path, batch_size=batch_size)
