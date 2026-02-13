"""
Create a balanced subset of the MIMIC-CXR dataset for training.

This module reads every report in the MIMIC-CXR dataset, applies
keyword-based condition detection aligned with the 14 CheXbert labels,
and produces a balanced CSV that ensures adequate representation of
all conditions — including rare ones that were nearly absent in the
smaller IU X-ray dataset.

Usage (standalone):
    python -m data.create_balanced_sample \
        --data-root mimic-cxr-dataset \
        --target-total 10000 \
        --output outputs/mimic_cxr_balanced.csv

Usage (via main.py):
    python main.py balance --target-total 10000
"""
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from data.preprocessing import parse_mimic_report_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# 14 CheXbert condition labels (same order as data_config.yaml)
# ---------------------------------------------------------------
CHEXBERT_CONDITIONS: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# ---------------------------------------------------------------
# Keyword patterns for each condition.
# Each entry is a list of (positive_pattern, negative_context) tuples.
# A study is labelled positive for a condition if *any* positive
# pattern matches AND the match is NOT inside a negation context.
# ---------------------------------------------------------------
_NEGATION_WINDOW = 60  # characters to look back for negation cues

_NEGATION_PREFIXES = re.compile(
    r"(no |no\s+evidence|without |absent |not |never |negative for |"
    r"rules?\s+out|excluded |free of |resolved |cleared |removed )",
    re.IGNORECASE,
)

_CONDITION_PATTERNS: Dict[str, List[re.Pattern]] = {
    "No Finding": [
        re.compile(r"no acute .{0,30}(cardiopulmonary|thoracic|intrathoracic|pulmonary|process|abnormal|finding|disease)", re.I),
        re.compile(r"(normal|unremarkable) (chest|cardiopulmonary|study|exam|radiograph)", re.I),
        re.compile(r"lungs are clear", re.I),
        re.compile(r"clear lungs", re.I),
        re.compile(r"no (focal|acute) (consolidation|opacity|abnormality)", re.I),
    ],
    "Enlarged Cardiomediastinum": [
        re.compile(r"enlarged (cardio)?mediastin", re.I),
        re.compile(r"mediastinal (widening|enlargement|prominence)", re.I),
        re.compile(r"widened mediastin", re.I),
    ],
    "Cardiomegaly": [
        re.compile(r"cardiomegal", re.I),
        re.compile(r"enlarged (cardiac|heart)", re.I),
        re.compile(r"heart.{0,15}(enlarged|prominent|increased)", re.I),
        re.compile(r"cardiac.{0,15}(enlarged|enlargement|silhouette.{0,15}(enlarged|prominent))", re.I),
    ],
    "Lung Opacity": [
        re.compile(r"lung opacit", re.I),
        re.compile(r"pulmonary opacit", re.I),
        re.compile(r"(airspace|parenchymal) opaci", re.I),
        re.compile(r"opacification.{0,20}(lung|lobe|pulmonary)", re.I),
        re.compile(r"(hazy|haziness|ground.glass) opaci", re.I),
    ],
    "Lung Lesion": [
        re.compile(r"lung (lesion|mass|nodule)", re.I),
        re.compile(r"pulmonary (lesion|mass|nodule)", re.I),
        re.compile(r"(nodular|mass).{0,20}(lung|pulmonary|lobe)", re.I),
    ],
    "Edema": [
        re.compile(r"(pulmonary |interstitial )?edema", re.I),
        re.compile(r"fluid overload", re.I),
        re.compile(r"vascular congestion", re.I),
        re.compile(r"cephalization", re.I),
    ],
    "Consolidation": [
        re.compile(r"consolidat", re.I),
    ],
    "Pneumonia": [
        re.compile(r"pneumonia", re.I),
        re.compile(r"infectious.{0,15}(process|infiltrate)", re.I),
    ],
    "Atelectasis": [
        re.compile(r"atelecta", re.I),
    ],
    "Pneumothorax": [
        re.compile(r"pneumothorax", re.I),
    ],
    "Pleural Effusion": [
        re.compile(r"pleural effusion", re.I),
        re.compile(r"(small|moderate|large|bilateral|left|right).{0,10}effusion", re.I),
        re.compile(r"effusion.{0,15}(pleural|layering|blunting)", re.I),
        re.compile(r"(costophrenic|meniscus).{0,15}(blunt|effusion)", re.I),
    ],
    "Pleural Other": [
        re.compile(r"pleural (thicken|calcif|plaque|abnormal)", re.I),
        re.compile(r"(thicken|calcif).{0,15}pleur", re.I),
        re.compile(r"empyema", re.I),
    ],
    "Fracture": [
        re.compile(r"fracture", re.I),
        re.compile(r"(rib|sternal|vertebral|clavicl).{0,15}(fracture|deformit)", re.I),
    ],
    "Support Devices": [
        re.compile(r"(support device|line|tube|catheter|port|pacer|pacemaker|icd|aicd|defibrillator)", re.I),
        re.compile(r"(endotracheal|nasogastric|enteric|chest) tube", re.I),
        re.compile(r"(central venous|picc|swan.ganz|dialysis).{0,10}(catheter|line)", re.I),
        re.compile(r"tracheostomy", re.I),
        re.compile(r"sternotomy", re.I),
    ],
}

# View-position mapping: MIMIC ViewPosition -> canonical category
VIEW_MAP: Dict[str, str] = {
    "PA": "Frontal",
    "AP": "Frontal",
    "LATERAL": "Lateral",
    "LL": "Lateral",
}

# Frontal-view preference (PA > AP)
FRONTAL_PREFERENCE = ["PA", "AP"]


# ---------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------

def _is_negated(text: str, match_start: int) -> bool:
    """Check whether a regex match is preceded by a negation cue."""
    window_start = max(0, match_start - _NEGATION_WINDOW)
    window = text[window_start:match_start]
    return bool(_NEGATION_PREFIXES.search(window))


def keyword_label_study(findings: str, impression: str) -> List[int]:
    """
    Assign keyword-based binary labels for all 14 CheXbert conditions.

    The combined text of findings + impression is searched.  A condition
    is positive (1) only if a keyword pattern fires AND the match is
    NOT inside a negation window.

    Args:
        findings: Extracted FINDINGS section text.
        impression: Extracted IMPRESSION section text.

    Returns:
        List of 14 integers (0 or 1), one per CheXbert condition.
    """
    combined = f"{findings} {impression}".strip()
    if not combined:
        return [0] * len(CHEXBERT_CONDITIONS)

    labels = []
    for condition in CHEXBERT_CONDITIONS:
        patterns = _CONDITION_PATTERNS.get(condition, [])
        found = False
        for pat in patterns:
            for m in pat.finditer(combined):
                if not _is_negated(combined, m.start()):
                    found = True
                    break
            if found:
                break
        labels.append(int(found))

    # If *any* disease condition is positive, clear No Finding
    if any(labels[1:]):
        labels[0] = 0

    return labels


def _build_study_records(
    data_root: Path,
    metadata_csv: str = "metadata.csv",
    images_dir: str = "official_data_iccv_final/files",
    reports_dir: str = "mimic-cxr-reports/files",
) -> pd.DataFrame:
    """
    Read metadata, parse every report, and build a study-level DataFrame.

    Returns a DataFrame with one row per (study_id, dicom_id) containing:
        subject_id, study_id, dicom_id, view_position, image_path,
        report_path, findings, impression, keyword_labels (JSON string)
    """
    logger.info("Reading metadata.csv ...")
    meta_df = pd.read_csv(data_root / metadata_csv)
    logger.info(f"  Total image records: {len(meta_df)}")

    # Derive directory prefix: first 3 chars of subject_id → "p" + first 2 digits
    # e.g. subject_id=10000032 → prefix "p10"
    meta_df["prefix"] = "p" + meta_df["subject_id"].astype(str).str[:2]

    records: List[Dict[str, Any]] = []
    studies_seen: set = set()
    study_reports: Dict[int, Dict[str, str]] = {}  # study_id -> {findings, impression}

    # Group by study_id to parse each report only once
    grouped = meta_df.groupby("study_id")
    total_studies = len(grouped)
    logger.info(f"  Total unique studies: {total_studies}")
    logger.info("Parsing reports and building study records ...")

    for idx, (study_id, group) in enumerate(grouped):
        if idx > 0 and idx % 5000 == 0:
            logger.info(f"  Processed {idx}/{total_studies} studies ...")

        row0 = group.iloc[0]
        subject_id = int(row0["subject_id"])
        prefix = row0["prefix"]

        # Parse report (once per study)
        if study_id not in study_reports:
            report_path = (
                data_root
                / reports_dir
                / prefix
                / f"p{subject_id}"
                / f"s{study_id}.txt"
            )
            if report_path.exists():
                report_text = report_path.read_text(encoding="utf-8", errors="replace")
                parsed = parse_mimic_report_text(report_text)
            else:
                parsed = {"findings": "", "impression": ""}
            study_reports[study_id] = parsed

        findings = study_reports[study_id]["findings"]
        impression = study_reports[study_id]["impression"]

        # Skip studies without any useful text
        if not findings.strip() and not impression.strip():
            continue

        # Keyword labels
        kw_labels = keyword_label_study(findings, impression)

        # Build one record per image (will select best later)
        for _, img_row in group.iterrows():
            dicom_id = img_row["dicom_id"]
            view_pos = str(img_row.get("ViewPosition", "")).strip()
            image_path = (
                data_root
                / images_dir
                / prefix
                / f"p{subject_id}"
                / f"s{study_id}"
                / f"{dicom_id}.jpg"
            )
            records.append({
                "study_id": int(study_id),
                "subject_id": subject_id,
                "dicom_id": dicom_id,
                "view_position": view_pos,
                "view_category": VIEW_MAP.get(view_pos, "Other"),
                "image_path": str(image_path),
                "report_path": str(
                    data_root / reports_dir / prefix / f"p{subject_id}" / f"s{study_id}.txt"
                ),
                "findings": findings,
                "impression": impression,
                "keyword_labels": json.dumps(kw_labels),
            })

    df = pd.DataFrame(records)
    logger.info(f"  Built {len(df)} image-level records across {df['study_id'].nunique()} studies")
    return df


def _select_best_frontal(group: pd.DataFrame) -> Optional[pd.Series]:
    """Pick the best frontal image for a study (PA preferred over AP)."""
    frontals = group[group["view_category"] == "Frontal"]
    if frontals.empty:
        return None
    for vp in FRONTAL_PREFERENCE:
        match = frontals[frontals["view_position"] == vp]
        if not match.empty:
            return match.iloc[0]
    return frontals.iloc[0]


def create_balanced_sample(
    data_root: str,
    target_total: int = 10000,
    output_path: str = "outputs/mimic_cxr_balanced.csv",
    metadata_csv: str = "metadata.csv",
    images_dir: str = "official_data_iccv_final/files",
    reports_dir: str = "mimic-cxr-reports/files",
    min_per_condition: int = 500,
    max_per_condition: int = 1500,
    max_no_finding: int = 1500,
    max_support_devices: int = 1500,
    seed: int = 42,
) -> Path:
    """
    Create a balanced MIMIC-CXR subset CSV.

    Algorithm:
      1. Parse all reports → keyword labels per study.
      2. Select best frontal image per study.
      3. Compute per-condition counts.
      4. Priority inclusion: for rare conditions (count < min_per_condition
         globally), include ALL positive studies.  For common conditions
         cap at ``max_per_condition``.
      5. Cap over-represented conditions (No Finding, Support Devices).
      6. Fill remaining quota with proportional random sampling.
      7. Save to CSV with per-study metadata.

    Args:
        data_root: Path to ``mimic-cxr-dataset/`` directory.
        target_total: Target number of studies in the balanced subset.
        output_path: Where to write the balanced CSV.
        metadata_csv: Filename of metadata CSV inside data_root.
        images_dir: Relative path to image directory inside data_root.
        reports_dir: Relative path to reports directory inside data_root.
        min_per_condition: Minimum positive studies per condition.
        max_per_condition: Maximum positive studies per disease condition.
        max_no_finding: Maximum No Finding-only studies to include.
        max_support_devices: Maximum Support Devices-only studies.
        seed: Random seed for reproducibility.

    Returns:
        Path to the written CSV.
    """
    rng = np.random.RandomState(seed)
    data_root = Path(data_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Build all study records
    all_records = _build_study_records(
        data_root, metadata_csv, images_dir, reports_dir,
    )

    # Step 2: Pick best frontal image per study
    logger.info("Selecting best frontal image per study ...")
    study_rows: List[pd.Series] = []
    for study_id, grp in all_records.groupby("study_id"):
        best = _select_best_frontal(grp)
        if best is not None:
            study_rows.append(best)

    study_df = pd.DataFrame(study_rows).reset_index(drop=True)
    logger.info(f"  Studies with valid frontal image: {len(study_df)}")

    # Parse keyword_labels column back to lists
    study_df["_labels"] = study_df["keyword_labels"].apply(json.loads)

    # Step 3: Compute per-condition counts (before balancing)
    logger.info("\n--- Condition prevalence (BEFORE balancing) ---")
    _print_condition_stats(study_df, CHEXBERT_CONDITIONS, prefix="  [FULL]")

    # Step 4: Balanced sampling
    selected_ids: set = set()

    # 4a: For each condition, ensure minimum representation
    for ci, cond in enumerate(CHEXBERT_CONDITIONS):
        positive_mask = study_df["_labels"].apply(lambda x, i=ci: x[i] == 1)
        positive_studies = study_df[positive_mask]

        if len(positive_studies) == 0:
            logger.warning(f"  No positive studies found for: {cond}")
            continue

        # Determine how many to include:
        #   - No Finding / Support Devices: use their dedicated caps
        #   - Rare conditions (fewer than min_per_condition): include ALL
        #   - Common conditions: cap at max_per_condition
        if cond == "No Finding":
            cap = max_no_finding
        elif cond == "Support Devices":
            cap = max_support_devices
        elif len(positive_studies) <= min_per_condition:
            cap = len(positive_studies)          # rare – take everything
        else:
            cap = max_per_condition               # common – cap

        # How many are already selected?
        already_in = positive_studies["study_id"].isin(selected_ids).sum()
        needed = max(0, min(cap, len(positive_studies)) - already_in)

        if needed > 0:
            available = positive_studies[~positive_studies["study_id"].isin(selected_ids)]
            n_take = min(needed, len(available))
            chosen = available.sample(n=n_take, random_state=rng)
            selected_ids.update(chosen["study_id"].tolist())

        logger.info(
            f"  {cond}: positives={len(positive_studies)}, "
            f"selected_so_far={len(study_df[study_df['study_id'].isin(selected_ids) & positive_mask])}"
        )

    logger.info(f"  After priority inclusion: {len(selected_ids)} studies")

    # 4b: Fill remaining quota with random sampling from unselected studies
    remaining = target_total - len(selected_ids)
    if remaining > 0:
        unselected = study_df[~study_df["study_id"].isin(selected_ids)]
        n_fill = min(remaining, len(unselected))
        filler = unselected.sample(n=n_fill, random_state=rng)
        selected_ids.update(filler["study_id"].tolist())
        logger.info(f"  Filled {n_fill} additional studies (total: {len(selected_ids)})")
    elif remaining < 0:
        logger.info(
            f"  Priority inclusion already reached {len(selected_ids)} studies "
            f"(target was {target_total}). Keeping all."
        )

    # Step 5: Build final balanced DataFrame
    balanced_df = study_df[study_df["study_id"].isin(selected_ids)].copy()
    balanced_df = balanced_df.drop(columns=["_labels"])
    balanced_df = balanced_df.sort_values("study_id").reset_index(drop=True)

    # Re-parse for stats
    balanced_df["_labels"] = balanced_df["keyword_labels"].apply(json.loads)
    logger.info(f"\n--- Condition prevalence (AFTER balancing) ---")
    _print_condition_stats(balanced_df, CHEXBERT_CONDITIONS, prefix="  [BAL]")
    balanced_df = balanced_df.drop(columns=["_labels"])

    # Step 6: Save
    balanced_df.to_csv(output_path, index=False)
    logger.info(f"\nBalanced sample saved to {output_path}")
    logger.info(f"  Total studies: {balanced_df['study_id'].nunique()}")
    logger.info(f"  Total patients: {balanced_df['subject_id'].nunique()}")

    return output_path


def _print_condition_stats(
    df: pd.DataFrame,
    conditions: List[str],
    prefix: str = "",
) -> None:
    """Print per-condition positive counts and percentages."""
    total = len(df)
    for ci, cond in enumerate(conditions):
        count = df["_labels"].apply(lambda x, i=ci: x[i] == 1).sum()
        pct = 100.0 * count / total if total > 0 else 0
        logger.info(f"{prefix} {cond:30s}: {count:6d} / {total:6d}  ({pct:5.1f}%)")


def create_balanced_sample_from_config(
    data_config: dict,
    target_total: Optional[int] = None,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> Path:
    """
    Config-driven wrapper for ``create_balanced_sample``.

    Reads dataset paths from the data configuration dictionary.

    Args:
        data_config: Parsed data_config.yaml dictionary.
        target_total: Override for target number of studies.
        output_path: Override for output CSV path.
        seed: Random seed.

    Returns:
        Path to the written CSV.
    """
    ds = data_config.get("dataset", {})
    bs = data_config.get("balanced_sampling", {})

    data_root = ds.get("data_root", "mimic-cxr-dataset")
    metadata_csv = ds.get("metadata_csv", "metadata.csv")
    images_dir = ds.get("images_dir", "official_data_iccv_final/files")
    reports_dir = ds.get("reports_dir", "mimic-cxr-reports/files")
    balanced_csv = output_path or ds.get("balanced_csv", "outputs/mimic_cxr_balanced.csv")

    _target = target_total or bs.get("target_total", 10000)
    _min_per = bs.get("min_per_condition", 500)
    _max_per = bs.get("max_per_condition", 1500)
    _max_nf = bs.get("max_no_finding", 1500)
    _max_sd = bs.get("max_support_devices", 1500)

    return create_balanced_sample(
        data_root=data_root,
        target_total=_target,
        output_path=balanced_csv,
        metadata_csv=metadata_csv,
        images_dir=images_dir,
        reports_dir=reports_dir,
        min_per_condition=_min_per,
        max_per_condition=_max_per,
        max_no_finding=_max_nf,
        max_support_devices=_max_sd,
        seed=seed,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Create balanced MIMIC-CXR subset")
    parser.add_argument("--data-root", default="mimic-cxr-dataset")
    parser.add_argument("--target-total", type=int, default=10000)
    parser.add_argument("--output", default="outputs/mimic_cxr_balanced.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_balanced_sample(
        data_root=args.data_root,
        target_total=args.target_total,
        output_path=args.output,
        seed=args.seed,
    )
