"""
Text preprocessing and tokenization utilities.
"""
import re
from typing import List


def clean_medical_text(text: str) -> str:
    """
    Clean medical report text.
    
    Args:
        text: Raw medical text
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    text = re.sub(r'XXXX+', '[MASKED]', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    return text.strip()


def preprocess_findings_and_impression(findings: str, impression: str) -> str:
    """
    Combine findings and impression into a single text.
    
    Args:
        findings: Findings section
        impression: Impression section
    
    Returns:
        Combined text
    """
    findings_clean = clean_medical_text(findings)
    impression_clean = clean_medical_text(impression)
    
    if findings_clean and impression_clean:
        return f"Findings: {findings_clean} Impression: {impression_clean}"
    elif impression_clean:
        return f"Impression: {impression_clean}"
    elif findings_clean:
        return f"Findings: {findings_clean}"
    else:
        return "No findings available."


def extract_impression_only(impression: str) -> str:
    """
    Extract and clean only the impression text.
    
    Args:
        impression: Impression section
    
    Returns:
        Cleaned impression
    """
    return clean_medical_text(impression)

