"""
Text preprocessing utilities for medical reports.
"""
import re
from typing import Optional


class TextPreprocessor:
    """
    Preprocessor for medical report text.
    Handles cleaning, formatting, and structured output generation.
    """
    
    def __init__(
        self,
        output_template: str = "Findings: {findings} | Impression: {impression}",
        max_length: int = 512,
    ):
        self.output_template = output_template
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean medical report text.
        
        Removes sentences containing PHI markers (XXXX) to prevent the model
        from learning to generate placeholder text. Only the affected sentences
        are removed; clinical content in other sentences is preserved.
        
        Args:
            text: Raw text from report
            
        Returns:
            Cleaned text with PHI-containing sentences removed
        """
        if not isinstance(text, str) or not text:
            return ""
        
        # Strip whitespace
        text = text.strip()
        
        # Remove sentences containing PHI markers (XXXX patterns)
        # This prevents model from learning to generate redaction markers
        text = self._remove_phi_sentences(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix sentence spacing
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def _remove_phi_sentences(self, text: str) -> str:
        """
        Remove sentences containing PHI markers (XXXX patterns).
        
        Medical datasets often use XXXX to de-identify Protected Health Information.
        Rather than replacing with placeholders (which the model might generate),
        we remove entire sentences containing PHI markers while preserving
        clinically relevant content.
        
        Args:
            text: Text potentially containing PHI markers
            
        Returns:
            Text with PHI-containing sentences removed
        """
        # Pattern to match XXXX (one or more X's, common PHI marker)
        phi_pattern = re.compile(r'X{3,}', re.IGNORECASE)
        
        # Split into sentences using common medical report delimiters
        # Handles: periods, but preserves abbreviations like "Dr." or "vs."
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$')
        
        # Split text into sentences
        sentences = sentence_pattern.split(text)
        
        # If splitting didn't work well, try simpler approach
        if len(sentences) <= 1:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out sentences containing PHI markers
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not phi_pattern.search(sentence):
                clean_sentences.append(sentence)
        
        # Rejoin sentences
        result = ' '.join(clean_sentences)
        
        # Ensure proper sentence ending
        if result and not result.endswith(('.', '!', '?')):
            result += '.'
        
        return result
    
    def format_structured_output(
        self,
        findings: Optional[str],
        impression: Optional[str],
    ) -> str:
        """
        Format findings and impression into structured output.
        
        Args:
            findings: Findings section text
            impression: Impression section text
            
        Returns:
            Structured output string
        """
        findings_clean = self.clean_text(findings) if findings else ""
        impression_clean = self.clean_text(impression) if impression else ""
        
        # Handle missing sections
        if not findings_clean and not impression_clean:
            return "Findings: No findings documented. | Impression: No impression documented."
        
        if not findings_clean:
            findings_clean = "No detailed findings documented."
        
        if not impression_clean:
            impression_clean = "No specific impression documented."
        
        return self.output_template.format(
            findings=findings_clean,
            impression=impression_clean
        )
    
    def normalize_for_metrics(self, text: str, remove_prefixes: bool = True) -> str:
        """
        Normalize text for metric computation.
        
        Args:
            text: Text to normalize
            remove_prefixes: Whether to remove Findings:/Impression: prefixes
            
        Returns:
            Normalized text
        """
        text = text.lower()
        
        if remove_prefixes:
            # Remove Findings: and Impression: prefixes
            text = re.sub(r'^findings:\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*\|\s*impression:\s*', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'^impression:\s*', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_sections(self, structured_text: str) -> dict:
        """
        Extract findings and impression from structured text.
        
        Args:
            structured_text: Text in format "Findings: ... | Impression: ..."
            
        Returns:
            Dictionary with 'findings' and 'impression' keys
        """
        result = {"findings": "", "impression": ""}
        
        # Try to split by |
        if "|" in structured_text:
            parts = structured_text.split("|")
            for part in parts:
                part = part.strip()
                if part.lower().startswith("findings:"):
                    result["findings"] = part[9:].strip()
                elif part.lower().startswith("impression:"):
                    result["impression"] = part[11:].strip()
        else:
            # Try to find sections without |
            findings_match = re.search(
                r'findings:\s*(.*?)(?=impression:|$)',
                structured_text,
                re.IGNORECASE | re.DOTALL
            )
            impression_match = re.search(
                r'impression:\s*(.*?)$',
                structured_text,
                re.IGNORECASE | re.DOTALL
            )
            
            if findings_match:
                result["findings"] = findings_match.group(1).strip()
            if impression_match:
                result["impression"] = impression_match.group(1).strip()
        
        return result


def strip_prompt(text: str, prompt_template: str) -> str:
    """Strip prompt/template text from generated output."""
    if not text:
        return text
    cleaned = text.strip()
    # Cut everything up to the last "Report:" marker if present
    report_match = re.search(r'report\s*:\s*', cleaned, flags=re.IGNORECASE)
    if report_match:
        cleaned = cleaned[report_match.end():].strip()
    # Fallback: remove the prompt prefix if present
    prompt = (prompt_template or "").strip()
    if prompt and cleaned.lower().startswith(prompt.lower()):
        cleaned = cleaned[len(prompt):].strip()
    return cleaned


def enforce_impression_consistency(structured_text: str) -> str:
    """
    If findings contain abnormal cues, prevent 'No acute abnormality' impressions.
    This is a lightweight post-hoc guardrail.
    """
    if not structured_text:
        return structured_text
    preprocessor = TextPreprocessor()
    sections = preprocessor.extract_sections(structured_text)
    findings = sections.get("findings", "")
    impression = sections.get("impression", "")
    if not impression:
        return structured_text
    # Abnormal cue list (from CheXbert labels)
    abnormal_terms = [
        "cardiomegaly", "pneumothorax", "pleural effusion", "effusion",
        "consolidation", "pneumonia", "edema", "atelectasis", "fracture",
        "lung opacity", "lung lesion", "support device", "pleural", "opacity",
    ]
    impression_negatives = [
        "no acute abnormality",
        "no acute cardiopulmonary abnormality",
        "no acute cardiopulmonary findings",
        "no acute disease",
    ]
    findings_lower = findings.lower()
    impression_lower = impression.lower()
    if any(term in findings_lower for term in abnormal_terms) and any(
        phrase in impression_lower for phrase in impression_negatives
    ):
        new_impression = "Abnormal findings noted. See findings for details."
        return preprocessor.output_template.format(
            findings=findings.strip(),
            impression=new_impression,
        )
    return structured_text


def postprocess_generated_report(
    text: str,
    prompt_template: str,
    apply_prompt_strip: bool = True,
    apply_impression_consistency: bool = False,
) -> str:
    """Apply post-processing steps to generated reports."""
    output = text or ""
    if apply_prompt_strip:
        output = strip_prompt(output, prompt_template)
    if apply_impression_consistency:
        output = enforce_impression_consistency(output)
    return output


def get_prompt_template() -> str:
    """Get the generation prompt template."""
    return """Generate a structured radiology report for this chest X-ray image.

Format your response as:
Findings: [Detailed observations about the chest X-ray including heart size, lung fields, mediastinum, bones, and any abnormalities]
| Impression: [Clinical conclusion summarizing the key findings and recommendations]

Report:"""
