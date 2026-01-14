"""
Comprehensive evaluation metrics for medical report generation.
"""
import re
import logging
from typing import List, Dict, Optional

import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class MedicalReportMetrics:
    """
    Comprehensive metrics for medical report generation evaluation.
    
    Includes:
    - BLEU (1-4)
    - ROUGE (1, 2, L)
    - METEOR
    - BERTScore
    - CheXbert F1
    """
    
    def __init__(self, normalization_config: Optional[Dict[str, bool]] = None):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction().method1
        self._chexbert_scorer: Optional[object] = None
        self._meteor_scorer: Optional[object] = None
        self.normalization_config = normalization_config or {}
    
    def _normalize_text(self, text: str, remove_punctuation: Optional[bool] = None) -> str:
        """
        Normalize text for fair n-gram based metric computation.
        
        Handles edge cases that can unfairly penalize BLEU/ROUGE scores:
        - Case sensitivity (lowercasing)
        - Structural prefixes (Findings:, Impression:)
        - Unicode variants (curly quotes, en-dashes)
        - Punctuation (optional, for fair tokenization)
        - Whitespace normalization
        
        Args:
            text: Raw text
            remove_punctuation: Whether to remove punctuation for n-gram fairness
            
        Returns:
            Normalized text ready for metric computation
        """
        # Handle None and non-string types
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        # Handle empty strings
        text = text.strip()
        if not text:
            return ""
        
        # Lowercase for case-insensitive comparison
        if self.normalization_config.get("lowercase", True):
            text = text.lower()
        
        # Remove structural markers from our output format
        # Handles both standard format "Findings: ... | Impression: ..."
        # AND non-standard formats where model deviates
        #
        # Order matters: remove | first, then labels anywhere in text
        
        if self.normalization_config.get("remove_prefixes", True):
            # Step 1: Remove pipe separator (with surrounding whitespace)
            text = re.sub(r'\s*\|\s*', ' ', text)
            
            # Step 2: Remove "findings:" label ANYWHERE in text (not just start)
            # This handles: "Findings: text" AND "Some text. Findings: more text"
            text = re.sub(r'findings:\s*', '', text, flags=re.IGNORECASE)
            
            # Step 3: Remove "impression:" label ANYWHERE in text
            # This handles: "Impression: text" AND "Some text. Impression: more text"
            text = re.sub(r'impression:\s*', '', text, flags=re.IGNORECASE)
        
        # Normalize unicode characters (smart quotes, dashes)
        # This prevents tokenization mismatches
        unicode_replacements = {
            ''': "'", ''': "'",           # Smart single quotes
            '"': '"', '"': '"',           # Smart double quotes
            '–': '-', '—': '-',           # En-dash, em-dash to hyphen
            '…': '...',                   # Ellipsis
            '\u00a0': ' ',                # Non-breaking space
            '\u2018': "'", '\u2019': "'", # More quote variants
            '\u201c': '"', '\u201d': '"',
        }
        for old_char, new_char in unicode_replacements.items():
            text = text.replace(old_char, new_char)
        
        # Remove punctuation for fair n-gram comparison
        # "normal." and "normal" should match as same token
        if remove_punctuation is None:
            remove_punctuation = self.normalization_config.get("remove_punctuation", True)
        if remove_punctuation:
            # Keep alphanumeric and whitespace only
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize multiple whitespaces to single space
        if self.normalization_config.get("normalize_whitespace", True):
            text = ' '.join(text.split())
        
        return text.strip()
    
    def compute_bleu(
        self,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """
        Compute BLEU scores (1-4) with proper edge case handling.
        
        Args:
            references: List of reference lists (each reference is a list with one string)
            hypotheses: List of hypothesis strings
            
        Returns:
            Dictionary of BLEU scores
        """
        # Tokenize with normalization
        refs_tokenized = []
        hyps_tokenized = []
        valid_pairs = 0
        
        for ref, hyp in zip(references, hypotheses):
            # Handle reference format
            ref_text = ref[0] if isinstance(ref, list) else ref
            
            # Normalize both
            normalized_ref = self._normalize_text(ref_text)
            normalized_hyp = self._normalize_text(hyp)
            
            # Skip pairs where either is empty (would unfairly skew scores)
            if not normalized_ref or not normalized_hyp:
                logger.debug(f"Skipping empty pair - ref: '{normalized_ref[:50]}...', hyp: '{normalized_hyp[:50]}...'")
                continue
            
            # Tokenize using NLTK
            ref_tokens = nltk.word_tokenize(normalized_ref)
            hyp_tokens = nltk.word_tokenize(normalized_hyp)
            
            # Skip if tokenization results in empty
            if not ref_tokens or not hyp_tokens:
                continue
            
            refs_tokenized.append([ref_tokens])
            hyps_tokenized.append(hyp_tokens)
            valid_pairs += 1
        
        # Handle case where no valid pairs exist
        if valid_pairs == 0:
            logger.warning("No valid reference-hypothesis pairs for BLEU computation")
            return {f'bleu_{n}': 0.0 for n in range(1, 5)}
        
        logger.info(f"Computing BLEU on {valid_pairs} valid pairs")
        
        # Compute corpus BLEU for each n-gram
        bleu_scores = {}
        
        for n in range(1, 5):
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            try:
                score = corpus_bleu(
                    refs_tokenized,
                    hyps_tokenized,
                    weights=weights,
                    smoothing_function=self.smoothing
                )
            except Exception as e:
                logger.warning(f"BLEU-{n} computation failed: {e}")
                score = 0.0
            bleu_scores[f'bleu_{n}'] = score
        
        return bleu_scores
    
    def compute_rouge(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores with proper edge case handling.
        
        Args:
            references: List of reference strings
            hypotheses: List of hypothesis strings
            
        Returns:
            Dictionary of ROUGE scores
        """
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        valid_pairs = 0
        
        for ref, hyp in zip(references, hypotheses):
            normalized_ref = self._normalize_text(ref)
            normalized_hyp = self._normalize_text(hyp)
            
            # Skip empty pairs
            if not normalized_ref or not normalized_hyp:
                continue
            
            try:
                scores = self.rouge_scorer.score(normalized_ref, normalized_hyp)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
                valid_pairs += 1
            except Exception as e:
                logger.warning(f"ROUGE computation failed for pair: {e}")
                continue
        
        # Handle case where no valid pairs exist
        if valid_pairs == 0:
            logger.warning("No valid reference-hypothesis pairs for ROUGE computation")
            return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
        
        return {
            'rouge_1': np.mean(rouge_scores['rouge1']),
            'rouge_2': np.mean(rouge_scores['rouge2']),
            'rouge_l': np.mean(rouge_scores['rougeL']),
        }
    
    def compute_meteor(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> float:
        """
        Compute METEOR score.
        
        Args:
            references: List of reference strings
            hypotheses: List of hypothesis strings
            
        Returns:
            Average METEOR score
        """
        try:
            if self._meteor_scorer is None:
                from pycocoevalcap.meteor.meteor import Meteor
                self._meteor_scorer = Meteor()
            
            # Format for COCO eval
            refs_dict = {}
            hyps_dict = {}
            
            for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
                refs_dict[i] = [self._normalize_text(ref)]
                hyps_dict[i] = [self._normalize_text(hyp)]
            
            score, _ = self._meteor_scorer.compute_score(refs_dict, hyps_dict)
            return score
            
        except Exception as e:
            logger.warning(f"METEOR computation failed: {e}")
            return 0.0
    
    def compute_bertscore(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """
        Compute BERTScore.
        
        Args:
            references: List of reference strings
            hypotheses: List of hypothesis strings
            
        Returns:
            Dictionary with precision, recall, F1
        """
        try:
            # Normalize texts
            refs_norm = [self._normalize_text(r) for r in references]
            hyps_norm = [self._normalize_text(h) for h in hypotheses]
            
            P, R, F1 = bert_score(
                hyps_norm,
                refs_norm,
                model_type="microsoft/deberta-xlarge-mnli",
                lang="en",
                verbose=False,
            )
            
            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item(),
            }
            
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0,
            }
    
    def compute_chexbert(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """
        Compute CheXbert F1 score.
        
        Args:
            references: List of reference strings
            hypotheses: List of hypothesis strings
            
        Returns:
            Dictionary with precision, recall, F1
        """
        try:
            if self._chexbert_scorer is None:
                from f1chexbert import F1CheXbert
                self._chexbert_scorer = F1CheXbert()
            
            accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = self._chexbert_scorer(
                hyps=hypotheses,
                refs=references,
            )
            
            return {
                'chexbert_precision': chexbert_all['precision'],
                'chexbert_recall': chexbert_all['recall'],
                'chexbert_f1': chexbert_all['f1'],
            }
            
        except Exception as e:
            logger.warning(f"CheXbert computation failed: {e}")
            return {
                'chexbert_precision': 0.0,
                'chexbert_recall': 0.0,
                'chexbert_f1': 0.0,
            }
    
    def compute_all_metrics(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            references: List of reference strings
            hypotheses: List of hypothesis strings
            
        Returns:
            Dictionary with all metrics
        """
        all_metrics = {}
        
        # BLEU
        refs_formatted = [[ref] for ref in references]
        bleu_scores = self.compute_bleu(refs_formatted, hypotheses)
        all_metrics.update(bleu_scores)
        
        # ROUGE
        rouge_scores = self.compute_rouge(references, hypotheses)
        all_metrics.update(rouge_scores)
        
        # METEOR
        meteor_score = self.compute_meteor(references, hypotheses)
        all_metrics['meteor'] = meteor_score
        
        # BERTScore
        bertscore_scores = self.compute_bertscore(references, hypotheses)
        all_metrics.update(bertscore_scores)
        
        # CheXbert
        chexbert_scores = self.compute_chexbert(references, hypotheses)
        all_metrics.update(chexbert_scores)
        
        return all_metrics
