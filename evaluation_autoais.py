#!/usr/bin/env python3
"""
AutoAIS-based evaluation module for citation quality assessment
Uses entailment-based verification for more precise evaluation
"""

import json
import re
import time
import random
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer, util
# transformers import removed (no longer using RoBERTa-SQuAD)
import torch
from sklearn.metrics import precision_recall_fscore_support
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gradio_client import Client
import os
from dotenv import load_dotenv
load_dotenv()
try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except Exception:
    sentence_bleu, SmoothingFunction = None, None
try:
    from bert_score import score as bert_score_compute
except Exception:
    bert_score_compute = None

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpaceAutoAIS:
    """
    Uses the hf_true_space_client to call Gradio Space for AutoAIS evaluation.
    IMPROVED SMART TRUNCATION VERSION - incorporates better citation quality evaluation logic.
    """
    
    def __init__(self, space_slug: str = None, hf_token: str = None, route: str = "/predict_greedy", timeout: int = 1800):
        if space_slug is None:
            space_slug = "za-zeeshan-33/true-model"
        
        if hf_token is None:
            hf_token = os.environ.get('HF_TOKEN')
        
        self.client = Client(space_slug, hf_token=hf_token)
        self.route = route
        self.timeout = timeout
    
    def _smart_truncate(self, text: str, max_length: int = 800) -> str:
        """
        Smart truncation that preserves the most relevant parts:
        1. Keep the beginning (title/context)
        2. Keep the end (conclusion/summary)
        3. If still too long, prioritize sentences with key terms
        """
        if len(text) <= max_length:
            return text
        
        # Strategy 1: Keep beginning and end with overlap
        half_length = max_length // 2
        beginning = text[:half_length]
        end = text[-half_length:]
        
        # Find a good break point in the middle
        middle_start = max(0, len(text) // 2 - 100)
        middle_end = min(len(text), len(text) // 2 + 100)
        middle_section = text[middle_start:middle_end]
        
        # Look for sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', middle_section)
        if len(sentences) > 1:
            # Take the first sentence from middle
            first_sentence = sentences[0]
            # Take the last sentence from middle
            last_sentence = sentences[-1]
            
            # Combine: beginning + first_sentence + "..." + last_sentence + end
            result = beginning + first_sentence + " ... " + last_sentence + end
        else:
            result = beginning + " ... " + end
        
        # Ensure we don't exceed max_length
        if len(result) > max_length:
            result = result[:max_length-3] + "..."
        
        return result
    
    def __call__(self, premise: str, hypothesis: str, threshold: float = 0.5) -> int:
        """
        Run AutoAIS inference using the Space client.
        Returns 1 if premise entails hypothesis, 0 otherwise.
        IMPROVED SMART TRUNCATION VERSION - incorporates better citation quality evaluation logic.
        """
        try:
            # Smart truncation to prevent memory issues
            max_premise_length = 2000  # characters
            max_hypothesis_length = 600  # characters
            
            original_premise_len = len(premise)
            original_hypothesis_len = len(hypothesis)
            
            if len(premise) > max_premise_length:
                logger.info(f"[SpaceAutoAIS-IMPROVED] Smart truncating premise from {len(premise)} to {max_premise_length} chars")
                premise = self._smart_truncate(premise, max_premise_length)
            
            if len(hypothesis) > max_hypothesis_length:
                logger.info(f"[SpaceAutoAIS-IMPROVED] Truncating hypothesis from {len(hypothesis)} to {max_hypothesis_length} chars")
                hypothesis = hypothesis[:max_hypothesis_length] + "..."
            
            # Use the appropriate API format based on endpoint
            if self.route == "/predict_threshold":
                result = self.client.predict(premise, hypothesis, threshold, api_name=self.route)
            else:
                # Use greedy endpoint (default) - no threshold parameter needed
                result = self.client.predict(premise, hypothesis, api_name=self.route)
            
            # Parse the result
            if isinstance(result, dict) and "label" in result:
                label = str(result["label"]).strip()
                return int(label) if label else 0
            else:
                return 0
                
        except Exception as e:
            logger.error(f"[SpaceAutoAIS-IMPROVED] Error during AutoAIS inference: {e}")
            return 0


def entails_autoais(autoais, premise: str, hypothesis: str) -> int:
    """Run AutoAIS entailment check - returns 1 if premise entails hypothesis, 0 otherwise"""
    logger.info(f"Testing AutoAIS entailment (IMPROVED): Premise: {premise[:100]}... Hypothesis: {hypothesis[:100]}...")
    result = autoais(premise, hypothesis)
    logger.info(f"AutoAIS result: {result}")
    return result


def sentence_split(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def citation_quality_autoais(answer: str, cited_docs: List[Dict], autoais, entailment_delay: float = 1.0) -> Dict[str, float]:
    """Citation quality using AutoAIS model - FIXED to match ALCE precision calculation"""
    import copy
    
    def _format_document(doc):
        """Format document for AutoAIS - improved robustness"""
        if isinstance(doc, dict):
            if "sent" in doc:
                # QA-extracted docs
                return "Title: %s\n%s" % (doc.get('title', ''), doc.get('sent', ''))
            else:
                return "Title: %s\n%s" % (doc.get('title', ''), doc.get('text', ''))
        else:
            return str(doc)
    
    def remove_citations(text):
        """Remove citations from text"""
        return re.sub(r'\[\d+\]', '', text).strip()
    
    # Special handling for QAMPARI - detect by answer format (comma-separated list)
    # QAMPARI answers typically have format: "Entity1 [1], Entity2 [2], Entity3 [3]"
    # More restrictive check: must have short entities and consistent citation pattern
    citations_pattern = re.findall(r'\[[^\]]+\]', answer)
    avg_words_between_citations = 0
    if len(citations_pattern) > 2:
        # Calculate average words between citations
        text_parts = re.split(r'\[[^\]]+\]', answer)
        word_counts = [len(part.strip().split()) for part in text_parts if part.strip()]
        avg_words_between_citations = sum(word_counts) / len(word_counts) if word_counts else 0
    
    # QAMPARI: multiple citations + short entities (avg < 8 words between citations)
    if (len(citations_pattern) > 2 and avg_words_between_citations < 8 and 
        ',' in answer):
        # For QAMPARI, split by comma and treat each entity as a separate unit
        sents = [s.strip() for s in re.split(r',', answer) if s.strip()]
        # Remove trailing periods from entities
        sents = [s.rstrip('.').strip() for s in sents if s.strip()]
        logger.info(f"  Detected QAMPARI format: splitting into {len(sents)} entities instead of sentences")
    else:
        # For other datasets, split into sentences
        sents = sentence_split(answer)
    target_sents = [remove_citations(sent) for sent in sents]
    
    if not sents:
        return {"citation_recall": 0.0, "citation_precision": 0.0}
    
    logger.info(f"  Processing {len(sents)} sentences with {entailment_delay}s delay between entailment tests")
    logger.info(f"  Using ALCE-style citation quality evaluation logic")
    
    # Following ALCE logic exactly
    entail = 0  # For recall calculation
    entail_prec = 0  # For precision calculation - count of necessary citations
    total_citations = 0  # Total number of citations across all sentences
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    
    for sent_id, sent in enumerate(sents):
        logger.info(f"    Sentence {sent_id+1}/{len(sents)}: {sent[:50]}...")
        
        target_sent = target_sents[sent_id]  # Citation removed
        joint_entail = -1  # Undecided
        
        # Extract citation numbers from text (e.g., [1][5] -> [1, 5])
        citation_numbers = [int(r[1:-1]) for r in re.findall(r"\[\d+\]", sent)]
        logger.info(f"      Found citation numbers: {citation_numbers}")
        
        # Initialize mapped_indices for this sentence
        mapped_indices = []
        
        if len(citation_numbers) == 0:
            # No citations - this sentence gets 0 for recall
            joint_entail = 0
            logger.info(f"      No citations found - sentence gets 0 for recall")
        else:
            # Map citation numbers directly to document indices (1-based to 0-based)
            available_count = len(cited_docs)
            
            for citation_num in citation_numbers:
                # Convert 1-based citation number to 0-based index
                doc_index = citation_num - 1
                if 0 <= doc_index < available_count:
                    mapped_indices.append(doc_index)
                    logger.info(f"      Mapped citation number {citation_num} to index {doc_index} (direct mapping)")
                else:
                    logger.warning(f"      Warning: Citation number {citation_num} (index {doc_index}) out of range (0-{available_count-1})")
            
            logger.info(f"      Final mapped indices: {mapped_indices}")
            
            if len(mapped_indices) == 0:
                # No valid citations found - this sentence gets 0 for recall
                joint_entail = 0
                logger.info(f"      No valid citations after mapping - sentence gets 0 for recall")
            else:
                total_citations += len(mapped_indices)  # Add to total citation count
                # Format documents following ALCE logic
                joint_passage = '\n'.join([_format_document(cited_docs[psgs_id]) for psgs_id in mapped_indices])
                
                # Calculate the recall score
                logger.info(f"      Testing recall with premise length: {len(joint_passage)} chars")
                joint_entail = entails_autoais(autoais, joint_passage, target_sent)
                
                # Add delay after recall test
                if entailment_delay > 0:
                    time.sleep(entailment_delay)
        
        entail += joint_entail  # Add to recall numerator
        if len(mapped_indices) > 1:
            sent_mcite += 1
        
        # FIXED: Calculate precision following ALCE logic exactly
        if joint_entail and len(mapped_indices) > 1:
            sent_mcite_support += 1
            # Precision check: count how many citations are necessary
            for psgs_id in mapped_indices:
                # condition A: Does this document alone support the claim?
                passage = _format_document(cited_docs[psgs_id])
                logger.info(f"      Testing precision for doc {psgs_id+1}/{len(mapped_indices)} (length: {len(passage)} chars)")
                nli_result = entails_autoais(autoais, passage, target_sent)
                
                # Add delay between individual doc tests
                if entailment_delay > 0:
                    time.sleep(entailment_delay)
                
                # condition B: If doc doesn't support alone, is it still necessary?
                if not nli_result:
                    subset_exclude = copy.deepcopy(mapped_indices)
                    subset_exclude.remove(psgs_id)
                    passage = '\n'.join([_format_document(cited_docs[pid]) for pid in subset_exclude])
                    logger.info(f"      Testing removal of doc {psgs_id+1} with others (total length: {len(passage)} chars)")
                    nli_result = entails_autoais(autoais, passage, target_sent)
                    
                    # Add delay after removal test
                    if entailment_delay > 0:
                        time.sleep(entailment_delay)
                
                    if nli_result:  # psgs_id is not necessary (removal test passed)
                        sent_mcite_overcite += 1
                        logger.info(f"      ❌ Document {psgs_id+1} is unnecessary (removal test passed)")
                        # Don't increment entail_prec for unnecessary citations
                    else:  # psgs_id is necessary (removal test failed)
                        logger.info(f"      ✅ Document {psgs_id+1} is necessary (removal test failed)")
                        entail_prec += 1  # Count as necessary citation
                else:  # condition A passed - document supports alone
                    logger.info(f"      ✅ Document {psgs_id+1} supports the claim")
                    entail_prec += 1  # Count as necessary citation
        else:
            # For single citations or unsupported sentences, precision = recall for this sentence
            # If joint_entail = 1 and single citation, then that citation is necessary
            # If joint_entail = 0, then no citations are counted as necessary
            entail_prec += joint_entail  # This matches ALCE logic exactly
    
    # FIXED: Calculate final metrics following ALCE exactly
    citation_recall = entail / len(sents) if sents else 0.0
    citation_precision = entail_prec / total_citations if total_citations > 0 else 0.0
    
    logger.info(f"  Final metrics: Recall={citation_recall:.3f}, Precision={citation_precision:.3f}")
    logger.info(f"  Citation counts: necessary={entail_prec}, total={total_citations}")
    
    return {
        "citation_recall": citation_recall,
        "citation_precision": citation_precision
    }


class AutoAISCitationEvaluator:
    """AutoAIS-based evaluator for citation quality in attributed question answering"""
    
    def __init__(self, space_slug: str = None, hf_token: str = None, 
                 batch_size: int = 5, delay_seconds: float = 2.0, entailment_delay: float = 1.0, capture_logs: bool = True,
                 enable_citations: bool = True, enable_qa: bool = True, enable_text_similarity: bool = False):
        # Initialize AutoAIS only if citations are enabled
        if enable_citations and space_slug:
            self.autoais = SpaceAutoAIS(space_slug=space_slug, hf_token=hf_token)
        else:
            self.autoais = None
            if enable_citations:
                print("⚠️  Warning: Citations enabled but no space_slug provided. Citation metrics will be skipped.")
        
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds
        self.entailment_delay = entailment_delay
        self.capture_logs = capture_logs
        
        # Initialize sentence transformer for semantic similarity (only if needed)
        if enable_text_similarity:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.sentence_model = None
        
        # QA model initialization removed (no longer using RoBERTa-SQuAD)
        
        # Metrics to compute
        self.metrics = [
            'citation_precision_autoais',
            'citation_recall_autoais',
            'citation_f1_autoais',
            'semantic_similarity',
            'answer_completeness'
        ]
        # category flags
        self.enable_citations = enable_citations
        self.enable_qa = enable_qa
        self.enable_text_similarity = enable_text_similarity
    
    def evaluate_single(self, result: Dict[str, Any], docs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate a single result using AutoAIS"""
        metrics = {}
        
        # Extract information
        generated_answer = result.get('generated_answer', '')
        citations = result.get('citations', [])
        reference_answer = result.get('reference_answer', '')
        
        # Basic checks
        if not generated_answer:
            return {metric: 0.0 for metric in self.metrics}
        
        # Fix empty documents
        docs = self.fix_empty_documents(docs)
        
        # 1. AutoAIS Citation Quality
        if self.enable_citations and self.autoais is not None:
            try:
                # QAMPARI detection is now handled automatically by answer format
                citation_metrics = citation_quality_autoais(generated_answer, docs, self.autoais, self.entailment_delay)
                metrics['citation_precision_autoais'] = citation_metrics['citation_precision']
                metrics['citation_recall_autoais'] = citation_metrics['citation_recall']
                # F1
                if metrics['citation_precision_autoais'] + metrics['citation_recall_autoais'] > 0:
                    metrics['citation_f1_autoais'] = 2 * (metrics['citation_precision_autoais'] * metrics['citation_recall_autoais']) / \
                                                   (metrics['citation_precision_autoais'] + metrics['citation_recall_autoais'])
                else:
                    metrics['citation_f1_autoais'] = 0.0
            except Exception as e:
                logger.error(f"Error in AutoAIS evaluation: {e}")
                metrics['citation_precision_autoais'] = 0.0
                metrics['citation_recall_autoais'] = 0.0
                metrics['citation_f1_autoais'] = 0.0
        elif self.enable_citations:
            # Citations requested but AutoAIS not available
            metrics['citation_precision_autoais'] = -1.0
            metrics['citation_recall_autoais'] = -1.0  
            metrics['citation_f1_autoais'] = -1.0
        
        # 2. Semantic Similarity with reference
        if reference_answer and self.sentence_model is not None:
            metrics['semantic_similarity'] = self.compute_semantic_similarity(generated_answer, reference_answer)
        else:
            metrics['semantic_similarity'] = -1.0  # No reference available or no model
        
        # 3. Answer Completeness
        metrics['answer_completeness'] = self.compute_answer_completeness(generated_answer, reference_answer)
        
        # 4. QA metrics (dataset-appropriate) if enabled
        # QA metrics are now computed in evaluate_dataset() for dataset-specific handling
        # No generic QA metrics computation in evaluate_single()
        
        return metrics
    
    def fix_empty_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix documents with empty or missing text"""
        fixed_docs = []
        for i, doc in enumerate(docs):
            text = doc.get('text') or doc.get('summary') or doc.get('content', '')
            
            if not text or len(text.strip()) < 10:
                title = doc.get('title', f'Document {i+1}')
                text = f"This document discusses {title.lower()}."
            
            fixed_doc = doc.copy()
            fixed_doc['text'] = text
            fixed_doc['id'] = doc.get('id', str(i))
            fixed_docs.append(fixed_doc)
        
        return fixed_docs
    
    def compute_semantic_similarity(self, generated: str, reference: str) -> float:
        """Compute semantic similarity between generated and reference answers"""
        if not generated or not reference:
            return 0.0
        
        try:
            # Encode both texts
            gen_embedding = self.sentence_model.encode(generated, convert_to_tensor=True)
            ref_embedding = self.sentence_model.encode(reference, convert_to_tensor=True)
            
            # Compute cosine similarity
            similarity = util.pytorch_cos_sim(gen_embedding, ref_embedding).item()
            
            return similarity
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def compute_answer_completeness(self, generated: str, reference: str) -> float:
        """Improved answer completeness evaluation"""
        if not reference:
            # If no reference, check if answer seems complete
            sentences = sent_tokenize(generated)
            word_count = len(generated.split())
            
            if len(sentences) >= 3 and word_count >= 30:
                return 0.8
            elif len(sentences) >= 2 and word_count >= 20:
                return 0.6
            elif word_count >= 10:
                return 0.4
            else:
                return 0.2
        
        # Extract key information from both texts
        ref_sentences = [s.strip() for s in sent_tokenize(reference) if s.strip()]
        gen_sentences = [s.strip() for s in sent_tokenize(generated) if s.strip()]
        
        if not ref_sentences:
            return 0.0
        
        # Check coverage with more lenient matching
        covered = 0
        for ref_sent in ref_sentences:
            if len(ref_sent) < 10:  # Skip very short sentences
                covered += 1
                continue
                
            # Check if any generated sentence covers this information
            max_similarity = 0
            for gen_sent in gen_sentences:
                if len(gen_sent) < 10:
                    continue
                
                # Simple word overlap check
                ref_words = set(word.lower() for word in ref_sent.split() if len(word) > 2)
                gen_words = set(word.lower() for word in gen_sent.split() if len(word) > 2)
                
                if ref_words and gen_words:
                    overlap = len(ref_words.intersection(gen_words))
                    similarity = overlap / len(ref_words)
                    max_similarity = max(max_similarity, similarity)
            
            # More lenient threshold
            if max_similarity > 0.3:
                covered += 1
        
        return covered / len(ref_sentences)

    # ----- v2 helpers -----
    @staticmethod
    def _remove_citations(text: str) -> str:
        return re.sub(r"\[\d+\]", "", text or "").strip()

    @staticmethod
    def _normalize_answer_simple(text: str) -> str:
        import string
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = ''.join(ch for ch in text if ch not in set(string.punctuation))
        return ' '.join(text.split())

    def _derive_hagrid_gold_answer(self, item: Dict[str, Any]) -> str:
        """Use AttributedIR's quality hierarchy for HAGRID evaluation"""
        answers = item.get('answers', []) or []
        
        # Apply AttributedIR's quality hierarchy
        att_and_info = []  # Priority 1: attributable=1 AND informative=1
        att_only = []      # Priority 2: attributable=1 AND informative=0
        info_only = []     # Priority 3: attributable=0 AND informative=1
        
        for answer in answers:
            att = answer.get('attributable', 0)
            info = answer.get('informative', 0)
            
            if att == 1 and info == 1:
                att_and_info.append(answer)
            elif att == 1 and info == 0:
                att_only.append(answer)
            elif att == 0 and info == 1:
                info_only.append(answer)
        
        # Selection hierarchy (same as get_attributable_answer)
        if att_and_info:
            selected = att_and_info[0]
        elif att_only:
            selected = att_only[0]
        elif info_only:
            selected = info_only[0]
        else:
            selected = answers[0] if answers else None
        
        if selected:
            return self._remove_citations(selected.get('answer', ''))
        return ''
    
    def _get_all_hagrid_answers(self, item: Dict[str, Any]) -> List[str]:
        """Get all HAGRID answers for comprehensive evaluation (AttributedIR style)"""
        answers = item.get('answers', []) or []
        return [self._remove_citations(ans.get('answer', '')) for ans in answers if ans.get('answer', '')]

    def _compute_rouge_lsum_single(self, pred: str, ref: str) -> float:
        if rouge_scorer is None:
            return 0.0
        try:
            def preprocess(t: str) -> str:
                t = (t or '').lower().strip()
                return '\n'.join(sent_tokenize(t))
            scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
            scores = scorer.score(preprocess(ref), preprocess(pred))
            return float(scores['rougeLsum'].fmeasure * 100.0)
        except Exception as e:
            logger.warning(f"ROUGE-Lsum failed: {e}")
            return 0.0

    def _compute_bleu_single(self, pred: str, ref: str) -> float:
        if sentence_bleu is None:
            return 0.0
        try:
            smoothie = SmoothingFunction().method3 if SmoothingFunction else None
            ref_tok = self._normalize_answer_simple(ref).split()
            hyp_tok = self._normalize_answer_simple(pred).split()
            if not ref_tok or not hyp_tok:
                return 0.0
            bleu = sentence_bleu([ref_tok], hyp_tok, smoothing_function=smoothie)
            return float(bleu * 100.0)
        except Exception as e:
            logger.warning(f"BLEU failed: {e}")
            return 0.0

    def _compute_bert_f1_single(self, pred: str, ref: str) -> float:
        try:
            if bert_score_compute is not None:
                P, R, F = bert_score_compute([pred], [ref], lang='en', rescale_with_baseline=True)
                return float(F.mean().item() * 100.0)
        except Exception as e:
            logger.warning(f"BERTScore failed, falling back to STS: {e}")
        try:
            gen_embedding = self.sentence_model.encode(pred, convert_to_tensor=True)
            ref_embedding = self.sentence_model.encode(ref, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(gen_embedding, ref_embedding).item()
            return float(max(0.0, min(1.0, sim)) * 100.0)
        except Exception as e:
            logger.warning(f"STS fallback failed: {e}")
            return 0.0

    def _compute_asqa_str_metrics(self, generated_answer: str, qa_pairs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute STR-EM/STR-Hit using simple substring containment (legacy method)"""
        if not qa_pairs:
            return {"STR-EM": 0.0, "STR-Hit": 0.0}
        n_context = self._normalize_answer_simple(self._remove_citations(generated_answer or ''))
        per_pair = []
        for qa in qa_pairs:
            short_answers = [ self._normalize_answer_simple(sa) for sa in qa.get('short_answers', []) ]
            found = any(sa in n_context for sa in short_answers)
            per_pair.append(1.0 if found else 0.0)
        acc = float(np.mean(per_pair)) if per_pair else 0.0
        hit = 1.0 if acc == 1.0 and per_pair else 0.0
        return {"STR-EM": 100.0 * acc, "STR-Hit": 100.0 * hit}

    def _compute_short_answer_str_metrics(self, generated_answer: str, reference_answers: List[str]) -> Dict[str, float]:
        """
        Compute STR-EM/STR-F1/STR-Hit for short-answer datasets (NQ/MS MARCO) using substring matching.
        Similar to ASQA's STR metrics but adapted for single-answer format.
        """
        if not reference_answers or not (generated_answer or '').strip():
            return {
                'STR-EM': 0.0, 'STR-F1': 0.0, 'STR-Hit': 0.0
            }

        from collections import Counter

        # Normalize generated answer (remove citations and normalize)
        n_context = self._normalize_answer_simple(self._remove_citations(generated_answer or ''))
        
        # Check each reference answer for substring match
        best_str_em = 0.0
        best_str_f1 = 0.0
        
        for ref in reference_answers:
            # Normalize reference answer
            n_ref = self._normalize_answer_simple(ref)
            
            # STR-EM: Check if reference appears as substring in generated answer
            str_em = 1.0 if n_ref in n_context else 0.0
            
            # STR-F1: Compute token-level overlap for substring matching
            # This gives partial credit for cases where some reference tokens appear
            ref_tokens = set(n_ref.split())
            context_tokens = set(n_context.split())
            
            if len(ref_tokens) == 0 and len(context_tokens) == 0:
                str_f1 = 1.0
            elif len(ref_tokens) == 0:
                str_f1 = 0.0
            else:
                # Compute how many reference tokens appear in the context
                common_tokens = ref_tokens & context_tokens
                recall = len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
                precision = len(common_tokens) / len(context_tokens) if len(context_tokens) > 0 else 0.0
                
                if precision + recall == 0:
                    str_f1 = 0.0
                else:
                    str_f1 = 2 * precision * recall / (precision + recall)
            
            # Take max over references (SQuAD convention)
            best_str_em = max(best_str_em, str_em)
            best_str_f1 = max(best_str_f1, str_f1)
        
        # STR-Hit: Perfect substring match achieved
        str_hit = 1.0 if best_str_em == 1.0 else 0.0
        
        return {
            'STR-EM': best_str_em * 100.0,
            'STR-F1': best_str_f1 * 100.0,
            'STR-Hit': str_hit * 100.0
        }

    # Removed experimental RoBERTa-SQuAD QA metrics for ASQA
    # ASQA now uses only STR metrics (substring matching) which are more reliable per ALCE research

    def _compute_qampari_item(self, generated_answer: str, answers: List[List[str]], cot: bool = False) -> Dict[str, float]:
        o = generated_answer or ''
        if cot and ':' in o:
            o = ':'.join(o.split(':')[1:])
        # Remove citations before parsing and normalizing
        o_no_citations = self._remove_citations(o)
        preds = [ self._normalize_answer_simple(x.strip()) for x in o_no_citations.rstrip().rstrip('.').rstrip(',').split(',') ]
        preds = [ p for p in preds if len(p) > 0 ]
        num_preds = len(preds)
        answers_norm = [[ self._normalize_answer_simple(x) for x in ans ] for ans in (answers or [])]
        flat_answers = [ item for sub in answers_norm for item in sub ]
        prec = (sum(1 for p in preds if p in flat_answers) / num_preds) if num_preds > 0 else 0.0
        rec_hits = sum(1 for a in answers_norm if any(x in preds for x in a))
        total = len(answers_norm) if answers_norm else 1
        rec = rec_hits / total
        rec_top5 = min(5, rec_hits) / min(5, total)
        f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
        f1_top5 = 0.0 if (prec + rec_top5) == 0 else (2 * prec * rec_top5) / (prec + rec_top5)
        return {
            "num_preds": float(num_preds),
            "qampari_prec": 100.0 * prec,
            "qampari_rec": 100.0 * rec,
            "qampari_rec_top5": 100.0 * rec_top5,
            "qampari_f1": 100.0 * f1,
            "qampari_f1_top5": 100.0 * f1_top5,
        }

    def _compute_eli5_claims_nli(self, generated_answer: str, claims: List[str]) -> float:
        if not claims:
            return 0.0
        if self.autoais is None:
            logger.warning("AutoAIS not available for claims NLI evaluation")
            return -1.0  # Indicate evaluation not possible
        normalized_output = self._remove_citations(generated_answer or '')
        entail = 0
        for claim in claims:
            try:
                entail += int(self.autoais(normalized_output, claim) == 1)
            except Exception as e:
                logger.warning(f"AutoAIS claim check failed: {e}")
        return 100.0 * (entail / len(claims))

    def _compute_qa_short_answer_metrics(self, generated_answer: str, reference_answers: List[str]) -> Dict[str, float]:
        """
        Direct string-based QA EM/F1/Hit for short-answer datasets (NQ/MS MARCO).
        Now follows the SQuAD convention: take max-over-references per example
        for EM/F1 (rather than mean), and expose token precision/recall for the
        best-F1 reference. Also adds Hit-EM while keeping QA-Hit (F1>0.5) for backward compatibility.
        """
        if not reference_answers or not (generated_answer or '').strip():
            return {
                'QA-EM': 0.0, 'QA-F1': 0.0, 'QA-Hit': 0.0,
                'QA-Hit-EM': 0.0, 'QA-Precision': 0.0, 'QA-Recall': 0.0
            }

        from collections import Counter

        def _pr_tokens(a_gold: str, a_pred: str) -> Tuple[float, float]:
            """Token precision/recall with the same normalization as EM/F1."""
            gold_toks = self._normalize_answer(a_gold).split()
            pred_toks = self._normalize_answer(a_pred).split()
            if len(pred_toks) == 0 and len(gold_toks) == 0:
                return 1.0, 1.0
            if len(pred_toks) == 0:
                return 0.0, 0.0
            common = Counter(gold_toks) & Counter(pred_toks)
            num_same = sum(common.values())
            precision = num_same / len(pred_toks) if len(pred_toks) > 0 else 0.0
            recall = num_same / len(gold_toks) if len(gold_toks) > 0 else 0.0
            return precision, recall

        clean_generated = self._remove_citations(generated_answer)

        best_em = 0.0
        best_f1 = 0.0
        best_p = 0.0
        best_r = 0.0

        for ref in reference_answers:
            em = float(self._compute_exact(ref, clean_generated))
            f1 = float(self._compute_f1(ref, clean_generated))
            if f1 > best_f1:
                p, r = _pr_tokens(ref, clean_generated)
                best_f1 = f1
                best_p = p
                best_r = r
            if em > best_em:
                best_em = em

        hit_em = 1.0 if best_em == 1.0 else 0.0
        hit_f1 = 1.0 if best_f1 > 0.5 else 0.0  # preserves your original "QA-Hit"

        return {
            'QA-EM': best_em * 100.0,
            'QA-F1': best_f1 * 100.0,
            'QA-Precision': best_p * 100.0,
            'QA-Recall': best_r * 100.0,
            'QA-Hit-EM': hit_em * 100.0,
            'QA-Hit': hit_f1 * 100.0  # legacy name retained (F1>0.5)
        }

    # --- QA metrics helper functions (EM/F1/Hit) ---
    @staticmethod
    def _normalize_answer(s: str) -> str:
        import re, string
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def _compute_f1(cls, a_gold: str, a_pred: str) -> float:
        from collections import Counter
        def _get_tokens(text):
            if not text:
                return []
            return cls._normalize_answer(text).split()
        gold_toks = _get_tokens(a_gold)
        pred_toks = _get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return float(gold_toks == pred_toks)
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def _compute_exact(cls, a_gold: str, a_pred: str) -> int:
        return int(cls._normalize_answer(a_gold) == cls._normalize_answer(a_pred))

    # Removed legacy compute_qa_metrics method (replaced by dataset-specific methods)
    
    def extract_documents_from_item(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract documents from dataset item based on dataset type"""
        docs = []
        
        # Try different document field names based on dataset structure
        if 'docs' in item:
            docs = item['docs']
        elif 'quotes' in item:  # HAGRID format
            for quote in item['quotes']:
                docs.append({
                    'id': quote.get('docid', ''),
                    'text': quote.get('text', ''),
                    'title': f"Document {quote.get('idx', '')}"
                })
        elif 'passages' in item:  # MS MARCO format
            passages = item['passages']
            for i, (text, url) in enumerate(zip(
                passages.get('passage_text', []), 
                passages.get('url', [])
            )):
                docs.append({
                    'id': str(i),
                    'text': text,
                    'title': '',
                    'url': url
                })
        elif 'search_results' in item:  # TriviaQA Web format
            search_results = item['search_results']
            for i, (desc, title, url) in enumerate(zip(
                search_results.get('description', []),
                search_results.get('title', []),
                search_results.get('url', [])
            )):
                search_contexts = search_results.get('search_context', [])
                context = search_contexts[i] if i < len(search_contexts) else desc
                
                docs.append({
                    'id': str(i),
                    'text': context,
                    'title': title,
                    'url': url,
                    'summary': desc
                })
        elif 'entity_pages' in item:  # TriviaQA Wiki format
            entity_pages = item['entity_pages']
            wiki_contexts = entity_pages.get('wiki_context', [])
            titles = entity_pages.get('title', [])
            filenames = entity_pages.get('filename', [])
            
            for i, (context, title, filename) in enumerate(zip(
                wiki_contexts,
                titles,
                filenames
            )):
                docs.append({
                    'id': str(i),
                    'text': context,
                    'title': title,
                    'filename': filename,
                    'source': 'wikipedia'
                })
        elif 'context' in item:  # 2WikiMultihop format
            for i, (entity, facts) in enumerate(item['context']):
                text = ' '.join(facts) if isinstance(facts, list) else str(facts)
                docs.append({
                    'id': str(i),
                    'text': text,
                    'title': entity,
                    'entity': entity
                })
        elif 'relevant_docs' in item:  # Natural Questions format
            relevant_docs = item['relevant_docs']
            for doc in relevant_docs:
                docs.append({
                    'id': doc.get('doc_id', ''),
                    'text': doc.get('text', ''),
                    'title': doc.get('title', ''),
                    'score': doc.get('score', 0)
                })
        
        return docs
    
    def evaluate_dataset(self, results_file: str, dataset_file: str) -> Dict[str, Any]:
        """Evaluate all results from a dataset using AutoAIS"""
        logger.info(f"Starting AutoAIS evaluation for {results_file}")
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Load original dataset to get documents
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Create a mapping of question to documents and QA pairs (if available)
        question_to_docs = {}
        question_to_qa = {}
        question_to_refs = {}
        for item in dataset:
            question = item.get('question') or item.get('query', '')
            if question:
                docs = self.extract_documents_from_item(item)
                question_to_docs[question] = self.fix_empty_documents(docs)
                if 'qa_pairs' in item and item['qa_pairs']:
                    question_to_qa[question] = item['qa_pairs']
                # Collect short reference answers for NQ/MS MARCO
                if 'answers' in item and isinstance(item['answers'], list):
                    # Flatten possible nested lists (NQ style is list of strings; ms marco list of strings)
                    refs = []
                    for ans in item['answers']:
                        if isinstance(ans, list):
                            refs.extend(ans)
                        else:
                            refs.append(ans)
                    question_to_refs[question] = [str(x) for x in refs if isinstance(x, (str, int, float))]
        
        # Evaluate each result
        all_metrics = defaultdict(list)
        individual_results = []  # Store detailed logs for each result
        
        # Detect dataset type flags
        path_lc = dataset_file.lower()
        is_asqa = 'asqa' in path_lc
        is_qampari = 'qampari' in path_lc
        is_eli5 = 'eli5' in path_lc
        is_hagrid = 'hagrid' in path_lc

        for i, result in enumerate(results):
            logger.info(f"Processing result {i+1}/{len(results)}")
            
            question = result.get('question', '')
            docs = question_to_docs.get(question, [])
            qa_pairs = question_to_qa.get(question, None)
            
            # Create detailed result log
            result_log = {
                'result_id': result.get('id', f'result_{i}'),
                'question': question,
                'generated_answer': result.get('generated_answer', ''),
                'citations': result.get('citations', []),
                'num_documents': len(docs),
                'evaluation_log': []
            }
            
            if self.capture_logs:
                # Capture evaluation logs by temporarily redirecting logger
                import io
                import sys
                import logging
                from contextlib import redirect_stdout, redirect_stderr
                
                # Create string buffer to capture logs
                log_buffer = io.StringIO()
                
                # Store original logging levels and handlers
                original_handlers = logger.handlers[:]
                original_level = logger.level
                
                # Temporarily redirect logger output
                logger.handlers.clear()
                logger.setLevel(logging.INFO)
                
                # Add a handler that writes to our buffer
                buffer_handler = logging.StreamHandler(log_buffer)
                buffer_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(message)s')
                buffer_handler.setFormatter(formatter)
                logger.addHandler(buffer_handler)
                
                # Also suppress other noisy loggers
                httpx_logger = logging.getLogger('httpx')
                sentence_transformers_logger = logging.getLogger('sentence_transformers')
                original_httpx_level = httpx_logger.level
                original_st_level = sentence_transformers_logger.level
                
                httpx_logger.setLevel(logging.WARNING)
                sentence_transformers_logger.setLevel(logging.WARNING)
                
                try:
                    # Attach qa_pairs to result temporarily for evaluate_single
                    if qa_pairs:
                        result['_qa_pairs'] = qa_pairs
                    metrics = self.evaluate_single(result, docs)
                    # QA/correctness category augmentations
                    if self.enable_qa:
                        if is_asqa:
                            qa_pairs = question_to_qa.get(question, [])
                            # Compute string-based metrics only (experimental QA metrics removed)
                            str_m = self._compute_asqa_str_metrics(result.get('generated_answer',''), qa_pairs)
                            metrics.update(str_m)
                        if is_qampari:
                            answers = question_to_docs.get('__placeholder_answers__', None)
                            # Retrieve answers from dataset item
                            item = next((it for it in dataset if (it.get('question') or it.get('query','')) == question), None)
                            if item is not None:
                                qamp = self._compute_qampari_item(result.get('generated_answer',''), item.get('answers', []))
                                metrics.update(qamp)
                        if is_eli5:
                            item = next((it for it in dataset if (it.get('question') or it.get('query','')) == question), None)
                            claims = item.get('claims', []) if item else []
                            metrics['claims_nli'] = self._compute_eli5_claims_nli(result.get('generated_answer',''), claims)
                        # Add short-answer QA EM/F1/Hit for NQ/MS MARCO
                        if (not is_qampari) and (not is_eli5) and (not is_asqa) and question in question_to_refs:
                            qa_short = self._compute_qa_short_answer_metrics(result.get('generated_answer',''), question_to_refs[question])
                            str_short = self._compute_short_answer_str_metrics(result.get('generated_answer',''), question_to_refs[question])
                            metrics.update(qa_short)
                            metrics.update(str_short)
                        if self.enable_text_similarity and (is_hagrid or is_eli5 or is_asqa):
                            # Derive gold text
                            item = next((it for it in dataset if (it.get('question') or it.get('query','')) == question), None)
                            if item is not None:
                                if is_hagrid:
                                    gold = self._derive_hagrid_gold_answer(item)
                                elif is_eli5:
                                    gold = self._remove_citations((item.get('answer') or '').strip())
                                elif is_asqa:
                                    anns = item.get('annotations') or []
                                    if isinstance(anns, list) and len(anns) > 0:
                                        gold = self._remove_citations((anns[0].get('long_answer') or '').strip())
                                    else:
                                        gold = self._remove_citations((item.get('answer') or '').strip())
                                else:
                                    gold = ''
                                if gold:
                                    clean_gen = self._remove_citations(result.get('generated_answer',''))
                                    metrics['rougeLsum'] = self._compute_rouge_lsum_single(clean_gen, gold)
                                    metrics['bleu'] = self._compute_bleu_single(clean_gen, gold)
                                    metrics['bert_f1'] = self._compute_bert_f1_single(clean_gen, gold)
                    
                    # Get the captured logs
                    captured_logs = log_buffer.getvalue()
                    result_log['evaluation_log'] = captured_logs.split('\n') if captured_logs else []
                    
                    # Add metrics to result log
                    result_log['metrics'] = metrics
                    
                    for metric_name, value in metrics.items():
                        if value >= 0:  # Skip -1 values (no reference)
                            all_metrics[metric_name].append(value)
                    
                finally:
                    # Restore original handlers and levels
                    logger.handlers.clear()
                    logger.handlers.extend(original_handlers)
                    logger.setLevel(original_level)
                    httpx_logger.setLevel(original_httpx_level)
                    sentence_transformers_logger.setLevel(original_st_level)
                    if '_qa_pairs' in result:
                        del result['_qa_pairs']
            else:
                # Just evaluate without capturing logs
                if qa_pairs:
                    result['_qa_pairs'] = qa_pairs
                metrics = self.evaluate_single(result, docs)
                if self.enable_qa:
                    if is_asqa:
                        qa_pairs = question_to_qa.get(question, [])
                        # Compute string-based metrics only (experimental QA metrics removed)
                        str_m = self._compute_asqa_str_metrics(result.get('generated_answer',''), qa_pairs)
                        metrics.update(str_m)
                    if is_qampari:
                        item = next((it for it in dataset if (it.get('question') or it.get('query','')) == question), None)
                        if item is not None:
                            qamp = self._compute_qampari_item(result.get('generated_answer',''), item.get('answers', []))
                            metrics.update(qamp)
                    if is_eli5:
                        item = next((it for it in dataset if (it.get('question') or it.get('query','')) == question), None)
                        claims = item.get('claims', []) if item else []
                        metrics['claims_nli'] = self._compute_eli5_claims_nli(result.get('generated_answer',''), claims)
                    if (not is_qampari) and (not is_eli5) and (not is_asqa) and question in question_to_refs:
                        qa_short = self._compute_qa_short_answer_metrics(result.get('generated_answer',''), question_to_refs[question])
                        str_short = self._compute_short_answer_str_metrics(result.get('generated_answer',''), question_to_refs[question])
                        metrics.update(qa_short)
                        metrics.update(str_short)
                    if self.enable_text_similarity and (is_hagrid or is_eli5 or is_asqa):
                        item = next((it for it in dataset if (it.get('question') or it.get('query','')) == question), None)
                        if item is not None:
                            if is_hagrid:
                                gold = self._derive_hagrid_gold_answer(item)
                            elif is_eli5:
                                gold = self._remove_citations((item.get('answer') or '').strip())
                            elif is_asqa:
                                anns = item.get('annotations') or []
                                if isinstance(anns, list) and len(anns) > 0:
                                    gold = self._remove_citations((anns[0].get('long_answer') or '').strip())
                                else:
                                    gold = self._remove_citations((item.get('answer') or '').strip())
                            else:
                                gold = ''
                            if gold:
                                clean_gen = self._remove_citations(result.get('generated_answer',''))
                                metrics['rougeLsum'] = self._compute_rouge_lsum_single(clean_gen, gold)
                                metrics['bleu'] = self._compute_bleu_single(clean_gen, gold)
                                metrics['bert_f1'] = self._compute_bert_f1_single(clean_gen, gold)
                if '_qa_pairs' in result:
                    del result['_qa_pairs']
                result_log['metrics'] = metrics
                
                for metric_name, value in metrics.items():
                    if value >= 0:  # Skip -1 values (no reference)
                        all_metrics[metric_name].append(value)
            
            individual_results.append(result_log)
            
            # Add delay between examples to avoid rate limiting (only if using AutoAIS)
            if i < len(results) - 1 and self.autoais is not None:  # Don't delay after the last example or if no AutoAIS
                delay = self.delay_seconds + random.uniform(0, 1.0)
                logger.info(f"Waiting {delay:.1f}s before next example...")
                time.sleep(delay)
        
        # Compute aggregate statistics
        aggregate_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregate_metrics[f'{metric_name}_mean'] = np.mean(values)
                aggregate_metrics[f'{metric_name}_std'] = np.std(values)
                aggregate_metrics[f'{metric_name}_min'] = np.min(values)
                aggregate_metrics[f'{metric_name}_max'] = np.max(values)
        
        # Add metadata
        aggregate_metrics['num_evaluated'] = len(results)
        aggregate_metrics['dataset'] = results[0].get('dataset', 'unknown') if results else 'unknown'
        aggregate_metrics['approach'] = results[0].get('approach', 'unknown') if results else 'unknown'
        
        # Add individual results to the return value
        aggregate_metrics['individual_results'] = individual_results
        
        return aggregate_metrics
    
    def generate_report(self, metrics: Dict[str, Any], output_file: str):
        """Generate evaluation report"""
        # Extract individual results if present
        individual_results = metrics.pop('individual_results', [])
        
        report = {
            'evaluation_metrics': metrics,
            'summary': {
                'overall_citation_quality_autoais': (
                    metrics.get('citation_f1_autoais_mean', 0) * 0.5 +
                    metrics.get('citation_precision_autoais_mean', 0) * 0.3 +
                    metrics.get('citation_recall_autoais_mean', 0) * 0.2
                ),
                'answer_quality': (
                    metrics.get('semantic_similarity_mean', 0) * 0.5 +
                    metrics.get('answer_completeness_mean', 0) * 0.5
                ) if metrics.get('semantic_similarity_mean', -1) >= 0 else metrics.get('answer_completeness_mean', 0)
            }
        }
        
        # Add individual results with detailed logging if available
        if individual_results:
            report['individual_results'] = individual_results
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"AutoAIS evaluation report saved to: {output_file}")
        
        # Print summary
        print("\n=== AutoAIS Evaluation Summary ===")
        print(f"Dataset: {metrics.get('dataset', 'unknown')}")
        print(f"Approach: {metrics.get('approach', 'unknown')}")
        print(f"Number of examples: {metrics.get('num_evaluated', 0)}")
        print(f"\nAutoAIS Citation Quality Metrics:")
        print(f"  Citation Precision: {metrics.get('citation_precision_autoais_mean', 0):.3f} (±{metrics.get('citation_precision_autoais_std', 0):.3f})")
        print(f"  Citation Recall: {metrics.get('citation_recall_autoais_mean', 0):.3f} (±{metrics.get('citation_recall_autoais_std', 0):.3f})")
        print(f"  Citation F1: {metrics.get('citation_f1_autoais_mean', 0):.3f} (±{metrics.get('citation_f1_autoais_std', 0):.3f})")
        
        if metrics.get('semantic_similarity_mean', -1) >= 0:
            print(f"\nAnswer Quality Metrics:")
            print(f"  Semantic Similarity: {metrics.get('semantic_similarity_mean', 0):.3f} (±{metrics.get('semantic_similarity_std', 0):.3f})")
            print(f"  Answer Completeness: {metrics.get('answer_completeness_mean', 0):.3f} (±{metrics.get('answer_completeness_std', 0):.3f})")
        
        # Optional: Print QA metrics if present
        if 'QA-EM_mean' in metrics or 'QA-F1_mean' in metrics or 'QA-Hit_mean' in metrics:
            # Determine the correct method based on which metrics are present
            if 'QAMPARI-Precision_mean' in metrics or 'QAMPARI-Recall_mean' in metrics:
                print(f"\nQA Metrics (QAMPARI List Evaluation):")
            elif 'Claims-NLI_mean' in metrics:
                print(f"\nQA Metrics (Claims NLI):")
            elif 'QA-EM_mean' in metrics and 'QA-F1_mean' in metrics and 'QA-Hit_mean' in metrics:
                # All datasets use string-based QA evaluation
                print(f"\nQA Metrics (SQuAD-style String Evaluation):")
            else:
                print(f"\nQA Metrics:")
            
            if 'QA-EM_mean' in metrics:
                print(f"  QA-EM: {metrics.get('QA-EM_mean', 0):.2f}")
            if 'QA-F1_mean' in metrics:
                print(f"  QA-F1: {metrics.get('QA-F1_mean', 0):.2f}")
            if 'QA-Hit_mean' in metrics:
                print(f"  QA-Hit: {metrics.get('QA-Hit_mean', 0):.2f}")
        
        # Print STR metrics if present (for NQ/MS MARCO and ASQA)
        if 'STR-EM_mean' in metrics or 'STR-F1_mean' in metrics or 'STR-Hit_mean' in metrics:
            print(f"\nSTR Metrics (Substring Matching):")
            if 'STR-EM_mean' in metrics:
                print(f"  STR-EM: {metrics.get('STR-EM_mean', 0):.2f}")
            if 'STR-F1_mean' in metrics:
                print(f"  STR-F1: {metrics.get('STR-F1_mean', 0):.2f}")
            if 'STR-Hit_mean' in metrics:
                print(f"  STR-Hit: {metrics.get('STR-Hit_mean', 0):.2f}")
        
        print(f"\nOverall AutoAIS Citation Quality Score: {report['summary']['overall_citation_quality_autoais']:.3f}")
        print(f"Overall Answer Quality Score: {report['summary']['answer_quality']:.3f}")
        
        if individual_results:
            print(f"\nDetailed evaluation logs saved for {len(individual_results)} individual results")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate citation quality and correctness (combined)')
    parser.add_argument('--results', type=str, required=True, help='Path to results file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to original dataset file')
    parser.add_argument('--output', type=str, required=True, help='Path to output evaluation report')
    parser.add_argument('--space_slug', type=str, default='za-zeeshan-33/true-model', help='Gradio space slug')
    parser.add_argument('--hf_token', type=str, help='Hugging Face token (or set HF_TOKEN env var)')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for processing')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between examples (seconds)')
    parser.add_argument('--entailment_delay', type=float, default=1.0, help='Delay between entailment tests (seconds)')
    # category flags (ALCE-like)
    parser.add_argument('--citations', action='store_true', help='Compute citation quality metrics (AutoAIS)')
    parser.add_argument('--qa', action='store_true', help='Compute dataset-appropriate correctness metrics')
    parser.add_argument('--text-sim', action='store_true', help='Also compute text similarity metrics (ROUGE-Lsum/BLEU/BERTScore)')
    parser.add_argument('--correctness-only', action='store_true', help='Only compute correctness metrics (no AutoAIS/citation evaluation)')
    parser.add_argument('--skip-autoais', action='store_true', help='Skip AutoAIS/Space API calls (same as --correctness-only)')
    
    args = parser.parse_args()
    
    # Handle correctness-only mode
    correctness_only = args.correctness_only or args.skip_autoais
    
    if correctness_only:
        # Force QA-only mode, disable citations
        enable_citations = False
        enable_qa = True
        print("🎯 Correctness-only mode: Skipping AutoAIS/citation evaluation")
    else:
        # default: both categories if neither specified
        enable_citations = args.citations or (not args.citations and not args.qa)
        enable_qa = args.qa or (not args.citations and not args.qa)
    
    # Enable text similarity by default for datasets with rich reference answers
    enable_text_sim = args.text_sim
    dataset_path_lower = args.dataset.lower()
    if not args.text_sim and any(dataset in dataset_path_lower for dataset in ['hagrid', 'eli5', 'asqa']):
        enable_text_sim = True
        detected_dataset = next(d for d in ['hagrid', 'eli5', 'asqa'] if d in dataset_path_lower)
        print(f"🎯 Auto-enabling text similarity metrics for {detected_dataset.upper()} dataset")

    # Handle token requirement - not needed for correctness-only mode
    hf_token = args.hf_token
    if correctness_only and not hf_token:
        # In correctness-only mode, we don't need HF token since we skip AutoAIS
        print("💡 No HF token needed for correctness-only mode")
    
    evaluator = AutoAISCitationEvaluator(
        space_slug=args.space_slug if not correctness_only else None,
        hf_token=hf_token,
        batch_size=args.batch_size,
        delay_seconds=args.delay,
        entailment_delay=args.entailment_delay,
        enable_citations=enable_citations,
        enable_qa=enable_qa,
        enable_text_similarity=enable_text_sim
    )
    
    metrics = evaluator.evaluate_dataset(args.results, args.dataset)
    evaluator.generate_report(metrics, args.output)


if __name__ == "__main__":
    main()
