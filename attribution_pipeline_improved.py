#!/usr/bin/env python3
"""
Improved Attribution Pipeline for Question Answering
Fixes identified issues with document retrieval and citation evaluation
"""

import json
import os
import sys
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import re
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from simplified_prompt_manager import SimplifiedPromptManager, MethodType
from searcher import SearcherWithinDocs

load_dotenv()


class ImprovedAttributionPipeline:
    """Improved pipeline for attributed question answering with fixes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config['model_name']
        self.approach = config['approach']
        self.top_k = config['top_k']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self.num_few_shot = config['num_few_shot']
        self.seed = config.get('seed')
        
        self.top_p = config.get('top_p', 1.0 if 'gpt' in self.model_name.lower() or 'o1' in self.model_name.lower() else 0.95)
        self.citation_temperature = 0.1
        if self.model_name in ["deepseekv3", "llama-3.1-8b", "llama-3.1-8b-instruct", "gemma-2-2b-it", "gemma-2-9b-it", "gemma-3-27b-it", "llama-3-70b-instruct", "mistral-7b-instruct-v0.2", "qwen2.5-72b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"]:
            from huggingface_hub import InferenceClient
        
        if self.model_name == "deepseekv3":
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                raise ValueError("HF_TOKEN not found in environment variable.")
            self.client = InferenceClient(provider="fireworks-ai", api_key=hf_token)

        elif self.model_name == "llama-3.1-8b":
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                raise ValueError("HF_TOKEN not found in environment variable.")
            self.client = InferenceClient(provider="featherless-ai", api_key=hf_token)

        elif self.model_name == "llama-3.1-8b-instruct":
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                raise ValueError("HF_TOKEN not found in environment variable.")
            self.client = InferenceClient(provider="nebius", api_key=hf_token)

        elif self.model_name == "gemma-2-2b":
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                raise ValueError("HF_TOKEN not found in environment variable.")
            self.client = InferenceClient(provider="nebius", api_key=hf_token)

        elif self.model_name == "llama-3.1-8b-instruct":
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                raise ValueError("HF_TOKEN not found in environment variable.")
            self.client = InferenceClient(provider="nebius", api_key=hf_token)
        
        elif self.model_name == "gemma-2-2b-it":
            hf_token = os.getenv('HF_TOKEN')
            self.client = InferenceClient(provider="nebius", api_key=hf_token)
        
        elif self.model_name == "gemma-2-9b-it":
            hf_token = os.getenv('HF_TOKEN')
            self.client = InferenceClient(provider="nebius", api_key=hf_token)
        
        elif self.model_name == "gemma-3-27b-it":
            hf_token = os.getenv('HF_TOKEN')
            self.client = InferenceClient(provider="nebius", api_key=hf_token)
        
        elif self.model_name == "llama-3-70b-instruct":
            hf_token = os.getenv('HF_TOKEN')
            self.client = InferenceClient(provider="hyperbolic", api_key=hf_token)
        
        elif self.model_name == "mistral-7b-instruct-v0.2":
            hf_token = os.getenv('HF_TOKEN')
            self.client = InferenceClient(provider="featherless-ai", api_key=hf_token)
        
        elif self.model_name == "qwen2.5-72b-instruct":
            hf_token = os.getenv('HF_TOKEN')
            self.client = InferenceClient(provider="nebius", api_key=hf_token)
        
        elif self.model_name == "qwen2.5-14b-instruct":
            hf_token = os.getenv('HF_TOKEN')
            self.client = InferenceClient(provider="featherless-ai", api_key=hf_token)
        
        elif self.model_name == "qwen2.5-7b-instruct":
            hf_token = os.getenv('HF_TOKEN')
            self.client = InferenceClient(provider="featherless-ai", api_key=hf_token)
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            self.client = OpenAI(api_key=api_key)
        
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.prompt_manager = SimplifiedPromptManager()
        
        self.dataset_loaders = {
            'asqa': self.load_asqa,
            'eli5': self.load_eli5,
            'hagrid': self.load_hagrid,
            'msmarco': self.load_msmarco,
            'qampari': self.load_qampari,
            'natural_questions': self.load_natural_questions
        }
    
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
    
    def select_top_k_docs(self, question: str, docs: List[Dict[str, Any]], k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Improved document selection using existing relevance scores when available, otherwise use original order"""
        if k is None:
            k = self.top_k
        
        # Multi-hop questions need more context
        if len(docs) > 8 and any(doc.get('entity') for doc in docs):
            k = min(k * 2, len(docs))
        
        # Wiki sources benefit from more documents
        if len(docs) > 5 and any('wiki' in str(doc.get('source', '')).lower() for doc in docs):
            k = min(k * 2, len(docs))
            
        if not docs:
            return []
        
        docs = self.fix_empty_documents(docs)
        
        if len(docs) <= k:
            return docs
        
        selected_docs = self._select_by_existing_scores(docs, k)
        if selected_docs:
            return selected_docs
        
        print(f"Using original document order: selected {k} docs from {len(docs)} docs")
        return docs[:k]
    
    def _select_by_existing_scores(self, docs: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Select documents using existing relevance scores when available"""
        if not docs:
            return []
        
        scored_docs = []
        
        for i, doc in enumerate(docs):
            score = None
            
            if 'score' in doc and doc['score'] is not None:
                score = float(doc['score'])
            
            if score is not None:
                scored_docs.append((doc, score, i))
        
        if scored_docs:
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            selected_docs = [doc for doc, score, idx in scored_docs[:k]]
            print(f"Using existing relevance scores: selected {len(selected_docs)} docs from {len(scored_docs)} scored docs")
            return selected_docs
        
        return []
    
    
    def select_relevant_docs_for_answer(self, answer: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Improved document selection for post-generation approach"""
        if not docs:
            return []
        
        docs = self.fix_empty_documents(docs)
        
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            return docs[:self.top_k]
        
        relevant_docs = []
        doc_scores = defaultdict(float)
        
        try:
            for sentence in sentences:
                sentence_embedding = self.sentence_model.encode(sentence, convert_to_tensor=True)
                doc_texts = [doc.get('text', '') for doc in docs]
                
                valid_docs = []
                valid_texts = []
                for doc, text in zip(docs, doc_texts):
                    if text and len(text.strip()) > 10:
                        valid_docs.append(doc)
                        valid_texts.append(text)
                
                if not valid_texts:
                    continue
                
                doc_embeddings = self.sentence_model.encode(valid_texts, convert_to_tensor=True)
                similarities = util.pytorch_cos_sim(sentence_embedding, doc_embeddings)[0]
                
                for i, score in enumerate(similarities):
                    doc_scores[i] += score.item()
        
        except Exception as e:
            print(f"Error in document scoring: {e}")
            return docs[:self.top_k]
        
        if not doc_scores:
            return docs[:self.top_k]
        
        sorted_indices = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        selected_indices = sorted_indices[:self.top_k]
        
        return [docs[i] for i in selected_indices]
    
    def create_context_with_references(self, docs: List[Dict[str, Any]]) -> str:
        """Create context string with numbered references"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.get('title', f'Document {i}')
            text = doc.get('text') or doc.get('summary', '')
            context_parts.append(f"[{i}] {title}\n{text}")
        
        return '\n\n'.join(context_parts)
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using API"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.model_name == "deepseekv3":
                    response = self.client.chat.completions.create(
                        model="deepseek-ai/DeepSeek-V3",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers the following questions with proper citations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p
                    )
                    return response.choices[0].message.content.strip()
                    
                elif self.model_name == "llama-3.1-8b":
                    formatted_prompt = f"{prompt}"
                    result = self.client.text_generation(
                        formatted_prompt,
                        model="meta-llama/Llama-3.1-8B",
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        repetition_penalty=1.1,
                        stop=["Question:", "Documents:\n", "Q:"]
                    )
                    return result.strip()
                    
                elif self.model_name == "llama-3.1-8b-instruct":
                    response = self.client.chat.completions.create(
                        model="meta-llama/Llama-3.1-8B-Instruct",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers the following questions with proper citations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stop=["Question:", "Documents:\n", "Q:"]
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.model_name == "gemma-2-2b-it":
                    response = self.client.chat.completions.create(
                        model="google/gemma-2-2b-it",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers the following questions with proper citations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stop=["Question:", "Documents:\n", "Q:"]
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.model_name == "gemma-2-9b-it":
                    response = self.client.chat.completions.create(
                        model="google/gemma-2-9b-it",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers the following questions with proper citations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stop=["Question:", "Documents:\n", "Q:"]
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.model_name == "gemma-3-27b-it":
                    response = self.client.chat.completions.create(
                        model="google/gemma-3-27b-it",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers the following questions with proper citations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stop=["Question:", "Documents:\n", "Q:"]
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.model_name == "llama-3-70b-instruct":
                    response = self.client.chat.completions.create(
                        model="meta-llama/Meta-Llama-3-70B-Instruct",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers the following questions with proper citations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stop=["Question:", "Documents:\n", "Q:"]
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.model_name == "mistral-7b-instruct-v0.2":
                    response = self.client.chat.completions.create(
                        model="mistralai/Mistral-7B-Instruct-v0.2",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers the following questions with proper citations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stop=["Question:", "Documents:\n", "Q:"]
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.model_name == "qwen2.5-72b-instruct":
                    response = self.client.chat.completions.create(
                        model="Qwen/Qwen2.5-72B-Instruct",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers the following questions with proper citations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stop=["Question:", "Documents:\n", "Q:"]
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.model_name == "qwen2.5-14b-instruct":
                    response = self.client.chat.completions.create(
                        model="Qwen/Qwen2.5-14B-Instruct",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers the following questions with proper citations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stop=["Question:", "Documents:\n", "Q:"]
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.model_name == "qwen2.5-7b-instruct":
                    response = self.client.chat.completions.create(
                        model="Qwen/Qwen2.5-7B-Instruct",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers the following questions with proper citations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stop=["Question:", "Documents:\n", "Q:"]
                    )
                    return response.choices[0].message.content.strip()
                else:
                    api_params = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that answers the following questions with proper citations."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens
                    }
                    
                    if not ('gpt' in self.model_name.lower() or 'o1' in self.model_name.lower()):
                        api_params["top_p"] = self.top_p
                    
                    if self.seed is not None:
                        api_params["seed"] = self.seed
                    
                    response = self.client.chat.completions.create(**api_params)
                    return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error generating response (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

    def generate_response_with_temp(self, prompt: str, temperature: float) -> str:
        """Generate a response using a temporary temperature, restoring afterwards."""
        original_temperature = self.temperature
        self.temperature = temperature
        try:
            return self.generate_response(prompt)
        finally:
            self.temperature = original_temperature
    
    def post_retrieval_answering(self, question: str, docs: List[Dict[str, Any]], dataset_name: str = None) -> Tuple[str, List[int], List[Dict[str, Any]], str]:
        """Generate answer with citations using post-retrieval approach"""
        
        selected_docs = self.select_top_k_docs(question, docs)
        context = self.create_context_with_references(selected_docs)
        
        if dataset_name:
            try:
                if hasattr(self.prompt_manager, 'get_prompt') and 'num_few_shot' in self.prompt_manager.get_prompt.__code__.co_varnames:
                    prompt = self.prompt_manager.get_prompt(
                        dataset_name=dataset_name,
                        method=MethodType.POST_RETRIEVAL,
                        question=question,
                        context=context,
                        include_few_shot=True,
                        num_few_shot=self.num_few_shot
                    )
                else:
                    prompt = self.prompt_manager.get_prompt(
                        dataset_name=dataset_name,
                        method=MethodType.POST_RETRIEVAL,
                        question=question,
                        context=context,
                        include_few_shot=True
                    )
            except ValueError:
                prompt = self._create_basic_post_retrieval_prompt(question, context)
        else:
            prompt = self._create_basic_post_retrieval_prompt(question, context)
        
        response = self.generate_response(prompt)
        citations = self.extract_citations(response)
        
        return response, citations, selected_docs, None
    
    
    def post_generation_llm_short(self, question: str, docs: List[Dict[str, Any]], dataset_name: str = None) -> Tuple[str, List[int], List[Dict[str, Any]], str]:
        """Generate answer first, then add citations using LLM-based post-generation approach with short citation prompts"""
        
        # Use clean prompting (without citation instructions) for post-generation
        if dataset_name and hasattr(self.prompt_manager, 'get_clean_instruction'):
            try:
                # Get clean instruction without citation parts
                clean_instruction = self.prompt_manager.get_clean_instruction(dataset_name)
                
                # Get clean few-shot examples (without citations)
                clean_examples = self.prompt_manager.get_clean_few_shot_examples(dataset_name, self.num_few_shot)
                
                # Build clean prompt
                prompt_parts = []
                
                # Add instruction first
                prompt_parts.append(f"{clean_instruction}\n\n")
                
                # Add few-shot examples if available
                for example in clean_examples:
                    prompt_parts.append(f"Question: {example['question']}\nAnswer: {example['answer']}\n\n")
                
                # Add the main task
                prompt_parts.append(f"Question: {question}\nAnswer: ")
                
                initial_prompt = ''.join(prompt_parts)
                
            except Exception as e:
                print(f"Error getting clean prompt: {e}")
                initial_prompt = self._create_basic_initial_prompt(question)
        else:
            initial_prompt = self._create_basic_initial_prompt(question)
        
        initial_answer = self.generate_response_with_temp(initial_prompt, self.temperature)
        selected_docs = self.select_top_k_docs(initial_answer, docs)
        context = self.create_context_with_references(selected_docs)
        
        if dataset_name:
            try:
                prompts = self.prompt_manager.get_prompt(
                    dataset_name=dataset_name,
                    method=MethodType.POST_GENERATION_LLM_SHORT,
                    question=question,
                    context=context,
                    include_few_shot=True
                )
                citation_prompt = prompts['citation_prompt'].format(initial_answer=initial_answer)
            except ValueError:
                citation_prompt = self._create_basic_citation_prompt(initial_answer, context)
        else:
            citation_prompt = self._create_basic_citation_prompt(initial_answer, context)
        
        # Generate answer with citations (lower temperature for stability)
        final_answer = self.generate_response_with_temp(citation_prompt, self.citation_temperature)
        
        # Extract citation numbers
        citations = self.extract_citations(final_answer)
        
        return final_answer, citations, selected_docs, initial_answer
    
    def post_generation_llm_long(self, question: str, docs: List[Dict[str, Any]], dataset_name: str = None) -> Tuple[str, List[int], List[Dict[str, Any]], str]:
        """Generate answer first, then add citations using LLM-based post-generation approach with long citation prompts"""
        
        if dataset_name and hasattr(self.prompt_manager, 'get_clean_instruction'):
            try:
                clean_instruction = self.prompt_manager.get_clean_instruction(dataset_name)
                clean_examples = self.prompt_manager.get_clean_few_shot_examples(dataset_name, self.num_few_shot)
                
                prompt_parts = []
                for example in clean_examples:
                    prompt_parts.append(f"{clean_instruction}\n\nQuestion: {example['question']}\n\nAnswer: {example['answer']}\n\n")
                
                prompt_parts.append(f"Question: {question}\n\nAnswer: ")
                initial_prompt = ''.join(prompt_parts)
                
            except Exception as e:
                print(f"Error getting clean prompt: {e}")
                initial_prompt = self._create_basic_initial_prompt(question)
        else:
            initial_prompt = self._create_basic_initial_prompt(question)
        
        initial_answer = self.generate_response_with_temp(initial_prompt, self.temperature)
        selected_docs = self.select_top_k_docs(initial_answer, docs)
        context = self.create_context_with_references(selected_docs)
        
        if dataset_name:
            try:
                prompts = self.prompt_manager.get_prompt(
                    dataset_name=dataset_name,
                    method=MethodType.POST_GENERATION_LLM_LONG,
                    question=question,
                    context=context,
                    include_few_shot=True
                )
                citation_prompt = prompts['citation_prompt'].format(initial_answer=initial_answer)
            except ValueError:
                citation_prompt = self._create_basic_citation_prompt(initial_answer, context)
        else:
            citation_prompt = self._create_basic_citation_prompt(initial_answer, context)
        
        # Generate answer with citations (lower temperature for stability)
        final_answer = self.generate_response_with_temp(citation_prompt, self.citation_temperature)
        
        # Extract citation numbers
        citations = self.extract_citations(final_answer)
        
        return final_answer, citations, selected_docs, initial_answer
    
    def post_generation_tfidf(self, question: str, docs: List[Dict[str, Any]], dataset_name: str = None) -> Tuple[str, List[int], List[Dict[str, Any]], str]:
        """
        Generate answer first, then add citations using TF-IDF post-generation approach (ALCE-style)
        
        Args:
            question: The question to answer
            docs: List of available documents
            dataset_name: Name of the dataset (for sophisticated prompting)
            
        Returns:
            Tuple of (final_answer, citations, selected_docs)
        """
        
        # Use clean prompting (without citation instructions) for TF-IDF post-generation
        if dataset_name and hasattr(self.prompt_manager, 'get_clean_instruction'):
            try:
                # Get clean instruction without citation parts
                clean_instruction = self.prompt_manager.get_clean_instruction(dataset_name)
                
                # Get clean few-shot examples (without citations)
                clean_examples = self.prompt_manager.get_clean_few_shot_examples(dataset_name, self.num_few_shot)
                
                # Build clean prompt
                prompt_parts = []
                
                # Add instruction first
                prompt_parts.append(f"{clean_instruction}\n\n")
                
                # Add few-shot examples if available
                for example in clean_examples:
                    prompt_parts.append(f"Question: {example['question']}\nAnswer: {example['answer']}\n\n")
                
                # Add the main task
                prompt_parts.append(f"Question: {question}\nAnswer: ")
                
                initial_prompt = ''.join(prompt_parts)
                
            except Exception as e:
                print(f"Error getting clean prompt: {e}")
                initial_prompt = self._create_basic_initial_prompt(question)
        else:
            initial_prompt = self._create_basic_initial_prompt(question)
        
        initial_answer = self.generate_response_with_temp(initial_prompt, self.temperature)
        searcher = SearcherWithinDocs(docs, retriever="tfidf", device="cpu")
        
        # Process answer sentence by sentence (ALCE approach)
        if dataset_name == 'qampari':
            # Special handling for QAMPARI: split into list items and search each
            raw_items = [x.strip() for x in re.split(r",|\n|;", initial_answer) if x.strip()]
            # Prefix question for better matching (as before) but keep clean item for final formatting
            sentences = [question + ' ' + item for item in raw_items]
        else:
            try:
                from nltk import sent_tokenize
                sentences = sent_tokenize(initial_answer)
                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
                
                if len(sentences) < 2:
                    sentences = [initial_answer.strip()]
            except ImportError:
                normalized_text = re.sub(r'\s+', ' ', initial_answer.strip())
                sentences = re.split(r'(?<!\d)[.!?]+(?!\d)', normalized_text)
                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
                
                if not sentences:
                    sentences = [initial_answer.strip()]
        
        def add_citation_before_punctuation(sentence: str, citation: int) -> str:
            """Add citation before the final punctuation mark in a sentence"""
            # Find the last punctuation mark
            punctuation_pattern = r'([.!?]+)$'
            match = re.search(punctuation_pattern, sentence)
            
            if match:
                # Split sentence and punctuation
                text_part = sentence[:match.start()]
                punctuation_part = match.group(1)
                return f"{text_part} [{citation}]{punctuation_part}"
            else:
                # No punctuation found, just add citation at the end
                return f"{sentence} [{citation}]"
        
        cited_sentences = []
        all_citations = set()
        cited_docs = set()
        
        for sent in sentences:
            if not sent:
                continue
                
            # Find best matching document for this sentence
            try:
                # Use score thresholding to avoid spurious citations
                if dataset_name == 'qampari':
                    best_doc_id, best_score = searcher.search_with_score(sent)
                    score_threshold = 0.08  # conservative TF-IDF cosine threshold for short items
                    if best_score >= score_threshold and best_doc_id is not None:
                        cited_sent = add_citation_before_punctuation(sent, best_doc_id + 1)
                        cited_sentences.append(cited_sent)
                        all_citations.add(best_doc_id + 1)
                        cited_docs.add(best_doc_id)
                    else:
                        # No confident support — keep item without citation to avoid misleading refs
                        cited_sentences.append(sent)
                else:
                    best_doc_id = searcher.search(sent)
                    if best_doc_id is not None:
                        cited_sent = add_citation_before_punctuation(sent, best_doc_id + 1)
                        cited_sentences.append(cited_sent)
                        all_citations.add(best_doc_id + 1)
                        cited_docs.add(best_doc_id)
                    else:
                        cited_sentences.append(sent)
            except Exception as e:
                print(f"Error in TF-IDF search for sentence: {e}")
                cited_sentences.append(sent)
        
        # Combine sentences back into final answer
        if dataset_name == 'qampari':
            # Ensure exactly one citation per item if any, and strip prefixed question
            cleaned_items = []
            for s in cited_sentences:
                item = s.replace(question, '').strip()
                # If multiple citations appear, keep only the first one
                item = re.sub(r"(\[\d+\])(\[\d+\])+", r"\1", item)
                cleaned_items.append(item)
            final_answer = ", ".join(cleaned_items).rstrip(",")
        else:
            final_answer = " ".join(cited_sentences)
        
        # Return documents in original order (same as other methods)
        # The main pipeline will map citations to documents using doc_index = citation_num - 1
        sorted_citations = sorted([int(c) for c in all_citations])
        
        return final_answer, sorted_citations, docs, initial_answer
    
    def extract_citations(self, text: str) -> List[int]:
        """Extract citation numbers from text (handles both [1][2][3] and [1], [2], [3] formats)"""
        citations = []
        # Pattern to match [number] with optional comma and space after
        pattern = r'\[(\d+)\](?:,\s*)?'
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                citations.append(int(match))
            except ValueError:
                continue
        return sorted(list(set(citations)))
    
    def _create_basic_post_retrieval_prompt(self, question: str, context: str) -> str:
        """Create a basic post-retrieval prompt (fallback)"""
        return f"""Given the following numbered documents and a question, provide a comprehensive answer with numeric in-text citations [1][2][3] etc. for each claim or fact you mention.

Documents:
{context}

Question: {question}

Instructions:
1. Answer the question comprehensively using information from the documents
2. Add numeric in-text citations [n] immediately after each claim or fact
3. Only cite documents that directly support the claim
4. Use multiple citations if multiple documents support the same claim (format: [1][2][3] - NO COMMAS between brackets)
5. Be specific and accurate in your citations
6. For multi-hop reasoning questions, cite ALL documents that contribute to your reasoning chain
7. If you need to make multiple logical steps, cite the documents that support each step
8. CRITICAL: Only state facts that are explicitly mentioned in the provided documents
9. If you cannot find the exact answer in the documents, say "I don't have enough information to answer this question"
10. Do not make assumptions or add information not present in the documents

Answer:"""
    
    def _create_basic_initial_prompt(self, question: str) -> str:
        """Create a basic initial prompt for post-generation (fallback)"""
        return f"""Answer the following question comprehensively:

Question: {question}

Answer:"""
    
    def _create_basic_citation_prompt(self, initial_answer: str, context: str) -> str:
        """Create a basic citation prompt for post-generation (fallback)"""
        return f"""Given the following answer and numbered documents, add numeric in-text citations [1][2][3] etc. to support each claim in the answer.

Original Answer:
{initial_answer}

Documents:
{context}

Instructions:
1. Keep the original answer text intact - DO NOT include the document text in your response
2. Add numeric in-text citations [n] immediately after each claim that can be supported by the documents
3. Only cite documents that directly support the claim
4. Use multiple citations if multiple documents support the same claim (format: [1][2][3] - NO COMMAS between brackets)
5. If a claim cannot be supported by any document, leave it without citation
6. Return ONLY the answer with citations - do not include the document text or any other content

Answer with Citations:"""
    
    # Dataset loaders (same as original)
    def load_dataset(self, dataset_name: str, file_path: str) -> List[Dict[str, Any]]:
        """Load dataset based on its type"""
        if dataset_name in self.dataset_loaders:
            return self.dataset_loaders[dataset_name](file_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def load_asqa(self, file_path: str) -> List[Dict[str, Any]]:
        """Load ASQA dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = []
        for item in data:
            processed.append({
                'id': item.get('sample_id', ''),
                'question': item['question'],
                'docs': self.fix_empty_documents(item['docs']),
                'reference_answer': item.get('answer', ''),
                'dataset': 'asqa'
            })
        return processed
    
    def load_eli5(self, file_path: str) -> List[Dict[str, Any]]:
        """Load ELI5 dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = []
        for item in data:
            processed.append({
                'id': str(len(processed)),
                'question': item['question'],
                'docs': self.fix_empty_documents(item['docs']),
                'reference_answer': item.get('answer', ''),
                'claims': item.get('claims', []),
                'dataset': 'eli5'
            })
        return processed
    
    def load_hagrid(self, file_path: str) -> List[Dict[str, Any]]:
        """Load HAGRID dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = []
        for item in data:
            # Extract documents from quotes
            docs = []
            for quote in item.get('quotes', []):
                docs.append({
                    'id': quote['docid'],
                    'text': quote['text'],
                    'title': f"Document {quote['idx']}"
                })
            
            processed.append({
                'id': str(item.get('query_id', len(processed))),
                'question': item['query'],
                'docs': self.fix_empty_documents(docs),
                'reference_answer': ' '.join([ans['answer'] for ans in item.get('answers', [])]),
                'dataset': 'hagrid'
            })
        return processed
    
    def load_msmarco(self, file_path: str) -> List[Dict[str, Any]]:
        """Load MS MARCO dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = []
        for item in data:
            # Extract documents from passages
            docs = []
            passages = item.get('passages', {})
            is_selected = passages.get('is_selected', [])
            
            for i, (text, url) in enumerate(zip(passages.get('passage_text', []), 
                                                passages.get('url', []))):
                # Use is_selected as relevance score (1.0 for selected, 0.0 for not selected)
                score = 1.0 if i < len(is_selected) and is_selected[i] == 1 else 0.0
                
                docs.append({
                    'id': str(i),
                    'text': text,
                    'title': url if url else f"Passage {i+1}",
                    'url': url,
                    'score': score  # Add relevance score based on is_selected
                })
            
            processed.append({
                'id': str(item.get('query_id', len(processed))),
                'question': item['query'],
                'docs': self.fix_empty_documents(docs),
                'reference_answer': ' '.join(item.get('answers', [])),
                'dataset': 'msmarco'
            })
        return processed
    
    def load_qampari(self, file_path: str) -> List[Dict[str, Any]]:
        """Load QAMPARI dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = []
        for item in data:
            processed.append({
                'id': item.get('id', str(len(processed))),
                'question': item['question'],
                'docs': self.fix_empty_documents(item.get('docs', [])),
                'reference_answer': item.get('answer', ''),
                'answers_list': item.get('answers', []),
                'dataset': 'qampari'
            })
        return processed
    
    def load_triviaqa(self, file_path: str) -> List[Dict[str, Any]]:
        """Load TriviaQA dataset (both web and wiki versions)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = []
        for item in data:
            docs = []
            
            # Try search_results first (for web version)
            search_results = item.get('search_results', {})
            if search_results and search_results.get('description'):
                # Extract documents from search results (web version)
                for i, (desc, title, url) in enumerate(zip(
                    search_results.get('description', []),
                    search_results.get('title', []),
                    search_results.get('url', [])
                )):
                    # Also try to get search context if available
                    search_contexts = search_results.get('search_context', [])
                    context = search_contexts[i] if i < len(search_contexts) else desc
                    
                    docs.append({
                        'id': str(i),
                        'text': context,
                        'title': title,
                        'url': url,
                        'summary': desc
                    })
            
            # Try entity_pages if search_results is empty (for wiki version)
            elif item.get('entity_pages', {}).get('wiki_context'):
                entity_pages = item.get('entity_pages', {})
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
            
            processed.append({
                'id': item.get('question_id', str(len(processed))),
                'question': item['question'],
                'docs': self.fix_empty_documents(docs),
                'reference_answer': item.get('answer', {}).get('value', ''),
                'dataset': 'triviaqa'
            })
        return processed
    
    def load_twowikimultihop(self, file_path: str) -> List[Dict[str, Any]]:
        """Load 2WikiMultihop dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = []
        for item in data:
            # Extract documents from context
            docs = []
            for i, (entity, facts) in enumerate(item.get('context', [])):
                text = ' '.join(facts) if isinstance(facts, list) else str(facts)
                docs.append({
                    'id': str(i),
                    'text': text,
                    'title': entity,
                    'entity': entity
                })
            
            processed.append({
                'id': item.get('_id', str(len(processed))),
                'question': item['question'],
                'docs': self.fix_empty_documents(docs),
                'reference_answer': item.get('answer', ''),
                'supporting_facts': item.get('supporting_facts', []),
                'dataset': 'twowikimultihop'
            })
        return processed
    
    def load_natural_questions(self, file_path: str) -> List[Dict[str, Any]]:
        """Load Natural Questions dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = []
        for item in data:
            # Extract documents from relevant_docs
            docs = []
            for doc in item.get('relevant_docs', []):
                docs.append({
                    'id': doc.get('doc_id', str(len(docs))),
                    'text': doc.get('text', ''),
                    'title': doc.get('title', ''),
                    'score': doc.get('score', 0)
                })
            
            processed.append({
                'id': str(item.get('query_id', len(processed))),
                'question': item['question'],
                'docs': self.fix_empty_documents(docs),
                'reference_answer': ' '.join(item.get('answers', [])),
                'answers': item.get('answers', []),
                'dataset': 'natural_questions'
            })
        return processed
    
    def run_pipeline(self, dataset_name: str, file_path: str, output_path: str, num_examples: int = None, save_initial_answers: bool = False):
        """Run the complete pipeline on a dataset"""
        print(f"Loading dataset: {dataset_name} from {file_path}")
        data = self.load_dataset(dataset_name, file_path)
        
        # Limit the number of examples if specified
        if num_examples is not None and num_examples > 0:
            data = data[:num_examples]
            print(f"Limited to first {num_examples} examples")
        
        results = []
        initial_answers = []  # Store initial answers for post-generation approaches
        
        for item in tqdm(data, desc=f"Processing {dataset_name}"):
            try:
                # Generate answer based on approach
                if self.approach == 'post-retrieval':
                    answer, citations, selected_docs, initial_answer = self.post_retrieval_answering(
                        item['question'], 
                        item['docs'],
                        dataset_name=dataset_name
                    )
                    # initial_answer will be None for post-retrieval
                elif self.approach == 'post-generation-tfidf':
                    answer, citations, selected_docs, initial_answer = self.post_generation_tfidf(
                        item['question'], 
                        item['docs'],
                        dataset_name=dataset_name
                    )
                    # initial_answer will be the clean answer before TF-IDF citation addition
                elif self.approach == 'post-generation-llm-short':
                    answer, citations, selected_docs, initial_answer = self.post_generation_llm_short(
                        item['question'], 
                        item['docs'],
                        dataset_name=dataset_name
                    )
                elif self.approach == 'post-generation-llm-long':
                    answer, citations, selected_docs, initial_answer = self.post_generation_llm_long(
                        item['question'], 
                        item['docs'],
                        dataset_name=dataset_name
                    )
                else:
                    raise ValueError(f"Unknown approach: {self.approach}")
                
                # Create cited evidences mapping
                cited_evidences = {}
                for citation_num in citations:
                    # Citation numbers are 1-indexed, so subtract 1 for array access
                    doc_index = int(citation_num) - 1
                    if 0 <= doc_index < len(selected_docs):
                        cited_evidences[str(int(citation_num))] = {
                            'id': selected_docs[doc_index].get('id', str(doc_index)),
                            'title': selected_docs[doc_index].get('title', f'Document {citation_num}'),
                            'text': selected_docs[doc_index].get('text', ''),
                            'url': selected_docs[doc_index].get('url', ''),
                            'source': selected_docs[doc_index].get('source', ''),
                            'summary': selected_docs[doc_index].get('summary', '')
                        }
                
                # Store result
                result = {
                    'id': item['id'],
                    'dataset': dataset_name,
                    'question': item['question'],
                    'generated_answer': answer,
                    'citations': citations,
                    'cited_evidences': cited_evidences,
                    'reference_answer': item.get('reference_answer', ''),
                    'approach': self.approach,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
                # Store initial answer if available and saving is requested
                if save_initial_answers and initial_answer is not None:
                    initial_answer_data = {
                        'id': item['id'],
                        'dataset': dataset_name,
                        'question': item['question'],
                        'initial_answer': initial_answer,
                        'final_answer': answer,
                        'approach': self.approach,
                        'timestamp': datetime.now().isoformat()
                    }
                    initial_answers.append(initial_answer_data)
                
            except Exception as e:
                print(f"Error processing item {item['id']}: {e}")
                continue
        
        # Save results
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if there's a directory component
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save initial answers if requested and available
        if save_initial_answers and initial_answers:
            # Create initial answers filename by adding "_initial_answers" before the file extension
            base_name = os.path.splitext(output_path)[0]
            extension = os.path.splitext(output_path)[1]
            initial_answers_path = f"{base_name}_initial_answers{extension}"
            
            with open(initial_answers_path, 'w', encoding='utf-8') as f:
                json.dump(initial_answers, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(initial_answers)} initial answers to: {initial_answers_path}")
        
        print(f"Results saved to: {output_path}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Improved Attribution Pipeline for Question Answering')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--approach', type=str, choices=['post-retrieval', 'post-generation-llm', 'post-generation-tfidf'], 
                       default='post-retrieval', help='Attribution approach')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line args
    config['approach'] = args.approach
    
    # Create pipeline
    pipeline = ImprovedAttributionPipeline(config)
    
    # Run pipeline
    pipeline.run_pipeline(args.dataset, args.input, args.output)


if __name__ == "__main__":
    main()
