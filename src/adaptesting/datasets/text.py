"""
Text datasets for two-sample testing using real datasets.

These datasets use established sources like HuggingFace datasets for 
authentic text data representing different conditions.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import requests
import json
from .base import TextTSTDataset

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: HuggingFace datasets not available. Install with: pip install datasets")


class HumanAIDetection(TextTSTDataset):
    """
    Human vs AI generated text using M4 dataset from HuggingFace.
    
    Uses the M4 (Multilingual, Multigenre, Multimodel, Machine-Generated) dataset
    for detecting AI-generated text vs human-written text.
    
    Citation: "M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection"
    """
    
    def __init__(
        self,
        root: str = './data',
        N: int = 500,
        M: int = 500,
        download: bool = True,
        subset: str = 'english',
        domain: str = 'wikipedia',
        **kwargs
    ):
        """
        Args:
            subset (str): Language subset ('english', 'chinese', 'urdu', etc.)
            domain (str): Domain subset ('wikipedia', 'reddit', 'wikihow', etc.)
        """
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace datasets required. Install with: pip install datasets")
            
        self.subset = subset
        self.domain = domain
        super().__init__(root, N, M, download, **kwargs)
    
    def _download(self):
        if not self._check_exists():
            # Download M4 dataset from HuggingFace
            try:
                self.dataset = load_dataset("m4", self.subset, cache_dir=self.root)
                print(f"Downloaded M4 dataset: {self.subset} subset")
            except Exception as e:
                print(f"Could not download M4 dataset: {e}")
                # Fallback to a simpler dataset
                try:
                    self.dataset = load_dataset(
                        "Hello-SimpleAI/HC3", 
                        cache_dir=self.root,
                        trust_remote_code=True
                    )
                    print("Using HC3 dataset as fallback")
                except Exception as e2:
                    raise RuntimeError(f"Could not download datasets: M4 failed with {e}, HC3 failed with {e2}")
    
    def _check_exists(self) -> bool:
        return hasattr(self, 'dataset') and self.dataset is not None
    
    def _load_data(self) -> Tuple[List[str], List[str]]:
        if not hasattr(self, 'dataset'):
            self._download()
        
        # Handle different dataset structures
        if 'train' in self.dataset:
            data = self.dataset['train']
            data = data.filter(lambda x: x['source'] == 'reddit_eli5')
        else:
            print("Failing to load data from source 'reddit_eli5'.")
        
        def flatten(list_of_lists):
            # Flatten a list of lists where each inner list contains one string
            return [item for sublist in list_of_lists for item in sublist]

        human_texts = flatten(data['human_answers'])
        ai_texts = flatten(data['chatgpt_answers'])
        
        # Type-I error check: draw both samples from the same distribution
        if self.t1_check:
            print("Type-I error check: Drawing both X and Y from AI-generated distribution")
            # Make sure we have enough AI samples for both X and Y
            total_needed = self.N + self.M
            if len(ai_texts) < total_needed:
                # Repeat samples if needed
                ai_texts = ai_texts * ((total_needed // len(ai_texts)) + 1)
            
            return ai_texts[:self.N], ai_texts[self.N:self.N+self.M]
        
        return human_texts[:self.N], ai_texts[:self.M]


# Keep HC3 as an alias for compatibility with the H3C name you mentioned
class HC3(HumanAIDetection):
    """
    HC3 (Human ChatGPT Comparison Corpus) dataset.
    
    This is an alias for HumanAIDetection using the HC3 dataset specifically.
    HC3 contains human-written text paired with ChatGPT-generated responses
    across multiple domains.
    
    Citation: "How Close is ChatGPT to Human Experts? Comparison Corpus, 
    Evaluation, and Detection" (https://arxiv.org/abs/2301.07597)
    """
    
    def __init__(self, **kwargs):
        # Override to use HC3 specifically
        super().__init__(**kwargs)
    
    def _download(self):
        if not self._check_exists():
            try:
                # HC3 requires trust_remote_code=True
                print("Loading HC3 dataset (may take a moment)...")
                # self.dataset = load_dataset(
                #     "Hello-SimpleAI/HC3", 
                #     cache_dir=self.root,
                #     trust_remote_code=True
                # )
                self.dataset = load_dataset(
                    "Hello-SimpleAI/HC3",
                    "all",
                    trust_remote_code=True,
                    cache_dir=self.root,
                )
                print("Downloaded HC3 dataset")
            except Exception as e:
                print(f"Could not download HC3 dataset: {e}")
                # Try alternative approach with specific subset
                try:
                    print("Trying HC3 with specific subset...")
                    self.dataset = load_dataset(
                        "Hello-SimpleAI/HC3", 
                        "all",
                        cache_dir=self.root,
                        trust_remote_code=True
                    )
                    print("Downloaded HC3 dataset with subset")
                except Exception as e2:
                    raise RuntimeError(f"Could not download HC3 dataset: {e2}")
