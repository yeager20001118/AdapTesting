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


# class FakeNewsDetection(TextTSTDataset):
#     """
#     Real vs Fake news detection using LIAR dataset or similar.
    
#     Uses established fake news datasets from HuggingFace for comparing
#     real news articles vs fabricated/fake news articles.
#     """
    
#     def __init__(
#         self,
#         root: str = './data',
#         N: int = 400,
#         M: int = 400,
#         download: bool = True,
#         dataset_name: str = 'liar',
#         **kwargs
#     ):
#         """
#         Args:
#             dataset_name (str): Which dataset to use ('liar', 'fake_news', etc.)
#         """
#         if not HF_AVAILABLE:
#             raise ImportError("HuggingFace datasets required. Install with: pip install datasets")
            
#         self.dataset_name = dataset_name
#         super().__init__(root, N, M, download, **kwargs)
    
#     def _download(self):
#         if not self._check_exists():
#             try:
#                 if self.dataset_name == 'liar':
#                     self.dataset = load_dataset("liar", cache_dir=self.root)
#                 else:
#                     # Try alternative fake news datasets
#                     self.dataset = load_dataset("GonzaloA/fake_news", cache_dir=self.root)
#                 print(f"Downloaded {self.dataset_name} dataset")
#             except Exception as e:
#                 raise RuntimeError(f"Could not download {self.dataset_name} dataset: {e}")
    
#     def _check_exists(self) -> bool:
#         return hasattr(self, 'dataset') and self.dataset is not None
    
#     def _load_data(self) -> Tuple[List[str], List[str]]:
#         if not hasattr(self, 'dataset'):
#             self._download()
        
#         if 'train' in self.dataset:
#             data = self.dataset['train']
#         else:
#             data = list(self.dataset.values())[0]
        
#         real_news = []
#         fake_news = []
        
#         np.random.seed(self.seed)
#         indices = np.random.permutation(len(data))
        
#         for idx in indices:
#             idx = int(idx)  # Convert numpy int32 to Python int
#             row = data[idx]
            
#             # LIAR dataset format
#             if 'label' in row and 'statement' in row:
#                 # LIAR has 6 labels: pants-fire, false, barely-true, half-true, mostly-true, true
#                 if row['label'] in ['true', 'mostly-true'] and len(real_news) < self.N:
#                     real_news.append(row['statement'])
#                 elif row['label'] in ['pants-fire', 'false'] and len(fake_news) < self.M:
#                     fake_news.append(row['statement'])
            
#             # General fake news format
#             elif 'title' in row and 'label' in row:
#                 if row['label'] == 'REAL' and len(real_news) < self.N:
#                     real_news.append(row['title'])
#                 elif row['label'] == 'FAKE' and len(fake_news) < self.M:
#                     fake_news.append(row['title'])
            
#             if len(real_news) >= self.N and len(fake_news) >= self.M:
#                 break
        
#         # Ensure we have enough samples
#         if len(real_news) < self.N:
#             repeats_needed = self.N - len(real_news)
#             if len(real_news) > 0:
#                 real_news.extend(np.random.choice(real_news, repeats_needed).tolist())
        
#         if len(fake_news) < self.M:
#             repeats_needed = self.M - len(fake_news)
#             if len(fake_news) > 0:
#                 fake_news.extend(np.random.choice(fake_news, repeats_needed).tolist())
        
#         return real_news[:self.N], fake_news[:self.M]


# class DomainShift(TextTSTDataset):
#     """
#     Text from different domains using AG News or similar multi-class datasets.
    
#     Compares text from different domains/categories to detect domain shifts.
#     """
    
#     def __init__(
#         self,
#         root: str = './data',
#         N: int = 400,
#         M: int = 400,
#         download: bool = True,
#         dataset_name: str = 'ag_news',
#         domain_pair: Tuple[int, int] = (0, 1),
#         **kwargs
#     ):
#         """
#         Args:
#             dataset_name (str): Which dataset to use ('ag_news', '20newsgroups', etc.)
#             domain_pair (Tuple[int, int]): Which two domains/classes to compare
#         """
#         if not HF_AVAILABLE:
#             raise ImportError("HuggingFace datasets required. Install with: pip install datasets")
            
#         self.dataset_name = dataset_name
#         self.domain_pair = domain_pair
#         super().__init__(root, N, M, download, **kwargs)
    
#     def _download(self):
#         if not self._check_exists():
#             try:
#                 if self.dataset_name == 'ag_news':
#                     self.dataset = load_dataset("ag_news", cache_dir=self.root)
#                 elif self.dataset_name == '20newsgroups':
#                     self.dataset = load_dataset("newsgroup", "20newsgroups", cache_dir=self.root)
#                 else:
#                     self.dataset = load_dataset(self.dataset_name, cache_dir=self.root)
#                 print(f"Downloaded {self.dataset_name} dataset")
#             except Exception as e:
#                 raise RuntimeError(f"Could not download {self.dataset_name} dataset: {e}")
    
#     def _check_exists(self) -> bool:
#         return hasattr(self, 'dataset') and self.dataset is not None
    
#     def _load_data(self) -> Tuple[List[str], List[str]]:
#         if not hasattr(self, 'dataset'):
#             self._download()
        
#         if 'train' in self.dataset:
#             data = self.dataset['train']
#         else:
#             data = list(self.dataset.values())[0]
        
#         domain1_texts = []
#         domain2_texts = []
        
#         domain1_label, domain2_label = self.domain_pair
        
#         np.random.seed(self.seed)
#         indices = np.random.permutation(len(data))
        
#         for idx in indices:
#             idx = int(idx)  # Convert numpy int32 to Python int
#             row = data[idx]
            
#             if 'label' in row and 'text' in row:
#                 if row['label'] == domain1_label and len(domain1_texts) < self.N:
#                     domain1_texts.append(row['text'])
#                 elif row['label'] == domain2_label and len(domain2_texts) < self.M:
#                     domain2_texts.append(row['text'])
            
#             if len(domain1_texts) >= self.N and len(domain2_texts) >= self.M:
#                 break
        
#         return domain1_texts[:self.N], domain2_texts[:self.M]


# class SentimentShift(TextTSTDataset):
#     """
#     Text with different sentiment distributions using established sentiment datasets.
    
#     Uses datasets like IMDB, Amazon reviews, or Stanford Sentiment Treebank
#     to compare different sentiment distributions.
#     """
    
#     def __init__(
#         self,
#         root: str = './data',
#         N: int = 400,
#         M: int = 400,
#         download: bool = True,
#         dataset_name: str = 'imdb',
#         **kwargs
#     ):
#         """
#         Args:
#             dataset_name (str): Which sentiment dataset to use ('imdb', 'amazon_polarity', 'sst')
#         """
#         if not HF_AVAILABLE:
#             raise ImportError("HuggingFace datasets required. Install with: pip install datasets")
            
#         self.dataset_name = dataset_name
#         super().__init__(root, N, M, download, **kwargs)
    
#     def _download(self):
#         if not self._check_exists():
#             try:
#                 if self.dataset_name == 'imdb':
#                     self.dataset = load_dataset("imdb", cache_dir=self.root)
#                 elif self.dataset_name == 'amazon_polarity':
#                     self.dataset = load_dataset("amazon_polarity", cache_dir=self.root)
#                 elif self.dataset_name == 'sst':
#                     self.dataset = load_dataset("sst", cache_dir=self.root)
#                 else:
#                     self.dataset = load_dataset(self.dataset_name, cache_dir=self.root)
#                 print(f"Downloaded {self.dataset_name} dataset")
#             except Exception as e:
#                 raise RuntimeError(f"Could not download {self.dataset_name} dataset: {e}")
    
#     def _check_exists(self) -> bool:
#         return hasattr(self, 'dataset') and self.dataset is not None
    
#     def _load_data(self) -> Tuple[List[str], List[str]]:
#         if not hasattr(self, 'dataset'):
#             self._download()
        
#         if 'train' in self.dataset:
#             data = self.dataset['train']
#         else:
#             data = list(self.dataset.values())[0]
        
#         positive_texts = []
#         negative_texts = []
        
#         np.random.seed(self.seed)
#         indices = np.random.permutation(len(data))
        
#         for idx in indices:
#             idx = int(idx)  # Convert numpy int32 to Python int
#             row = data[idx]
            
#             if 'label' in row and 'text' in row:
#                 # Binary sentiment: 0 = negative, 1 = positive
#                 if row['label'] == 1 and len(positive_texts) < self.N:
#                     positive_texts.append(row['text'])
#                 elif row['label'] == 0 and len(negative_texts) < self.M:
#                     negative_texts.append(row['text'])
            
#             if len(positive_texts) >= self.N and len(negative_texts) >= self.M:
#                 break
        
#         return positive_texts[:self.N], negative_texts[:self.M]


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
