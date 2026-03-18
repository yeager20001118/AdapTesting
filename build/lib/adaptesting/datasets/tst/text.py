"""
Text datasets for two-sample testing.
"""

from typing import Tuple, List

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
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace datasets required. Install with: pip install datasets")

        self.subset = subset
        self.domain = domain
        super().__init__(root, N, M, download, **kwargs)

    def _download(self):
        if not self._check_exists():
            try:
                self.dataset = load_dataset("m4", self.subset, cache_dir=self.root)
                print(f"Downloaded M4 dataset: {self.subset} subset")
            except Exception as e:
                print(f"Could not download M4 dataset: {e}")
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

        if 'train' in self.dataset:
            data = self.dataset['train']
            data = data.filter(lambda x: x['source'] == 'reddit_eli5')
        else:
            print("Failing to load data from source 'reddit_eli5'.")

        def flatten(list_of_lists):
            return [item for sublist in list_of_lists for item in sublist]

        human_texts = flatten(data['human_answers'])
        ai_texts = flatten(data['chatgpt_answers'])

        if self.t1_check:
            print("Type-I error check: Drawing both X and Y from AI-generated distribution")
            total_needed = self.N + self.M
            if len(ai_texts) < total_needed:
                ai_texts = ai_texts * ((total_needed // len(ai_texts)) + 1)
            return ai_texts[:self.N], ai_texts[self.N:self.N + self.M]

        return human_texts[:self.N], ai_texts[:self.M]


class HC3(HumanAIDetection):
    """
    HC3 (Human ChatGPT Comparison Corpus) dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _download(self):
        if not self._check_exists():
            try:
                print("Loading HC3 dataset (may take a moment)...")
                self.dataset = load_dataset(
                    "Hello-SimpleAI/HC3",
                    "all",
                    trust_remote_code=True,
                    cache_dir=self.root,
                )
                print("Downloaded HC3 dataset")
            except Exception as e:
                print(f"Could not download HC3 dataset: {e}")
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


__all__ = [
    'HumanAIDetection',
    'HC3',
]
