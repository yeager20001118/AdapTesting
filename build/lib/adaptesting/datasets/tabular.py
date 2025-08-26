"""
Tabular datasets for two-sample testing using real datasets.

These datasets use established sources like sklearn datasets, UCI ML repository,
and other real-world datasets for authentic tabular data comparisons.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, load_iris, load_digits
from sklearn.model_selection import train_test_split
import gzip
import urllib.request
from .base import TabularTSTDataset


# class BreastCancer(TabularTSTDataset):
#     """
#     Breast cancer dataset: malignant vs benign tumors.
    
#     Uses the sklearn breast cancer dataset, splitting malignant and benign
#     cases to create two distributions for testing.
#     """
    
#     def __init__(
#         self,
#         root: str = './data',
#         N: int = 200,
#         M: int = 200,
#         download: bool = True,
#         feature_subset: Optional[str] = None,
#         noise_level: float = 0.0,
#         **kwargs
#     ):
#         """
#         Args:
#             feature_subset (str): 'mean', 'se', 'worst', or None for all features
#             noise_level (float): Add Gaussian noise with this std deviation
#         """
#         self.feature_subset = feature_subset
#         self.noise_level = noise_level
#         super().__init__(root, N, M, download, **kwargs)
    
#     def _download(self):
#         # sklearn datasets are built-in, no download needed
#         pass
    
#     def _check_exists(self) -> bool:
#         # Always exists since it's a built-in sklearn dataset
#         return True
    
#     def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Load breast cancer dataset
#         data = load_breast_cancer()
#         X, y = data.data, data.target
        
#         # Select feature subset if specified
#         if self.feature_subset:
#             feature_names = data.feature_names
#             if self.feature_subset == 'mean':
#                 indices = [i for i, name in enumerate(feature_names) if 'mean' in name]
#             elif self.feature_subset == 'se':
#                 indices = [i for i, name in enumerate(feature_names) if 'se' in name]
#             elif self.feature_subset == 'worst':
#                 indices = [i for i, name in enumerate(feature_names) if 'worst' in name]
#             else:
#                 indices = list(range(X.shape[1]))
#             X = X[:, indices]
        
#         # Split by target: 0 = malignant, 1 = benign
#         malignant_data = X[y == 0]
#         benign_data = X[y == 1]
        
#         # Sample N from malignant, M from benign
#         np.random.seed(self.seed)
        
#         if len(malignant_data) < self.N:
#             # Bootstrap if not enough samples
#             malignant_indices = np.random.choice(len(malignant_data), self.N, replace=True)
#         else:
#             malignant_indices = np.random.choice(len(malignant_data), self.N, replace=False)
        
#         if len(benign_data) < self.M:
#             benign_indices = np.random.choice(len(benign_data), self.M, replace=True)
#         else:
#             benign_indices = np.random.choice(len(benign_data), self.M, replace=False)
        
#         X_samples = malignant_data[malignant_indices]
#         Y_samples = benign_data[benign_indices]
        
#         # Add noise if specified
#         if self.noise_level > 0:
#             X_samples += np.random.normal(0, self.noise_level, X_samples.shape)
#             Y_samples += np.random.normal(0, self.noise_level, Y_samples.shape)
        
#         return torch.tensor(X_samples, dtype=torch.float32), torch.tensor(Y_samples, dtype=torch.float32)


# class Diabetes(TabularTSTDataset):
#     """
#     Diabetes dataset: high vs low progression.
    
#     Splits the sklearn diabetes dataset by progression score to create
#     two populations representing different disease progression levels.
#     """
    
#     def __init__(
#         self,
#         root: str = './data',
#         N: int = 150,
#         M: int = 150,
#         download: bool = True,
#         threshold_percentile: float = 50.0,
#         **kwargs
#     ):
#         """
#         Args:
#             threshold_percentile (float): Percentile to split high/low progression
#         """
#         self.threshold_percentile = threshold_percentile
#         super().__init__(root, N, M, download, **kwargs)
    
#     def _download(self):
#         pass
    
#     def _check_exists(self) -> bool:
#         return True
    
#     def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Load diabetes dataset
#         data = load_diabetes()
#         X, y = data.data, data.target
        
#         # Split by progression score
#         threshold = np.percentile(y, self.threshold_percentile)
#         high_progression_mask = y >= threshold
#         low_progression_mask = y < threshold
        
#         high_data = X[high_progression_mask]
#         low_data = X[low_progression_mask]
        
#         # Sample N from high progression, M from low progression
#         np.random.seed(self.seed)
        
#         high_indices = np.random.choice(len(high_data), min(self.N, len(high_data)), replace=False)
#         low_indices = np.random.choice(len(low_data), min(self.M, len(low_data)), replace=False)
        
#         X_samples = high_data[high_indices]
#         Y_samples = low_data[low_indices]
        
#         return torch.tensor(X_samples, dtype=torch.float32), torch.tensor(Y_samples, dtype=torch.float32)


# class Wine(TabularTSTDataset):
#     """
#     Wine dataset: different wine cultivars.
    
#     Uses the sklearn wine dataset, selecting two different cultivars
#     to create distinct distributions.
#     """
    
#     def __init__(
#         self,
#         root: str = './data',
#         N: int = 100,
#         M: int = 100,
#         download: bool = True,
#         cultivar_pair: Tuple[int, int] = (0, 1),
#         normalize_features: bool = True,
#         **kwargs
#     ):
#         """
#         Args:
#             cultivar_pair (Tuple[int, int]): Which two cultivars to compare (0, 1, 2)
#             normalize_features (bool): Whether to normalize features
#         """
#         self.cultivar_pair = cultivar_pair
#         self.normalize_features = normalize_features
#         super().__init__(root, N, M, download, **kwargs)
    
#     def _download(self):
#         pass
    
#     def _check_exists(self) -> bool:
#         return True
    
#     def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Load wine dataset
#         data = load_wine()
#         X, y = data.data, data.target
        
#         # Normalize features if requested
#         if self.normalize_features:
#             X = (X - X.mean(axis=0)) / X.std(axis=0)
        
#         # Select two cultivars
#         cultivar1, cultivar2 = self.cultivar_pair
#         data1 = X[y == cultivar1]
#         data2 = X[y == cultivar2]
        
#         # Sample N from cultivar1, M from cultivar2
#         np.random.seed(self.seed)
        
#         indices1 = np.random.choice(len(data1), min(self.N, len(data1)), replace=len(data1) < self.N)
#         indices2 = np.random.choice(len(data2), min(self.M, len(data2)), replace=len(data2) < self.M)
        
#         X_samples = data1[indices1]
#         Y_samples = data2[indices2]
        
#         return torch.tensor(X_samples, dtype=torch.float32), torch.tensor(Y_samples, dtype=torch.float32)


# class Heart(TabularTSTDataset):
#     """
#     Heart disease dataset: patients with vs without heart disease.
    
#     Downloads and processes the UCI Heart Disease dataset.
#     """
    
#     url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
#     filename = "heart.csv"
    
#     def __init__(
#         self,
#         root: str = './data',
#         N: int = 150,
#         M: int = 150,
#         download: bool = True,
#         missing_strategy: str = 'drop',
#         **kwargs
#     ):
#         """
#         Args:
#             missing_strategy (str): 'drop', 'mean', or 'median' for handling missing values
#         """
#         self.missing_strategy = missing_strategy
#         super().__init__(root, N, M, download, **kwargs)
    
#     def _download(self):
#         filepath = os.path.join(self.root, self.filename)
#         if not os.path.exists(filepath):
#             self._download_url(self.url, filepath)
    
#     def _check_exists(self) -> bool:
#         return os.path.exists(os.path.join(self.root, self.filename))
    
#     def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         filepath = os.path.join(self.root, self.filename)
        
#         # Load data
#         column_names = [
#             'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
#         ]
        
#         df = pd.read_csv(filepath, names=column_names, na_values='?')
        
#         # Handle missing values
#         if self.missing_strategy == 'drop':
#             df = df.dropna()
#         elif self.missing_strategy == 'mean':
#             df = df.fillna(df.mean())
#         elif self.missing_strategy == 'median':
#             df = df.fillna(df.median())
        
#         # Separate features and target
#         X = df.drop('target', axis=1).values
#         y = df['target'].values
        
#         # Binary classification: 0 = no disease, >0 = disease
#         y_binary = (y > 0).astype(int)
        
#         # Split by target
#         no_disease_data = X[y_binary == 0]
#         disease_data = X[y_binary == 1]
        
#         # Sample N from no disease, M from disease
#         np.random.seed(self.seed)
        
#         no_disease_indices = np.random.choice(
#             len(no_disease_data), 
#             min(self.N, len(no_disease_data)), 
#             replace=len(no_disease_data) < self.N
#         )
#         disease_indices = np.random.choice(
#             len(disease_data), 
#             min(self.M, len(disease_data)), 
#             replace=len(disease_data) < self.M
#         )
        
#         X_samples = no_disease_data[no_disease_indices]
#         Y_samples = disease_data[disease_indices]
        
#         return torch.tensor(X_samples, dtype=torch.float32), torch.tensor(Y_samples, dtype=torch.float32)


# class SyntheticTabular(TabularTSTDataset):
#     """
#     Synthetic tabular dataset with controllable differences.
    
#     Generates two multivariate normal distributions with different
#     parameters for controlled experiments.
#     """
    
#     def __init__(
#         self,
#         root: str = './data',
#         N: int = 500,
#         M: int = 500,
#         download: bool = True,
#         n_features: int = 10,
#         mean_shift: float = 0.5,
#         cov_shift: float = 0.1,
#         **kwargs
#     ):
#         """
#         Args:
#             n_features (int): Number of features
#             mean_shift (float): Mean difference between distributions
#             cov_shift (float): Covariance difference between distributions
#         """
#         self.n_features = n_features
#         self.mean_shift = mean_shift
#         self.cov_shift = cov_shift
#         super().__init__(root, N, M, download, **kwargs)
    
#     def _download(self):
#         pass
    
#     def _check_exists(self) -> bool:
#         return True
    
#     def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         np.random.seed(self.seed)
        
#         # Distribution P parameters
#         mean_P = np.zeros(self.n_features)
#         cov_P = np.eye(self.n_features)
        
#         # Distribution Q parameters (shifted)
#         mean_Q = np.ones(self.n_features) * self.mean_shift
#         cov_Q = np.eye(self.n_features) * (1 + self.cov_shift)
        
#         # Generate samples
#         X_samples = np.random.multivariate_normal(mean_P, cov_P, self.N)
#         Y_samples = np.random.multivariate_normal(mean_Q, cov_Q, self.M)
        
#         return torch.tensor(X_samples, dtype=torch.float32), torch.tensor(Y_samples, dtype=torch.float32)


class HiggsBoson(TabularTSTDataset):
    """
    Higgs Boson dataset: signal vs background events.
    
    This is a challenging physics dataset from the ATLAS experiment at CERN.
    The goal is to distinguish Higgs boson signal events from background events
    based on kinematic properties measured by the detector.
    
    This is a very challenging dataset where signal and background have significant overlap.
    
    Citation: "Searching for exotic particles in high-energy physics with deep learning"
    Dataset: https://archive.ics.uci.edu/ml/datasets/HIGGS
    """
    
    # Use smaller Higgs samples from academic/ML repositories
    filename = "higgs_sample.csv"
    
    def __init__(
        self,
        root: str = './data',
        N: int = 50000,
        M: int = 50000,
        download: bool = True,
        use_low_level_features: bool = True,
        **kwargs
    ):
        """
        Args:
            use_low_level_features (bool): If True, use all 28 features. If False, use only high-level features.
        """
        self.use_low_level_features = use_low_level_features
        super().__init__(root, N, M, download, **kwargs)
    
    def _download(self):
        filepath = os.path.join(self.root, self.filename)
        
        if not os.path.exists(filepath):
            print("Downloading first 100k samples from official Higgs dataset...")
            try:
                self._download_higgs_sample(filepath)
                print("Downloaded successfully!")
            except Exception as e:
                raise RuntimeError(f"Could not download Higgs dataset: {e}")
    
    def _download_higgs_sample(self, output_path):
        """Download only a sample of the Higgs dataset to avoid 2.6GB download."""
        import urllib.request
        import gzip
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
        max_samples = 100000  # Only download first 100k samples
        
        print(f"Streaming first {max_samples} samples from Higgs dataset...")
        
        with urllib.request.urlopen(url) as response:
            with gzip.open(response, 'rt') as gz_file:
                with open(output_path, 'w') as out_file:
                    for i, line in enumerate(gz_file):
                        if i >= max_samples:
                            break
                        out_file.write(line)
                        
                        # Show progress
                        if (i + 1) % 10000 == 0:
                            print(f"Downloaded {i + 1} samples...")
        
        print(f"Successfully downloaded {min(max_samples, i + 1)} samples!")
    
    def _check_exists(self) -> bool:
        filepath = os.path.join(self.root, self.filename)
        return os.path.exists(filepath)
    
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        filepath = os.path.join(self.root, self.filename)
        
        # Load data
        data = pd.read_csv(filepath, header=None)
        
        # First column is the label (0 = background, 1 = signal)
        labels = data.iloc[:, 0].values
        features = data.iloc[:, 1:].values
        
        # Select feature subset
        if not self.use_low_level_features:
            # Use only the first 21 high-level features
            features = features[:, :21]
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Split by signal/background
        signal_indices = np.where(labels == 1)[0]
        background_indices = np.where(labels == 0)[0]
        
        # Sample data
        np.random.seed(self.seed)
        
        signal_sample_indices = np.random.choice(
            signal_indices, 
            min(self.N, len(signal_indices)), 
            replace=False
        )
        background_sample_indices = np.random.choice(
            background_indices, 
            min(self.M, len(background_indices)), 
            replace=False
        )
        
        X_samples = features[signal_sample_indices]
        Y_samples = features[background_sample_indices]
        
        # Type-I error check: draw both samples from the same distribution
        if self.t1_check:
            print("Type-I error check: Drawing both X and Y from background distribution")
            # Make sure we have enough background samples for both X and Y
            total_needed = self.N + self.M
            if len(background_indices) >= total_needed:
                combined_indices = np.random.choice(background_indices, total_needed, replace=False)
            else:
                combined_indices = np.random.choice(background_indices, total_needed, replace=True)
            
            X_samples = features[combined_indices[:self.N]]
            Y_samples = features[combined_indices[self.N:self.N+self.M]]
        
        return torch.tensor(X_samples, dtype=torch.float32), torch.tensor(Y_samples, dtype=torch.float32)


class HDGM(TabularTSTDataset):
    """
    Hierarchical Dirichlet Gaussian Mixture (HDGM) synthetic dataset.
    
    This synthetic dataset generates samples from a hierarchical mixture of Gaussians
    with controllable difficulty levels and cluster structures.
    """
    
    def __init__(
        self,
        root: str = './data',
        N: int = 500,
        M: int = 500,
        download: bool = True,
        d: int = 10,
        n_clusters: int = 2,
        level: str = "hard",
        kk: int = 0,
        **kwargs
    ):
        """
        Args:
            d (int): Dimensionality of the data
            n_clusters (int): Number of clusters in the mixture
            level (str): Difficulty level - 'hard', 'medium', or other
            kk (int): Random seed modifier
        """
        self.d = d
        self.n_clusters = n_clusters
        self.level = level
        self.kk = kk
        super().__init__(root, N, M, download, **kwargs)
    
    def _download(self):
        # Synthetic data, no download needed
        pass
    
    def _check_exists(self) -> bool:
        # Always exists since it's synthetic
        return True
    
    def _generate_hdgm_cov_matrix(self, n_clusters, d, cluster_gap):
        """Generate covariance matrices and means for HDGM - exact copy from your implementation."""
        mu_mx = np.zeros([n_clusters, d])
        for i in range(n_clusters):
            mu_mx[i] = mu_mx[i] + cluster_gap*i
        sigma_mx_1 = np.eye(d)
        sigma_mx_2 = [np.eye(d), np.eye(d)]
        sigma_mx_2[0][0, 1] = 0.5
        sigma_mx_2[0][1, 0] = 0.5
        sigma_mx_2[1][0, 1] = -0.5
        sigma_mx_2[1][1, 0] = -0.5
        
        return mu_mx, sigma_mx_1, sigma_mx_2
    
    def _sample_hdgm(self, N, M, d, n_clusters, kk, level, t1_check):
        """Generate HDGM samples - exact copy from your implementation."""
        if level == "hard":
            mu_mx_1, sigma_mx_1, sigma_mx_2 = self._generate_hdgm_cov_matrix(n_clusters, d, 0.5)
            mu_mx_2 = mu_mx_1
        elif level == "medium":
            mu_mx_1, sigma_mx_1, sigma_mx_2 = self._generate_hdgm_cov_matrix(n_clusters, d, 10)
            mu_mx_2 = mu_mx_1
        else:
            mu_mx_1, sigma_mx_1, sigma_mx_2 = self._generate_hdgm_cov_matrix(n_clusters, d, 10)
            mu_mx_2 = mu_mx_1 - 1.5

        P = np.zeros([N*n_clusters, d])
        Q = np.zeros([M*n_clusters, d])

        if t1_check:
            for i in range(n_clusters):
                np.random.seed(seed=1102*kk + i + N + M)
                P[i*N:(i+1)*N, :] = np.random.multivariate_normal(mu_mx_1[i], sigma_mx_1, N)
                np.random.seed(seed=819*kk + i + N + M)
                Q[i*M:(i+1)*M, :] = np.random.multivariate_normal(mu_mx_1[i], sigma_mx_1, M)
        else:
            for i in range(n_clusters):
                np.random.seed(seed=1102*kk + i + N + M)
                P[i*N:(i+1)*N, :] = np.random.multivariate_normal(mu_mx_1[i], sigma_mx_1, N)
                np.random.seed(seed=819*kk + i + N + M)
                Q[i*M:(i+1)*M, :] = np.random.multivariate_normal(mu_mx_2[i], sigma_mx_2[i], M)

        idx_P = np.random.choice(len(P), N, replace=False)
        idx_Q = np.random.choice(len(Q), M, replace=False)
        return P[idx_P], Q[idx_Q]
    
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate HDGM samples using your exact implementation
        P, Q = self._sample_hdgm(self.N, self.M, self.d, self.n_clusters, self.kk, self.level, self.t1_check)
        
        return torch.tensor(P, dtype=torch.float32), torch.tensor(Q, dtype=torch.float32)


# class AdultIncome(TabularTSTDataset):
#     """
#     Adult Income dataset with challenging distribution shifts.
    
#     Uses the Adult Census Income dataset but creates challenging comparisons
#     by splitting on demographic features that create subtle distribution differences.
    
#     This creates realistic distribution shifts that are common in real-world ML.
#     """
    
#     url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
#     filename_train = "adult.data"
    
#     def __init__(
#         self,
#         root: str = './data',
#         N: int = 5000,
#         M: int = 5000,
#         download: bool = True,
#         shift_type: str = 'age_groups',
#         **kwargs
#     ):
#         """
#         Args:
#             shift_type (str): Type of shift ('age_groups', 'education_levels', 'work_hours')
#         """
#         self.shift_type = shift_type
#         super().__init__(root, N, M, download, **kwargs)
    
#     def _download(self):
#         train_path = os.path.join(self.root, self.filename_train)
        
#         if not os.path.exists(train_path):
#             print("Downloading Adult Income dataset...")
#             self._download_url(self.url_train, train_path)
    
#     def _check_exists(self) -> bool:
#         return os.path.exists(os.path.join(self.root, self.filename_train))
    
#     def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         train_path = os.path.join(self.root, self.filename_train)
        
#         # Column names for Adult dataset
#         columns = [
#             'age', 'workclass', 'fnlwgt', 'education', 'education-num',
#             'marital-status', 'occupation', 'relationship', 'race', 'sex',
#             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
#         ]
        
#         # Load data
#         df = pd.read_csv(train_path, names=columns, na_values=' ?', skipinitialspace=True)
#         df = df.dropna()  # Remove missing values
        
#         # Select numerical features for simplicity
#         numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
#                              'capital-loss', 'hours-per-week']
        
#         X = df[numerical_features].values
        
#         # Normalize features
#         X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
#         # Create challenging splits based on shift_type
#         if self.shift_type == 'age_groups':
#             # Split by age groups - creates subtle distribution differences
#             young_mask = df['age'] <= 35
#             old_mask = df['age'] > 50
            
#             X_young = X[young_mask]
#             X_old = X[old_mask]
            
#         elif self.shift_type == 'education_levels':
#             # Split by education level
#             low_edu_mask = df['education-num'] <= 9
#             high_edu_mask = df['education-num'] >= 13
            
#             X_young = X[low_edu_mask]  # Using same variable names
#             X_old = X[high_edu_mask]
            
#         elif self.shift_type == 'work_hours':
#             # Split by work hours
#             part_time_mask = df['hours-per-week'] <= 35
#             full_time_mask = df['hours-per-week'] >= 40
            
#             X_young = X[part_time_mask]
#             X_old = X[full_time_mask]
        
#         else:
#             raise ValueError(f"Unknown shift_type: {self.shift_type}")
        
#         # Sample data
#         np.random.seed(self.seed)
        
#         young_indices = np.random.choice(
#             len(X_young), 
#             min(self.N, len(X_young)), 
#             replace=len(X_young) < self.N
#         )
#         old_indices = np.random.choice(
#             len(X_old), 
#             min(self.M, len(X_old)), 
#             replace=len(X_old) < self.M
#         )
        
#         X_samples = X_young[young_indices]
#         Y_samples = X_old[old_indices]
        
#         return torch.tensor(X_samples, dtype=torch.float32), torch.tensor(Y_samples, dtype=torch.float32)


# class ChallengingSynthetic(TabularTSTDataset):
#     """
#     Challenging synthetic dataset with subtle distribution differences.
    
#     Creates synthetic data where the two distributions have very subtle differences,
#     making them difficult to distinguish even for sophisticated models.
#     """
    
#     def __init__(
#         self,
#         root: str = './data',
#         N: int = 5000,
#         M: int = 5000,
#         download: bool = True,
#         n_features: int = 20,
#         difficulty: str = 'hard',
#         **kwargs
#     ):
#         """
#         Args:
#             n_features (int): Number of features
#             difficulty (str): 'medium', 'hard', 'very_hard'
#         """
#         self.n_features = n_features
#         self.difficulty = difficulty
#         super().__init__(root, N, M, download, **kwargs)
    
#     def _download(self):
#         pass
    
#     def _check_exists(self) -> bool:
#         return True
    
#     def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         np.random.seed(self.seed)
        
#         if self.difficulty == 'medium':
#             shift_magnitude = 0.3
#             cov_change = 0.1
#         elif self.difficulty == 'hard':
#             shift_magnitude = 0.15
#             cov_change = 0.05
#         else:  # very_hard
#             shift_magnitude = 0.08
#             cov_change = 0.02
        
#         # Distribution P: base distribution
#         mean_P = np.zeros(self.n_features)
#         cov_P = np.eye(self.n_features)
        
#         # Add some correlation structure to make it more realistic
#         for i in range(self.n_features - 1):
#             cov_P[i, i+1] = 0.3
#             cov_P[i+1, i] = 0.3
        
#         # Distribution Q: subtle shift
#         # Only shift a few dimensions to make it more challenging
#         n_shift_dims = max(1, self.n_features // 4)
#         shift_dims = np.random.choice(self.n_features, n_shift_dims, replace=False)
        
#         mean_Q = mean_P.copy()
#         mean_Q[shift_dims] += shift_magnitude
        
#         # Subtle covariance change
#         cov_Q = cov_P * (1 + np.random.normal(0, cov_change, cov_P.shape))
#         cov_Q = (cov_Q + cov_Q.T) / 2  # Make symmetric
#         cov_Q += np.eye(self.n_features) * 0.01  # Ensure positive definite
        
#         # Add high-dimensional noise to make it even harder
#         noise_dims = self.n_features // 2
#         noise_mean = np.random.normal(0, 0.1, noise_dims)
#         noise_cov = np.eye(noise_dims) * 0.5
        
#         # Generate samples
#         samples_P = np.random.multivariate_normal(mean_P, cov_P, self.N)
#         samples_Q = np.random.multivariate_normal(mean_Q, cov_Q, self.M)
        
#         # Add noise dimensions
#         noise_P = np.random.multivariate_normal(noise_mean, noise_cov, self.N)
#         noise_Q = np.random.multivariate_normal(noise_mean, noise_cov, self.M)
        
#         # Concatenate signal and noise
#         X_samples = np.concatenate([samples_P, noise_P], axis=1)
#         Y_samples = np.concatenate([samples_Q, noise_Q], axis=1)
        
#         # Type-I error check: draw both samples from the same distribution
#         if self.t1_check:
#             print("Type-I error check: Drawing both X and Y from distribution P")
#             # Generate additional samples from distribution P for Y
#             additional_samples_P = np.random.multivariate_normal(mean_P, cov_P, self.M)
#             additional_noise_P = np.random.multivariate_normal(noise_mean, noise_cov, self.M)
#             Y_samples = np.concatenate([additional_samples_P, additional_noise_P], axis=1)
        
#         return torch.tensor(X_samples, dtype=torch.float32), torch.tensor(Y_samples, dtype=torch.float32)