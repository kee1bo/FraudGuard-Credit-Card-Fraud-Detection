"""
ULB Dataset Analysis Pipeline for Feature Mapping
Analyzes the ULB dataset to extract patterns and create training data for feature mapping models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

from fraudguard.entity.feature_mapping_entity import (
    TrainingDataPoint, MerchantCategory, LocationRisk, SpendingPattern, TimeContext
)
from fraudguard.logger import fraud_logger


class ULBDatasetAnalyzer:
    """Analyzes ULB dataset to extract interpretable features and patterns"""
    
    def __init__(self, dataset_path: str = "data/creditcard.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.feature_stats = {}
        self.correlation_matrix = None
        self.merchant_clusters = None
        
    def load_dataset(self) -> pd.DataFrame:
        """Load the ULB dataset"""
        try:
            fraud_logger.info(f"Loading ULB dataset from {self.dataset_path}")
            self.df = pd.read_csv(self.dataset_path)
            fraud_logger.info(f"Dataset loaded: {len(self.df)} transactions, {len(self.df.columns)} features")
            return self.df
        except Exception as e:
            fraud_logger.error(f"Error loading dataset: {e}")
            raise
    
    def analyze_feature_distributions(self) -> Dict[str, Dict]:
        """Analyze statistical distributions of all features"""
        if self.df is None:
            self.load_dataset()
            
        fraud_logger.info("Analyzing feature distributions...")
        
        self.feature_stats = {}
        
        # Analyze each feature
        for col in self.df.columns:
            if col != 'Class':  # Skip target variable
                stats = {
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'median': self.df[col].median(),
                    'q25': self.df[col].quantile(0.25),
                    'q75': self.df[col].quantile(0.75)
                }
                
                # Separate stats for fraud vs normal transactions
                fraud_mask = self.df['Class'] == 1
                normal_mask = self.df['Class'] == 0
                
                stats['fraud_mean'] = self.df.loc[fraud_mask, col].mean()
                stats['fraud_std'] = self.df.loc[fraud_mask, col].std()
                stats['normal_mean'] = self.df.loc[normal_mask, col].mean()
                stats['normal_std'] = self.df.loc[normal_mask, col].std()
                
                self.feature_stats[col] = stats
        
        fraud_logger.info("Feature distribution analysis completed")
        return self.feature_stats
    
    def compute_correlation_matrix(self) -> np.ndarray:
        """Compute correlation matrix between all features"""
        if self.df is None:
            self.load_dataset()
            
        fraud_logger.info("Computing correlation matrix...")
        
        # Select only numerical features (exclude Class)
        numerical_features = [col for col in self.df.columns if col != 'Class']
        self.correlation_matrix = self.df[numerical_features].corr().values
        
        fraud_logger.info("Correlation matrix computed")
        return self.correlation_matrix
    
    def create_merchant_categories(self, n_clusters: int = 10) -> Dict[int, MerchantCategory]:
        """Create merchant categories based on transaction patterns"""
        if self.df is None:
            self.load_dataset()
            
        fraud_logger.info(f"Creating merchant categories using {n_clusters} clusters...")
        
        # Use Amount, Time, and some V features to cluster transactions
        clustering_features = ['Amount', 'Time', 'V1', 'V2', 'V3', 'V4', 'V5']
        X_cluster = self.df[clustering_features].values
        
        # Standardize features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters to assign merchant categories
        cluster_to_merchant = {}
        merchant_categories = list(MerchantCategory)
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = self.df[cluster_mask]
            
            # Analyze cluster characteristics
            avg_amount = cluster_data['Amount'].mean()
            fraud_rate = cluster_data['Class'].mean()
            
            # Assign merchant category based on patterns
            if avg_amount < 50:
                if fraud_rate > 0.005:
                    category = MerchantCategory.ATM_WITHDRAWAL
                else:
                    category = MerchantCategory.GROCERY
            elif avg_amount < 200:
                if fraud_rate > 0.003:
                    category = MerchantCategory.GAS_STATION
                else:
                    category = MerchantCategory.RESTAURANT
            elif avg_amount < 500:
                category = MerchantCategory.DEPARTMENT_STORE
            elif avg_amount < 1000:
                category = MerchantCategory.ONLINE_RETAIL
            else:
                if fraud_rate > 0.01:
                    category = MerchantCategory.TRAVEL
                else:
                    category = MerchantCategory.ENTERTAINMENT
            
            cluster_to_merchant[cluster_id] = category
        
        self.merchant_clusters = {
            'kmeans_model': kmeans,
            'scaler': scaler,
            'cluster_to_merchant': cluster_to_merchant
        }
        
        fraud_logger.info("Merchant categories created")
        return cluster_to_merchant
    
    def derive_interpretable_features(self, transaction_row: pd.Series) -> Dict:
        """Derive interpretable features from a ULB transaction"""
        
        # Time context
        time_seconds = transaction_row['Time']
        hour_of_day = int((time_seconds / 3600) % 24)
        day_of_week = int((time_seconds / 86400) % 7)
        is_weekend = day_of_week >= 5
        
        time_context = TimeContext(
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            is_weekend=is_weekend
        )
        
        # Merchant category (use clustering if available)
        if self.merchant_clusters is not None:
            clustering_features = ['Amount', 'Time', 'V1', 'V2', 'V3', 'V4', 'V5']
            feature_values = transaction_row[clustering_features].values.reshape(1, -1)
            scaled_features = self.merchant_clusters['scaler'].transform(feature_values)
            cluster_id = self.merchant_clusters['kmeans_model'].predict(scaled_features)[0]
            merchant_category = self.merchant_clusters['cluster_to_merchant'][cluster_id]
        else:
            # Default categorization based on amount
            amount = transaction_row['Amount']
            if amount < 50:
                merchant_category = MerchantCategory.GROCERY
            elif amount < 200:
                merchant_category = MerchantCategory.RESTAURANT
            elif amount < 500:
                merchant_category = MerchantCategory.DEPARTMENT_STORE
            else:
                merchant_category = MerchantCategory.ONLINE_RETAIL
        
        # Location risk (based on V features that might indicate location)
        location_features = [transaction_row[f'V{i}'] for i in [1, 2, 3, 14]]
        location_score = np.mean(np.abs(location_features))
        
        if location_score < 0.5:
            location_risk = LocationRisk.NORMAL
        elif location_score < 1.0:
            location_risk = LocationRisk.SLIGHTLY_UNUSUAL
        elif location_score < 2.0:
            location_risk = LocationRisk.HIGHLY_UNUSUAL
        else:
            location_risk = LocationRisk.FOREIGN_COUNTRY
        
        # Spending pattern (based on amount and V features)
        amount = transaction_row['Amount']
        spending_features = [transaction_row[f'V{i}'] for i in [4, 11, 12]]
        spending_score = np.mean(np.abs(spending_features)) + np.log1p(amount) / 10
        
        if spending_score < 1.0:
            spending_pattern = SpendingPattern.TYPICAL
        elif spending_score < 2.0:
            spending_pattern = SpendingPattern.SLIGHTLY_HIGHER
        elif spending_score < 3.0:
            spending_pattern = SpendingPattern.MUCH_HIGHER
        else:
            spending_pattern = SpendingPattern.SUSPICIOUS
        
        return {
            'merchant_category': merchant_category,
            'time_context': time_context,
            'location_risk': location_risk,
            'spending_pattern': spending_pattern
        }
    
    def create_training_dataset(self, sample_size: Optional[int] = None) -> List[TrainingDataPoint]:
        """Create training dataset for feature mapping models"""
        if self.df is None:
            self.load_dataset()
            
        fraud_logger.info("Creating training dataset for feature mapping...")
        
        # Create merchant categories if not done
        if self.merchant_clusters is None:
            self.create_merchant_categories()
        
        # Sample data if requested
        df_sample = self.df.sample(n=sample_size, random_state=42) if sample_size else self.df
        
        training_data = []
        
        for idx, row in df_sample.iterrows():
            # Extract original ULB features
            time = row['Time']
            amount = row['Amount']
            v1_to_v28 = np.array([row[f'V{i}'] for i in range(1, 29)])
            class_label = row['Class']
            
            # Derive interpretable features
            interpretable = self.derive_interpretable_features(row)
            
            # Encode categorical features
            merchant_categories = list(MerchantCategory)
            merchant_encoded = merchant_categories.index(interpretable['merchant_category'])
            
            location_risks = list(LocationRisk)
            location_encoded = location_risks.index(interpretable['location_risk'])
            
            spending_patterns = list(SpendingPattern)
            spending_encoded = spending_patterns.index(interpretable['spending_pattern'])
            
            # Time context features
            time_ctx = interpretable['time_context']
            hour_sin = np.sin(2 * np.pi * time_ctx.hour_of_day / 24)
            hour_cos = np.cos(2 * np.pi * time_ctx.hour_of_day / 24)
            time_features = np.array([hour_sin, hour_cos, time_ctx.day_of_week, int(time_ctx.is_weekend)])
            
            # Create training data point
            training_point = TrainingDataPoint(
                time=time,
                amount=amount,
                v1_to_v28=v1_to_v28,
                class_label=class_label,
                merchant_category_encoded=merchant_encoded,
                time_context_features=time_features,
                location_risk_score=location_encoded,
                spending_pattern_score=spending_encoded
            )
            
            training_data.append(training_point)
        
        fraud_logger.info(f"Training dataset created with {len(training_data)} samples")
        return training_data
    
    def save_analysis_results(self, output_dir: str = "artifacts/feature_mapping"):
        """Save analysis results for later use"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature statistics
        if self.feature_stats:
            joblib.dump(self.feature_stats, output_path / "feature_stats.pkl")
        
        # Save correlation matrix
        if self.correlation_matrix is not None:
            np.save(output_path / "correlation_matrix.npy", self.correlation_matrix)
        
        # Save merchant clustering results
        if self.merchant_clusters:
            joblib.dump(self.merchant_clusters, output_path / "merchant_clusters.pkl")
        
        fraud_logger.info(f"Analysis results saved to {output_path}")
    
    def load_analysis_results(self, input_dir: str = "artifacts/feature_mapping"):
        """Load previously saved analysis results"""
        input_path = Path(input_dir)
        
        # Load feature statistics
        stats_path = input_path / "feature_stats.pkl"
        if stats_path.exists():
            self.feature_stats = joblib.load(stats_path)
        
        # Load correlation matrix
        corr_path = input_path / "correlation_matrix.npy"
        if corr_path.exists():
            self.correlation_matrix = np.load(corr_path)
        
        # Load merchant clustering results
        clusters_path = input_path / "merchant_clusters.pkl"
        if clusters_path.exists():
            self.merchant_clusters = joblib.load(clusters_path)
        
        fraud_logger.info(f"Analysis results loaded from {input_path}")


def analyze_ulb_dataset(dataset_path: str = "data/creditcard.csv", 
                       output_dir: str = "artifacts/feature_mapping",
                       sample_size: Optional[int] = 10000) -> ULBDatasetAnalyzer:
    """
    Convenience function to perform complete ULB dataset analysis
    
    Args:
        dataset_path: Path to creditcard.csv file
        output_dir: Directory to save analysis results
        sample_size: Number of samples for training data (None for full dataset)
    
    Returns:
        ULBDatasetAnalyzer instance with completed analysis
    """
    analyzer = ULBDatasetAnalyzer(dataset_path)
    
    # Perform analysis
    analyzer.analyze_feature_distributions()
    analyzer.compute_correlation_matrix()
    analyzer.create_merchant_categories()
    
    # Create and save training dataset
    training_data = analyzer.create_training_dataset(sample_size)
    
    # Save results
    analyzer.save_analysis_results(output_dir)
    
    # Also save training data
    output_path = Path(output_dir)
    joblib.dump(training_data, output_path / "training_data.pkl")
    
    return analyzer