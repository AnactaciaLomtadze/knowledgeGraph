#src/enhanced_evaluation_metrics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import logging
import datetime
import math
import os
from collections import defaultdict
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import json

class EnhancedEvaluationMetrics:
    """
    Enhanced evaluation metrics for assessing the performance of forgetting mechanisms.
    
    This class provides a comprehensive set of metrics for evaluating how
    forgetting mechanisms affect recommendation quality, diversity, and other
    important aspects of the system, with improved summary capabilities.
    """
    def __init__(self, knowledge_graph, forgetting_mechanism=None, output_dir='./results'):
        """
        Initialize the enhanced evaluation metrics.
        
        Args:
            knowledge_graph: The MovieLensKnowledgeGraph instance
            forgetting_mechanism: Optional ForgettingMechanism instance
            output_dir: Directory to save results and visualizations
        """
        self.kg = knowledge_graph
        self.fm = forgetting_mechanism
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('EnhancedEvaluationMetrics')
        
        # Define metrics categories
        self.accuracy_metrics = ['hit_rate', 'precision', 'recall', 'ndcg', 'mrr']
        self.beyond_accuracy_metrics = ['serendipity', 'novelty', 'diversity', 'coverage']
        self.forgetting_impact_metrics = ['jaccard_similarity', 'new_item_percentage', 'memory_strength']
        self.efficiency_metrics = ['inference_time', 'memory_usage']
        
        # Baseline performance storage
        self.baseline_performances = {}
    
    def calculate_hit_rate_at_k(self, test_set, recommendations, k=10):
        """
        Calculate Hit Rate@K, which measures if a relevant item appears in top-K recommendations.
        
        Args:
            test_set: Set of movie IDs that are relevant
            recommendations: List of recommended movie IDs
            k: Number of top recommendations to consider
            
        Returns:
            1 if at least one item in the test set appears in top-k recommendations, 0 otherwise
        """
        if not test_set or not recommendations:
            return 0.0
        
        top_k_recommendations = recommendations[:k]
        
        for item in test_set:
            if item in top_k_recommendations:
                return 1.0
        
        return 0.0
    
    def calculate_precision_at_k(self, test_set, recommendations, k=10):
        """
        Calculate Precision@K, which is the proportion of recommended items that are relevant.
        
        Args:
            test_set: Set of movie IDs that are relevant
            recommendations: List of recommended movie IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score between 0 and 1
        """
        if not test_set or not recommendations:
            return 0.0
        
        top_k_recommendations = recommendations[:k]
        
        relevant_items = [item for item in top_k_recommendations if item in test_set]
        
        return len(relevant_items) / min(k, len(top_k_recommendations))

    def calculate_recall_at_k(self, test_set, recommendations, k=10):
        """
        Calculate Recall@K, which is the proportion of relevant items that are recommended.

        Args:
            test_set: Set of movie IDs that are relevant
            recommendations: List of recommended movie IDs
            k: Number of top recommendations to consider

        Returns:
            Recall@K score between 0 and 1
        """
        if not test_set or not recommendations:
            return 0.0

        top_k_recommendations = recommendations[:k]
        relevant_items = [item for item in top_k_recommendations if item in test_set]

        return len(relevant_items) / len(test_set) if len(test_set) > 0 else 0.0

    def calculate_ndcg_at_k(self, test_set, recommendations, k=10):
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) at K.
        
        Args:
            test_set: Set of movie IDs that are relevant
            recommendations: List of recommended movie IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score between 0 and 1
        """
        if not test_set or not recommendations:
            return 0.0
        
        top_k_recommendations = recommendations[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recommendations):
            if item in test_set:
                dcg += 1.0 / math.log2(i + 2)
        
        # Calculate ideal DCG (IDCG)
        idcg = 0.0
        for i in range(min(len(test_set), k)):
            idcg += 1.0 / math.log2(i + 2)
        
        # If IDCG is 0, return 0
        if idcg == 0.0:
            return 0.0
        
        # Normalize DCG
        ndcg = dcg / idcg
        
        return ndcg

    def calculate_mrr(self, test_set, recommendations):
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            test_set: Set of movie IDs that are relevant
            recommendations: List of recommended movie IDs

        Returns:
            MRR score between 0 and 1
        """
        if not test_set or not recommendations:
            return 0.0

        for i, item in enumerate(recommendations):
            if item in test_set:
                return 1.0 / (i + 1)

        return 0.0
    
    def calculate_serendipity(self, user_id, recommendations, k=10, popularity_threshold=0.8):
        """
        Calculate serendipity of recommendations.
        
        Args:
            user_id: The user ID
            recommendations: List of recommended movie IDs
            k: Number of recommendations to consider
            popularity_threshold: Threshold to determine popular items
            
        Returns:
            Serendipity score between 0 and 1
        """
        if not recommendations:
            return 0.0
            
        top_k_recommendations = recommendations[:k]
        
        # Calculate popularity of all movies
        movie_popularity = {}
        for movie_id in self.kg.movie_features:
            movie_popularity[movie_id] = len(self.kg.ratings_df[self.kg.ratings_df['movie_id'] == movie_id])
        
        # Calculate popularity threshold
        popularity_values = list(movie_popularity.values())
        popularity_cutoff = np.percentile(popularity_values, popularity_threshold * 100)
        
        # Get user's preferred genres
        if user_id in self.kg.user_profiles:
            user_genres = self.kg.user_profiles[user_id]['genre_preferences']
        else:
            user_genres = np.ones(19) / 19
        
        serendipity_scores = []
        
        for movie_id in top_k_recommendations:
            # Calculate unexpectedness based on popularity
            if movie_id in movie_popularity:
                popularity = movie_popularity[movie_id]
                unexpectedness = 1.0 - min(1.0, popularity / popularity_cutoff)
            else:
                unexpectedness = 0.5
            
            # Calculate relevance based on genre match
            if movie_id in self.kg.movie_features:
                movie_genres = self.kg.movie_features[movie_id]
                norm_user_genres = user_genres - np.min(user_genres)
                if np.max(norm_user_genres) > 0:
                    norm_user_genres = norm_user_genres / np.max(norm_user_genres)
                relevance = np.sum(norm_user_genres * movie_genres) / (np.sum(movie_genres) + 1e-10)
            else:
                relevance = 0.0
            
            # Serendipity is a product of unexpectedness and relevance
            serendipity = unexpectedness * relevance
            serendipity_scores.append(serendipity)
        
        if not serendipity_scores:
            return 0.0
            
        return np.mean(serendipity_scores)
    
    def calculate_novelty(self, recommendations, k=10):
        """
        Calculate novelty of recommendations.
        
        Args:
            recommendations: List of recommended movie IDs
            k: Number of recommendations to consider
            
        Returns:
            Novelty score between 0 and 1
        """
        if not recommendations:
            return 0.0
            
        top_k_recommendations = recommendations[:k]
        
        # Calculate self-information of all movies based on their popularity
        total_ratings = len(self.kg.ratings_df)
        self_information = {}
        
        for movie_id in self.kg.movie_features:
            movie_count = len(self.kg.ratings_df[self.kg.ratings_df['movie_id'] == movie_id])
            # Calculate probability of movie being rated
            prob = movie_count / total_ratings
            # Self-information: -log(prob)
            self_information[movie_id] = -np.log2(prob) if prob > 0 else 0
        
        # Normalize self-information to [0,1]
        max_info = max(self_information.values()) if self_information else 1
        norm_self_information = {mid: info/max_info for mid, info in self_information.items()}
        
        # Calculate novelty as average self-information of recommended items
        novelty_scores = []
        for movie_id in top_k_recommendations:
            if movie_id in norm_self_information:
                novelty_scores.append(norm_self_information[movie_id])
            else:
                novelty_scores.append(0.5)
        
        if not novelty_scores:
            return 0.0
            
        return np.mean(novelty_scores)
    
    def calculate_diversity(self, recommendations, k=10):
        """
        Calculate diversity of recommendations.
        
        Args:
            recommendations: List of recommended movie IDs
            k: Number of recommendations to consider
            
        Returns:
            Diversity score between 0 and 1
        """
        if not recommendations:
            return 0.0
            
        top_k_recommendations = recommendations[:k]
        
        # Get genre vectors for recommended movies
        genre_vectors = []
        for movie_id in top_k_recommendations:
            if movie_id in self.kg.movie_features:
                genre_vectors.append(self.kg.movie_features[movie_id])
        
        if len(genre_vectors) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        sim_matrix = cosine_similarity(genre_vectors)
        np.fill_diagonal(sim_matrix, 0)  # Exclude self-similarities
        
        # Diversity is the inverse of average similarity
        diversity = 1.0 - np.mean(sim_matrix)
        
        return diversity
    
    def calculate_coverage(self, recommendations_by_user, all_items=None):
        """
        Calculate catalog coverage.
        
        Args:
            recommendations_by_user: Dict mapping user_id to list of recommendations
            all_items: Set of all available items
            
        Returns:
            Coverage percentage between 0 and 100
        """
        if not recommendations_by_user:
            return 0.0
            
        # Get all recommended items across all users
        recommended_items = set()
        for recs in recommendations_by_user.values():
            recommended_items.update(recs)
        
        # Get all available items if not provided
        if all_items is None:
            all_items = set(self.kg.movie_features.keys())
        
        if not all_items:
            return 0.0
            
        return 100.0 * len(recommended_items) / len(all_items)
    
    def calculate_jaccard_similarity(self, recommendations1, recommendations2):
        """
        Calculate Jaccard similarity between two sets of recommendations.
        
        Args:
            recommendations1: First list of movie IDs
            recommendations2: Second list of movie IDs
            
        Returns:
            Jaccard similarity score between 0 and 1
        """
        if not recommendations1 or not recommendations2:
            return 0.0
            
        set1 = set(recommendations1)
        set2 = set(recommendations2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_rank_correlation(self, recommendations1, recommendations2, k=10):
        """
        Calculate rank correlation between two ranked lists.
        
        Args:
            recommendations1: First list of movie IDs
            recommendations2: Second list of movie IDs
            k: Number of top recommendations to consider
            
        Returns:
            Spearman's rank correlation coefficient between -1 and 1
        """
        if not recommendations1 or not recommendations2:
            return 0.0
            
        # Consider only top-k recommendations
        top_k_rec1 = recommendations1[:k]
        top_k_rec2 = recommendations2[:k]
        
        # Create rankings dictionary for each list
        ranking1 = {item: i+1 for i, item in enumerate(top_k_rec1)}
        ranking2 = {item: i+1 for i, item in enumerate(top_k_rec2)}
        
        # Get common items
        common_items = set(ranking1.keys()).intersection(set(ranking2.keys()))
        
        if len(common_items) < 2:
            return 0.0
        
        # Calculate Spearman's rank correlation
        n = len(common_items)
        sum_d_squared = 0
        
        for item in common_items:
            d = ranking1[item] - ranking2[item]
            sum_d_squared += d * d
        
        return 1 - (6 * sum_d_squared) / (n * (n * n - 1))
    
    def evaluate_all_metrics(self, user_id, test_set, recommendations, k_values=[5, 10, 20]):
        """
        Evaluate all metrics for a single user and recommendation list.
        
        Args:
            user_id: The user ID
            test_set: Set of movie IDs that are relevant
            recommendations: List of recommended movie IDs
            k_values: List of k values to calculate metrics for
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Calculate accuracy metrics for each k value
        for k in k_values:
            metrics[f'hit_rate@{k}'] = self.calculate_hit_rate_at_k(test_set, recommendations, k)
            metrics[f'precision@{k}'] = self.calculate_precision_at_k(test_set, recommendations, k)
            metrics[f'recall@{k}'] = self.calculate_recall_at_k(test_set, recommendations, k)
            metrics[f'ndcg@{k}'] = self.calculate_ndcg_at_k(test_set, recommendations, k)
        
        # Calculate other metrics
        metrics['mrr'] = self.calculate_mrr(test_set, recommendations)
        metrics['serendipity'] = self.calculate_serendipity(user_id, recommendations)
        metrics['novelty'] = self.calculate_novelty(recommendations)
        metrics['diversity'] = self.calculate_diversity(recommendations)
        
        return metrics
    
    
    def compare_strategies(self, user_ids, test_data, recommendation_strategies, k_values=[5, 10, 20]):
        """
        Compare multiple recommendation strategies.
        
        Args:
            user_ids: List of user IDs to evaluate
            test_data: Dictionary mapping user_id to set of relevant movie IDs
            recommendation_strategies: Dict mapping strategy name to recommendation function
            k_values: List of k values to calculate metrics for
            
        Returns:
            DataFrame with evaluation metrics for all strategies
        """
        all_results = []
        
        # Initialize progress bar
        total_evaluations = len(user_ids) * len(recommendation_strategies)
        progress_bar = tqdm(total=total_evaluations, desc="Evaluating strategies")
        
        # Track coverage data
        all_recs_by_strategy = defaultdict(dict)
        
        # Evaluate each strategy on each user
        for user_id in user_ids:
            if user_id not in test_data or not test_data[user_id]:
                # Skip users without test data
                progress_bar.update(len(recommendation_strategies))
                continue
            
            test_set = test_data[user_id]
            
            for strategy_name, recommendation_fn in recommendation_strategies.items():
                try:
                    # Get recommendations
                    start_time = datetime.datetime.now()
                    recommendations = recommendation_fn(user_id)
                    end_time = datetime.datetime.now()
                    inference_time = (end_time - start_time).total_seconds()
                    
                    # Store recommendations for coverage calculation
                    all_recs_by_strategy[strategy_name][user_id] = recommendations
                    
                    # Evaluate all metrics
                    metrics = self.evaluate_all_metrics(user_id, test_set, recommendations, k_values)
                    
                    # Add metadata
                    result = {
                        'user_id': user_id,
                        'strategy': strategy_name,
                        'inference_time': inference_time,
                        **metrics
                    }
                    
                    all_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating {strategy_name} for user {user_id}: {e}")
                
                # Update progress bar
                progress_bar.update(1)
        
        progress_bar.close()
        
        # Calculate coverage for each strategy
        all_items = set(self.kg.movie_features.keys())
        for strategy_name, recs_by_user in all_recs_by_strategy.items():
            coverage = self.calculate_coverage(recs_by_user, all_items)
            
            # Add coverage to all results for this strategy
            for result in all_results:
                if result['strategy'] == strategy_name:
                    result['coverage'] = coverage
        
        return pd.DataFrame(all_results)
    
    def compare_forgetting_mechanisms(self, user_ids, test_data, baseline_recommender, 
                                     forgetting_mechanisms, k_values=[5, 10, 20]):
        """
        Compare multiple forgetting mechanisms.
        
        Args:
            user_ids: List of user IDs to evaluate
            test_data: Dictionary mapping user_id to set of relevant movie IDs
            baseline_recommender: Function that returns baseline recommendations for a user
            forgetting_mechanisms: Dict mapping mechanism name to forgetting function
            k_values: List of k values to calculate metrics for
            
        Returns:
            DataFrame with evaluation metrics for all mechanisms
        """
        all_results = []
        
        # Initialize progress bar
        total_evaluations = len(user_ids) * (len(forgetting_mechanisms) + 1)  # +1 for baseline
        progress_bar = tqdm(total=total_evaluations, desc="Evaluating forgetting mechanisms")
        
        # Evaluate each user
        for user_id in user_ids:
            if user_id not in test_data or not test_data[user_id]:
                # Skip users without test data
                progress_bar.update(len(forgetting_mechanisms) + 1)
                continue
            
            test_set = test_data[user_id]
            
            # Get baseline recommendations
            baseline_recommendations = baseline_recommender(user_id)
            
            # Evaluate baseline
            baseline_metrics = self.evaluate_all_metrics(user_id, test_set, baseline_recommendations, k_values)
            
            # Add baseline result
            all_results.append({
                'user_id': user_id,
                'mechanism': 'Baseline',
                'type': 'Baseline',
                **baseline_metrics
            })
            
            progress_bar.update(1)
            
            # Store baseline recommendations for comparison
            for mechanism_name, (forgetting_fn, rec_fn, mech_type) in forgetting_mechanisms.items():
                try:
                    # Apply forgetting mechanism
                    forgetting_fn(user_id)
                    
                    # Get recommendations with forgetting
                    forgetting_recommendations = rec_fn(user_id)
                    
                    # Evaluate metrics
                    metrics = self.evaluate_all_metrics(user_id, test_set, forgetting_recommendations, k_values)
                    
                    # Calculate similarity to baseline
                    jaccard_sim = self.calculate_jaccard_similarity(
                        baseline_recommendations, forgetting_recommendations)
                    
                    rank_corr = self.calculate_rank_correlation(
                        baseline_recommendations, forgetting_recommendations)
                    
                    # Add result
                    all_results.append({
                        'user_id': user_id,
                        'mechanism': mechanism_name,
                        'type': mech_type,
                        'jaccard_similarity': jaccard_sim,
                        'rank_correlation': rank_corr,
                        **metrics
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating {mechanism_name} for user {user_id}: {e}")
                
                progress_bar.update(1)
        
        progress_bar.close()
        
        return pd.DataFrame(all_results)
    
    def evaluate_temporal_impact(self, user_ids, test_data, forgetting_mechanisms, 
                                time_points=5, max_days=180, k=10):
        """
        Evaluate how forgetting mechanisms impact recommendations over time.
        
        Args:
            user_ids: List of user IDs to evaluate
            test_data: Dictionary mapping user_id to set of relevant movie IDs
            forgetting_mechanisms: Dict mapping mechanism name to (forgetting_fn, recommender_fn) tuple
            time_points: Number of time points to simulate
            max_days: Maximum number of days to simulate
            k: Value of k to use for metrics
            
        Returns:
            DataFrame with evaluation metrics over time
        """
        # Generate time points
        time_deltas = np.linspace(0, max_days, time_points)
        
        all_results = []
        
        # Evaluate each user
        for user_id in tqdm(user_ids, desc="Evaluating temporal impact"):
            if user_id not in test_data or not test_data[user_id]:
                continue
            
            test_set = test_data[user_id]
            
            # Save original memory strengths for this user
            if self.fm is not None:
                original_strengths = {}
                original_times = {}
                for key, value in self.fm.memory_strength.items():
                    if key[0] == user_id:
                        original_strengths[key] = value
                        if key in self.fm.last_interaction_time:
                            original_times[key] = self.fm.last_interaction_time[key]
            
            # Evaluate each mechanism at each time point
            for mechanism_name, (forgetting_fn, rec_fn, _) in forgetting_mechanisms.items():
                for days in time_deltas:
                    try:
                        # If we have a forgetting mechanism, simulate time passage
                        if self.fm is not None:
                            # Reset memory strengths
                            for key, value in original_strengths.items():
                                self.fm.memory_strength[key] = value
                            
                            # Set interaction times to simulate time passage
                            current_time = datetime.datetime.now().timestamp()
                            for key in original_strengths:
                                self.fm.last_interaction_time[key] = current_time - (days * 24 * 60 * 60)
                        
                        # Apply forgetting mechanism
                        forgetting_fn(user_id)
                        
                        # Get recommendations
                        recommendations = rec_fn(user_id)
                        
                        # Calculate key metrics
                        hit_rate = self.calculate_hit_rate_at_k(test_set, recommendations, k)
                        serendipity = self.calculate_serendipity(user_id, recommendations, k)
                        novelty = self.calculate_novelty(recommendations, k)
                        diversity = self.calculate_diversity(recommendations, k)
                        
                        # Add result
                        all_results.append({
                            'user_id': user_id,
                            'mechanism': mechanism_name,
                            'days': days,
                            'hit_rate': hit_rate,
                            'serendipity': serendipity,
                            'novelty': novelty,
                            'diversity': diversity
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error in temporal evaluation for {mechanism_name}, user {user_id}, days {days}: {e}")
            
            # Restore original memory strengths if needed
            if self.fm is not None:
                for key, value in original_strengths.items():
                    self.fm.memory_strength[key] = value
                
                for key, value in original_times.items():
                    self.fm.last_interaction_time[key] = value
        
        return pd.DataFrame(all_results)
    
    def create_comprehensive_summary(self, results_df, output_prefix="evaluation_summary", 
                                k=10, save_summary=True):
        """
        Create a comprehensive summary of evaluation results.
        
        Args:
            results_df: DataFrame with evaluation results
            output_prefix: Prefix for output files
            k: Value of k to use for metrics in the summary
            save_summary: Whether to save the summary to file
            
        Returns:
            Dictionary with summary metrics
        """
        summary = {}
        
        # Check if DataFrame has required columns
        required_cols = [f'hit_rate@{k}', f'precision@{k}', f'ndcg@{k}', 'mrr', 'serendipity', 'novelty', 'diversity']
        
        if not all(col in results_df.columns for col in required_cols):
            available_k = [int(col.split('@')[1]) for col in results_df.columns if col.startswith('hit_rate@')]
            if available_k:
                k = available_k[0]
                self.logger.info(f"Using k={k} for summary because requested k not found in results")
                required_cols = [f'hit_rate@{k}', f'precision@{k}', f'ndcg@{k}', 'mrr', 'serendipity', 'novelty', 'diversity']
            else:
                self.logger.error("Required columns not found in results_df")
                return summary
        
        # Filter columns for the summary
        metrics_of_interest = [
            f'hit_rate@{k}', 
            f'precision@{k}', 
            f'ndcg@{k}', 
            'mrr', 
            'serendipity', 
            'novelty', 
            'diversity'
        ]
        
        # Group by strategy/mechanism and calculate mean and std
        if 'strategy' in results_df.columns:
            group_col = 'strategy'
        elif 'mechanism' in results_df.columns:
            group_col = 'mechanism'
        else:
            self.logger.error("DataFrame must have either 'strategy' or 'mechanism' column")
            return summary
        
        # Calculate mean metrics for each strategy/mechanism
        grouped_means = results_df.groupby(group_col)[metrics_of_interest].mean().reset_index()
        grouped_stds = results_df.groupby(group_col)[metrics_of_interest].std().reset_index()
        
        # Format for summary
        summary_data = []
        for i, row in grouped_means.iterrows():
            method_name = row[group_col]
            std_row = grouped_stds[grouped_stds[group_col] == method_name].iloc[0]
            
            method_data = {
                'method': method_name,
                'type': results_df[results_df[group_col] == method_name]['type'].iloc[0] if 'type' in results_df.columns else 'N/A',
                'metrics': {}
            }
            
            for metric in metrics_of_interest:
                method_data['metrics'][metric] = {
                    'mean': row[metric],
                    'std': std_row[metric],
                    'rank': 0  # Will be filled in later
                }
            
            summary_data.append(method_data)
        
        # Calculate ranks for each metric
        for metric in metrics_of_interest:
            # Sort methods by mean metric value (higher is better)
            sorted_methods = sorted(summary_data, key=lambda x: x['metrics'][metric]['mean'], reverse=True)
            
            # Assign ranks
            for i, method_data in enumerate(sorted_methods):
                method_name = method_data['method']
                rank = i + 1
                
                # Find this method in summary_data and update rank
                for data in summary_data:
                    if data['method'] == method_name:
                        data['metrics'][metric]['rank'] = rank
        
        # Calculate average rank across all metrics
        for method_data in summary_data:
            ranks = [method_data['metrics'][m]['rank'] for m in metrics_of_interest]
            method_data['avg_rank'] = np.mean(ranks)
        
        # Sort by average rank
        summary_data = sorted(summary_data, key=lambda x: x['avg_rank'])
        
        # Calculate overall metrics
        avg_metrics = {}
        for metric in metrics_of_interest:
            avg_metrics[metric] = np.mean([data['metrics'][metric]['mean'] for data in summary_data])
        
        # Add summary data to result
        summary['methods'] = summary_data
        summary['overall'] = avg_metrics
        summary['k'] = k
        
        # Calculate improvement over baseline if baseline is present
        if any(data['method'] == 'Baseline' for data in summary_data):
            baseline_data = next(data for data in summary_data if data['method'] == 'Baseline')
            baseline_metrics = {m: baseline_data['metrics'][m]['mean'] for m in metrics_of_interest}
            
            for method_data in summary_data:
                if method_data['method'] != 'Baseline':
                    method_data['improvements'] = {}
                    for metric in metrics_of_interest:
                        baseline_value = baseline_metrics[metric]
                        method_value = method_data['metrics'][metric]['mean']
                        
                        if baseline_value > 0:
                            improvement = (method_value - baseline_value) / baseline_value
                            improvement_pct = improvement * 100
                        else:
                            improvement = method_value - baseline_value
                            improvement_pct = float('inf') if method_value > 0 else 0
                        
                        method_data['improvements'][metric] = {
                            'absolute': method_value - baseline_value,
                            'relative': improvement,
                            'percentage': improvement_pct
                        }
        
        # Create visualizations
        if save_summary:
            self._create_summary_visualizations(summary_data, metrics_of_interest, output_prefix)
            
            # Save summary to JSON
            output_path = os.path.join(self.output_dir, f"{output_prefix}.json")
            with open(output_path, 'w') as f:
                # Convert numpy values to Python native types for JSON serialization
                json_summary = {
                    'methods': summary_data,
                    'overall': {k: float(v) for k, v in avg_metrics.items()},
                    'k': k
                }
                json.dump(json_summary, f, indent=2)
                
            self.logger.info(f"Saved summary to {output_path}")
        
        return summary
    
    def _create_summary_visualizations(self, summary_data, metrics, output_prefix):
        """Create visualizations for the summary."""
        # Create a directory for visualizations
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Prepare data for plotting
        method_names = [data['method'] for data in summary_data]
        method_types = [data['type'] for data in summary_data]
        
        # Create color map based on method type
        unique_types = list(set(method_types))
        color_map = {}
        for i, t in enumerate(unique_types):
            colors = plt.cm.tab10.colors
            color_map[t] = colors[i % len(colors)]
        
        # 1. Create radar chart for top methods
        self._create_radar_chart(
            summary_data[:5],  # Top 5 methods
            metrics,
            os.path.join(vis_dir, f"{output_prefix}_radar.png")
        )
        
        # 2. Create bar chart for each metric
        for metric in metrics:
            self._create_metric_bar_chart(
                summary_data,
                metric,
                color_map,
                os.path.join(vis_dir, f"{output_prefix}_{metric}_bar.png")
            )
        
        # 3. Create heatmap of all methods vs all metrics
        self._create_methods_metrics_heatmap(
            summary_data,
            metrics,
            os.path.join(vis_dir, f"{output_prefix}_heatmap.png")
        )
        
        # 4. Create rank correlation matrix between metrics
        self._create_metric_correlation_matrix(
            summary_data,
            metrics,
            os.path.join(vis_dir, f"{output_prefix}_correlation.png")
        )
        
        # 5. If we have baseline, create improvement visualization
        if any(data['method'] == 'Baseline' for data in summary_data):
            self._create_improvement_visualization(
                summary_data,
                metrics,
                os.path.join(vis_dir, f"{output_prefix}_improvements.png")
            )
            
        # 6. Create final summary dashboard
        self._create_summary_dashboard(
            summary_data,
            metrics,
            os.path.join(vis_dir, f"{output_prefix}_dashboard.png")
        )
    
    def _create_radar_chart(self, methods_data, metrics, output_path):
        """Create a radar chart comparing top methods."""
        # Number of metrics
        N = len(metrics)
        
        # Create angles for radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        
        # Close the loop
        angles += angles[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add labels
        plt.xticks(angles[:-1], [m.replace('@', ' at ') for m in metrics], size=12)
        
        # Add radial lines at 0.25, 0.5, 0.75
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=10)
        ax.set_rlabel_position(180 / N)  # Move radial labels to first axis
        
        # Plot data
        for i, method_data in enumerate(methods_data):
            method_name = method_data['method']
            
            # Get values for this method
            values = [method_data['metrics'][m]['mean'] for m in metrics]
            
            # Close the loop
            values += values[:1]
            
            # Plot with different color and line style
            ax.plot(angles, values, 'o-', linewidth=2, label=method_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Comparison of Top Methods Across Metrics', size=15)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metric_bar_chart(self, methods_data, metric, color_map, output_path):
        """Create a bar chart for a single metric across all methods."""
        # Prepare data
        method_names = [data['method'] for data in methods_data]
        method_types = [data['type'] for data in methods_data]
        values = [data['metrics'][metric]['mean'] for data in methods_data]
        errors = [data['metrics'][metric]['std'] for data in methods_data]
        colors = [color_map[t] for t in method_types]
        
        # Sort by value
        sorted_indices = np.argsort(values)[::-1]  # Descending
        method_names = [method_names[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        errors = [errors[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create bars
        bars = plt.bar(range(len(method_names)), values, yerr=errors, 
                      capsize=7, color=colors)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add labels and title
        plt.xlabel('Method')
        plt.ylabel(metric.replace('@', ' at '))
        plt.title(f'Comparison of {metric.replace("@", " at ")} Across Methods')
        
        # Add x-ticks
        plt.xticks(range(len(method_names)), method_names, rotation=45, ha='right')
        
        # Add grid
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add legend for method types
        unique_types = list(set(method_types))
        legend_patches = [plt.Rectangle((0,0), 1, 1, color=color_map[t]) for t in unique_types]
        plt.legend(legend_patches, unique_types, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_methods_metrics_heatmap(self, methods_data, metrics, output_path):
        """Create a heatmap of all methods vs all metrics."""
        # Prepare data
        method_names = [data['method'] for data in methods_data]
        values = np.zeros((len(method_names), len(metrics)))
        
        for i, method_data in enumerate(methods_data):
            for j, metric in enumerate(metrics):
                values[i, j] = method_data['metrics'][metric]['mean']
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(values, annot=True, fmt='.3f', cmap='YlGnBu',
                   xticklabels=[m.replace('@', ' at ') for m in metrics],
                   yticklabels=method_names)
        
        plt.title('Performance Heatmap: Methods vs Metrics')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metric_correlation_matrix(self, methods_data, metrics, output_path):
        """Create a correlation matrix between metrics."""
        # Prepare data
        values = np.zeros((len(methods_data), len(metrics)))
        
        for i, method_data in enumerate(methods_data):
            for j, metric in enumerate(metrics):
                values[i, j] = method_data['metrics'][metric]['mean']
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(values.T)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=[m.replace('@', ' at ') for m in metrics],
                   yticklabels=[m.replace('@', ' at ') for m in metrics],
                   vmin=-1, vmax=1)
        
        plt.title('Correlation Matrix Between Metrics')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_improvement_visualization(self, methods_data, metrics, output_path):
        """Create a visualization of improvements over baseline."""
        # Find baseline method
        baseline_data = next(data for data in methods_data if data['method'] == 'Baseline')
        
        # Filter out baseline from visualization
        filtered_methods = [data for data in methods_data if data['method'] != 'Baseline']
        
        if not filtered_methods:
            return
        
        # Prepare data
        method_names = [data['method'] for data in filtered_methods]
        improvements = {}
        
        for metric in metrics:
            improvements[metric] = []
            for method_data in filtered_methods:
                if 'improvements' in method_data and metric in method_data['improvements']:
                    improvements[metric].append(method_data['improvements'][metric]['percentage'])
                else:
                    # Calculate improvement
                    baseline_value = baseline_data['metrics'][metric]['mean']
                    method_value = method_data['metrics'][metric]['mean']
                    
                    if baseline_value > 0:
                        improvement_pct = ((method_value - baseline_value) / baseline_value) * 100
                    else:
                        improvement_pct = float('inf') if method_value > 0 else 0
                    
                    improvements[metric].append(improvement_pct)
        
        # Create figure with subplots for each metric
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        
        # Flatten axes if there's only one row
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        
        # Handle case with only one subplot
        if n_metrics == 1:
            axes = np.array([axes])
        
        # Create a bar chart for each metric
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Create bars with colors indicating positive/negative
            colors = ['green' if imp > 0 else 'red' for imp in improvements[metric]]
            bars = ax.bar(range(len(method_names)), improvements[metric], color=colors)
            
            # Add zero line
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                label_text = f'{height:.1f}%' if abs(height) < 1000 else 'N/A'
                va = 'bottom' if height > 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.05, label_text,
                      ha='center', va=va, fontsize=10)
            
            # Add labels and title
            ax.set_xlabel('Method')
            ax.set_ylabel('Improvement (%)')
            ax.set_title(f'Improvement in {metric.replace("@", " at ")}')
            
            # Add x-ticks
            ax.set_xticks(range(len(method_names)))
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])
        
        plt.suptitle('Percentage Improvement Over Baseline', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_dashboard(self, methods_data, metrics, output_path):
        """Create a summary dashboard with key insights."""
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 2, 1.5])
        
        # 1. Top section: Title and key stats
        ax_title = plt.subplot(gs[0, :])
        ax_title.axis('off')
        
        # Find best method based on average rank
        sorted_methods = sorted(methods_data, key=lambda x: x['avg_rank'])
        best_method = sorted_methods[0]
        
        # Calculate average improvement if we have baseline
        avg_improvement = 'N/A'
        if any(data['method'] == 'Baseline' for data in methods_data) and 'improvements' in best_method:
            improvements = [best_method['improvements'][m]['percentage'] for m in metrics 
                           if m in best_method['improvements']]
            if improvements:
                avg_improvement = f"{np.mean(improvements):.2f}%"
        
        title_text = f"Forgetting Mechanism Evaluation Summary\n\n"
        title_text += f"Best Overall Method: {best_method['method']} (Avg Rank: {best_method['avg_rank']:.2f})\n"
        title_text += f"Average Improvement Over Baseline: {avg_improvement}\n"
        
        ax_title.text(0.5, 0.5, title_text, fontsize=16, ha='center', va='center')
        
        # 2. Middle left: Radar chart of top 3 methods
        ax_radar = plt.subplot(gs[1, 0], polar=True)
        
        # Number of metrics
        N = len(metrics)
        
        # Create angles for radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        
        # Close the loop
        angles += angles[:1]
        
        # Add labels
        plt.xticks(angles[:-1], [m.replace('@', ' at ') for m in metrics], size=12)
        
        # Add radial lines
        ax_radar.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax_radar.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=10)
        
        # Plot data for top 3 methods
        for i, method_data in enumerate(sorted_methods[:3]):
            method_name = method_data['method']
            
            # Get values for this method
            values = [method_data['metrics'][m]['mean'] for m in metrics]
            
            # Close the loop
            values += values[:1]
            
            # Plot with different color and line style
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=method_name)
            ax_radar.fill(angles, values, alpha=0.1)
        
        # Add legend
        ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        ax_radar.set_title('Top 3 Methods Across Metrics', size=14)
        
        # 3. Middle right: Bar chart of best metric for each method
        ax_best = plt.subplot(gs[1, 1])
        
        # For each method, find its best metric
        best_metrics = []
        for method_data in methods_data:
            method_name = method_data['method']
            best_metric = max(metrics, key=lambda m: method_data['metrics'][m]['rank'])
            best_value = method_data['metrics'][best_metric]['mean']
            best_metrics.append((method_name, best_metric, best_value))
        
        # Sort by method name
        best_metrics.sort(key=lambda x: x[0])
        
        # Create bars
        method_names = [bm[0] for bm in best_metrics]
        best_metric_names = [bm[1].replace('@', ' at ') for bm in best_metrics]
        values = [bm[2] for bm in best_metrics]
        
        # Define a custom colormap
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(metrics)))
        color_map = {m: colors[i] for i, m in enumerate(metrics)}
        bar_colors = [color_map[bm[1]] for bm in best_metrics]
        
        bars = ax_best.bar(range(len(method_names)), values, color=bar_colors)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            metric_name = best_metric_names[i]
            ax_best.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                       f'{height:.3f}\n({metric_name})', ha='center', va='bottom', fontsize=9)
        
        # Add labels and title
        ax_best.set_xlabel('Method')
        ax_best.set_ylabel('Best Metric Value')
        ax_best.set_title('Best Performing Metric for Each Method')
        
        # Add x-ticks
        ax_best.set_xticks(range(len(method_names)))
        ax_best.set_xticklabels(method_names, rotation=45, ha='right')
        
        # Add grid
        ax_best.grid(True, alpha=0.3, axis='y')
        
        # 4. Bottom: Heatmap of improvements over baseline
        ax_heatmap = plt.subplot(gs[2, :])
        
        # Find baseline method if exists
        if any(data['method'] == 'Baseline' for data in methods_data):
            # Filter out baseline
            filtered_methods = [data for data in methods_data if data['method'] != 'Baseline']
            
            if filtered_methods:
                # Prepare data
                method_names = [data['method'] for data in filtered_methods]
                improvements = np.zeros((len(method_names), len(metrics)))
                
                for i, method_data in enumerate(filtered_methods):
                    for j, metric in enumerate(metrics):
                        if 'improvements' in method_data and metric in method_data['improvements']:
                            improvements[i, j] = method_data['improvements'][metric]['percentage']
                
                # Create heatmap
                heatmap = sns.heatmap(improvements, annot=True, fmt='.1f', cmap='RdYlGn',
                                   xticklabels=[m.replace('@', ' at ') for m in metrics],
                                   yticklabels=method_names,
                                   center=0, ax=ax_heatmap)
                
                ax_heatmap.set_title('Percentage Improvement Over Baseline')
        else:
            ax_heatmap.axis('off')
            ax_heatmap.text(0.5, 0.5, "No baseline method found for comparison", 
                          fontsize=14, ha='center', va='center')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def evaluate_user_segments(self, results_df, user_segments, k=10):
        """
        Evaluate performance across different user segments.
        
        Args:
            results_df: DataFrame with evaluation results
            user_segments: Dict mapping segment name to list of user IDs
            k: Value of k to use for metrics
            
        Returns:
            DataFrame with segment-level results
        """
        segment_results = []
        
        # Check if DataFrame has required columns
        required_cols = [f'hit_rate@{k}', f'precision@{k}', f'ndcg@{k}', 'mrr', 'serendipity', 'novelty', 'diversity']
        
        if not all(col in results_df.columns for col in required_cols):
            self.logger.error("Required columns not found in results_df")
            return pd.DataFrame()
        
        # Identify strategy/mechanism column
        if 'strategy' in results_df.columns:
            group_col = 'strategy'
        elif 'mechanism' in results_df.columns:
            group_col = 'mechanism'
        else:
            self.logger.error("DataFrame must have either 'strategy' or 'mechanism' column")
            return pd.DataFrame()
        
        # Evaluate each segment
        for segment_name, user_ids in user_segments.items():
            # Filter results for this segment
            segment_df = results_df[results_df['user_id'].isin(user_ids)]
            
            if segment_df.empty:
                self.logger.warning(f"No data for segment: {segment_name}")
                continue
            
            # Calculate mean metrics for each strategy/mechanism
            for strat in segment_df[group_col].unique():
                strat_segment_df = segment_df[segment_df[group_col] == strat]
                
                # Calculate mean metrics
                metrics_means = {}
                for col in required_cols:
                    metrics_means[col] = strat_segment_df[col].mean()
                
                # Add result
                segment_results.append({
                    'segment': segment_name,
                    group_col: strat,
                    'user_count': len(user_ids),
                    **metrics_means
                })
        
        return pd.DataFrame(segment_results)
    
    def visualize_user_segments(self, segment_results_df, output_prefix="segment_analysis"):
        """
        Visualize performance across user segments.
        
        Args:
            segment_results_df: DataFrame from evaluate_user_segments
            output_prefix: Prefix for output files
            
        Returns:
            None
        """
        if segment_results_df.empty:
            self.logger.error("Empty segment results DataFrame")
            return
        
        # Create a directory for visualizations
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Identify metrics columns
        metrics_cols = [col for col in segment_results_df.columns 
                      if col not in ['segment', 'strategy', 'mechanism', 'user_count']]
        
        # Identify strategy/mechanism column
        if 'strategy' in segment_results_df.columns:
            group_col = 'strategy'
        elif 'mechanism' in segment_results_df.columns:
            group_col = 'mechanism'
        else:
            self.logger.error("DataFrame must have either 'strategy' or 'mechanism' column")
            return
        
        # Create a heatmap for each metric
        for metric in metrics_cols:
            # Create pivot table
            pivot_df = segment_results_df.pivot(index='segment', columns=group_col, values=metric)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlGnBu')
            
            plt.title(f'{metric.replace("@", " at ")} by User Segment and Method')
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(vis_dir, f"{output_prefix}_{metric}_heatmap.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a bar chart comparing segments for each method
        methods = segment_results_df[group_col].unique()
        
        for method in methods:
            method_df = segment_results_df[segment_results_df[group_col] == method]
            
            # Create subplots for each metric
            n_metrics = len(metrics_cols)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            
            # Handle case with only one subplot
            if n_metrics == 1:
                axes = np.array([axes])
                
            # Flatten axes if there's only one row
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            
            # Create a bar chart for each metric
            for i, metric in enumerate(metrics_cols):
                row = i // n_cols
                col = i % n_cols
                
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                # Sort segments by metric value
                sorted_indices = np.argsort(method_df[metric].values)[::-1]  # Descending
                segments = method_df['segment'].values[sorted_indices]
                values = method_df[metric].values[sorted_indices]
                
                # Create bars
                bars = ax.bar(range(len(segments)), values)
                
                # Add value labels
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=10)
                
                # Add labels and title
                ax.set_xlabel('Segment')
                ax.set_ylabel(metric.replace('@', ' at '))
                ax.set_title(f'{metric.replace("@", " at ")} by Segment')
                
                # Add x-ticks
                ax.set_xticks(range(len(segments)))
                ax.set_xticklabels(segments, rotation=45, ha='right')
                
                # Add grid
                ax.grid(True, alpha=0.3, axis='y')
            
            # Remove empty subplots
            for i in range(n_metrics, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])
            
            plt.suptitle(f'Performance of {method} Across User Segments', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
            
            # Save figure
            output_path = os.path.join(vis_dir, f"{output_prefix}_{method}_segments.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def analyze_tradeoffs(self, results_df, accuracy_metrics, beyond_accuracy_metrics, output_prefix="tradeoff_analysis"):
        """
        Analyze tradeoffs between accuracy and beyond-accuracy metrics.
        
        Args:
            results_df: DataFrame with evaluation results
            accuracy_metrics: List of accuracy metric columns
            beyond_accuracy_metrics: List of beyond-accuracy metric columns
            output_prefix: Prefix for output files
            
        Returns:
            DataFrame with tradeoff analysis
        """
        if results_df.empty:
            self.logger.error("Empty results DataFrame")
            return pd.DataFrame()
        
        # Check if all required columns exist
        all_metrics = accuracy_metrics + beyond_accuracy_metrics
        if not all(col in results_df.columns for col in all_metrics):
            self.logger.error("Not all required metrics found in results_df")
            return pd.DataFrame()
        
        # Identify strategy/mechanism column
        if 'strategy' in results_df.columns:
            group_col = 'strategy'
        elif 'mechanism' in results_df.columns:
            group_col = 'mechanism'
        else:
            self.logger.error("DataFrame must have either 'strategy' or 'mechanism' column")
            return pd.DataFrame()
        
        # Create a directory for visualizations
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Group by strategy/mechanism
        grouped_results = results_df.groupby(group_col)[all_metrics].mean().reset_index()
        
        # Calculate normalized scores for each metric
        normalized_results = grouped_results.copy()
        
        for metric in all_metrics:
            min_val = grouped_results[metric].min()
            max_val = grouped_results[metric].max()
            
            if max_val > min_val:
                normalized_results[f"norm_{metric}"] = (grouped_results[metric] - min_val) / (max_val - min_val)
            else:
                normalized_results[f"norm_{metric}"] = 1.0  # All values are the same
        
        # Calculate aggregate accuracy and beyond-accuracy scores
        normalized_results["accuracy_score"] = normalized_results[[f"norm_{m}" for m in accuracy_metrics]].mean(axis=1)
        normalized_results["beyond_accuracy_score"] = normalized_results[[f"norm_{m}" for m in beyond_accuracy_metrics]].mean(axis=1)
        
        # Calculate tradeoff score
        normalized_results["tradeoff_score"] = (normalized_results["accuracy_score"] + normalized_results["beyond_accuracy_score"]) / 2
        
        # Create scatter plot of accuracy vs. beyond-accuracy
        plt.figure(figsize=(12, 10))
        
        # Create a scatter plot
        scatter = plt.scatter(
            normalized_results["accuracy_score"],
            normalized_results["beyond_accuracy_score"],
            s=100,
            c=normalized_results["tradeoff_score"],
            cmap="viridis",
            alpha=0.8
        )
        
        # Add method names as labels
        for i, row in normalized_results.iterrows():
            plt.annotate(
                row[group_col],
                (row["accuracy_score"], row["beyond_accuracy_score"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10
            )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Tradeoff Score (Higher is Better)")
        
        # Add reference lines
        plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add labels and title
        plt.xlabel("Accuracy Score (Normalized)")
        plt.ylabel("Beyond-Accuracy Score (Normalized)")
        plt.title("Tradeoff Analysis: Accuracy vs. Beyond-Accuracy Metrics")
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = os.path.join(vis_dir, f"{output_prefix}_scatter.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create radar chart for selected methods
        self._create_tradeoff_radar_chart(
            normalized_results,
            group_col,
            accuracy_metrics + beyond_accuracy_metrics,
            os.path.join(vis_dir, f"{output_prefix}_radar.png")
        )
        
        # Create a table showing detailed metric scores
        self._create_tradeoff_table(
            normalized_results,
            group_col,
            accuracy_metrics,
            beyond_accuracy_metrics,
            os.path.join(vis_dir, f"{output_prefix}_table.png")
        )
        
        # Identify methods with best tradeoffs
        best_tradeoff = normalized_results.loc[normalized_results['tradeoff_score'].idxmax()]
        best_accuracy = normalized_results.loc[normalized_results['accuracy_score'].idxmax()]
        best_beyond_accuracy = normalized_results.loc[normalized_results['beyond_accuracy_score'].idxmax()]
        
        # Log best methods
        self.logger.info(f"Best overall tradeoff: {best_tradeoff[group_col]} (score: {best_tradeoff['tradeoff_score']:.3f})")
        self.logger.info(f"Best accuracy: {best_accuracy[group_col]} (score: {best_accuracy['accuracy_score']:.3f})")
        self.logger.info(f"Best beyond-accuracy: {best_beyond_accuracy[group_col]} (score: {best_beyond_accuracy['beyond_accuracy_score']:.3f})")
        
        # Create a summary visualization
        self._create_tradeoff_summary(
            normalized_results,
            group_col,
            best_tradeoff[group_col],
            best_accuracy[group_col],
            best_beyond_accuracy[group_col],
            os.path.join(vis_dir, f"{output_prefix}_summary.png")
        )
        
        return normalized_results
        
    def _create_tradeoff_table(self, normalized_results, group_col, accuracy_metrics, beyond_accuracy_metrics, output_path):
        """Create a table showing detailed metric scores for each method."""
        # Get top methods sorted by tradeoff score
        sorted_results = normalized_results.sort_values("tradeoff_score", ascending=False)
        
        # Create figure
        plt.figure(figsize=(14, len(sorted_results) * 0.5 + 2))
        ax = plt.gca()
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        method_names = sorted_results[group_col].tolist()
        
        # Format data for the table
        table_data = []
        for i, row in sorted_results.iterrows():
            row_data = [row[group_col]]
            # Add accuracy score
            row_data.append(f"{row['accuracy_score']:.3f}")
            # Add beyond-accuracy score
            row_data.append(f"{row['beyond_accuracy_score']:.3f}")
            # Add tradeoff score
            row_data.append(f"{row['tradeoff_score']:.3f}")
            
            table_data.append(row_data)
        
        # Create table
        column_labels = ['Method', 'Accuracy Score', 'Beyond-Accuracy Score', 'Tradeoff Score']
        table = ax.table(
            cellText=table_data,
            colLabels=column_labels,
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color code the cells based on values
        for i in range(1, len(table_data) + 1):
            for j in range(1, 4):  # Score columns
                cell = table[i, j]
                score = float(table_data[i-1][j])
                
                # Color from red (0.0) to green (1.0)
                if score < 0.3:
                    cell.set_facecolor('#ffcccc')
                elif score < 0.5:
                    cell.set_facecolor('#ffffcc')
                elif score < 0.7:
                    cell.set_facecolor('#ccffcc')
                else:
                    cell.set_facecolor('#99ff99')
        
        plt.title('Tradeoff Analysis: Method Scores', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_tradeoff_radar_chart(self, normalized_results, group_col, metrics, output_path):
        """Create a radar chart comparing methods based on normalized metrics."""
        # Get top methods sorted by tradeoff score
        sorted_results = normalized_results.sort_values("tradeoff_score", ascending=False)
        top_methods = sorted_results.head(5)  # Show top 5 methods

        # Number of metrics
        N = len(metrics)

        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = plt.subplot(111, polar=True)

        # Create angles for radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()

        # Close the loop
        angles += angles[:1]

        # Add labels
        plt.xticks(angles[:-1], [m.replace('@', ' at ') for m in metrics], size=12)

        # Add radial lines
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=10)

        # Define colors
        colors = plt.cm.tab10.colors

        # Plot data for each method
        for i, (_, method_data) in enumerate(top_methods.iterrows()):
            method_name = method_data[group_col]

            # Get normalized values for this method
            values = [method_data[f"norm_{m}"] for m in metrics]

            # Close the loop
            values += values[:1]

            # Plot with different color and line style
            ax.plot(angles, values, 'o-', linewidth=2, label=method_name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.title('Top Methods Across Normalized Metrics', size=15)
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    
    def _create_tradeoff_summary(self, normalized_results, group_col, best_tradeoff, best_accuracy, best_beyond_accuracy, output_path):
        """Create a summary visualization highlighting the best methods."""
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        plt.scatter(
            normalized_results["accuracy_score"],
            normalized_results["beyond_accuracy_score"],
            s=80,
            c=normalized_results["tradeoff_score"],
            cmap="viridis",
            alpha=0.7
        )
        
        # Highlight best methods
        for method_name, color, marker in [
            (best_tradeoff, 'red', '*'),
            (best_accuracy, 'blue', 's'),
            (best_beyond_accuracy, 'green', '^')
        ]:
            method_data = normalized_results[normalized_results[group_col] == method_name]
            if not method_data.empty:
                plt.scatter(
                    method_data["accuracy_score"],
                    method_data["beyond_accuracy_score"],
                    s=150,
                    c=color,
                    marker=marker,
                    label=f"{method_name}"
                )
        
        # Add labels for all methods
        for i, row in normalized_results.iterrows():
            plt.annotate(
                row[group_col],
                (row["accuracy_score"], row["beyond_accuracy_score"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )
        
        # Add reference lines
        plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add labels and title
        plt.xlabel("Accuracy Score (Normalized)")
        plt.ylabel("Beyond-Accuracy Score (Normalized)")
        plt.title("Tradeoff Summary: Best Methods Highlighted", fontsize=14)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add legend with custom labels
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=12, label=f'Best Tradeoff: {best_tradeoff}'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label=f'Best Accuracy: {best_accuracy}'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label=f'Best Beyond-Accuracy: {best_beyond_accuracy}')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Add text summary
        text_summary = [
            "Analysis Summary:",
            f"• Best Overall Tradeoff: {best_tradeoff}",
            f"• Best for Accuracy Metrics: {best_accuracy}",
            f"• Best for Beyond-Accuracy Metrics: {best_beyond_accuracy}"
        ]
        
        plt.figtext(0.05, 0.05, "\n".join(text_summary), fontsize=11, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_temporal_dynamics(self, temporal_results, output_prefix="temporal_analysis"):
        """
        Analyze how metrics change over time for different forgetting mechanisms.

        Args:
            temporal_results: DataFrame from evaluate_temporal_impact
            output_prefix: Prefix for output files

        Returns:
            DataFrame with temporal analysis results
        """
        if temporal_results.empty:
            self.logger.error("Empty temporal results DataFrame")
            return pd.DataFrame()

        # Create a directory for visualizations
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Check required columns
        required_cols = ['mechanism', 'days', 'hit_rate', 'serendipity', 'novelty', 'diversity']
        if not all(col in temporal_results.columns for col in required_cols):
            self.logger.error("Required columns not found in temporal_results")
            return pd.DataFrame()

        # Create metrics over time visualization
        self._create_metrics_over_time_visualization(
            temporal_results,
            os.path.join(vis_dir, f"{output_prefix}_metrics_over_time.png")
        )

        # Create rate of decay visualization
        self._create_rate_of_decay_visualization(
            temporal_results,
            os.path.join(vis_dir, f"{output_prefix}_decay_rate.png")
        )

        # Calculate summary statistics
        analysis_results = self._calculate_temporal_summary_stats(temporal_results)

        # Save results
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv(os.path.join(self.output_dir, f"{output_prefix}_summary.csv"), index=False)

        return analysis_df

    def _create_metrics_over_time_visualization(self, temporal_results, output_path):
        """Create a visualization of how metrics change over time for each mechanism."""
        # Get unique mechanisms and metrics
        mechanisms = temporal_results['mechanism'].unique()
        metrics = ['hit_rate', 'serendipity', 'novelty', 'diversity']

        # Create figure with subplots for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)), sharex=True)

        # Handle case with only one subplot
        if len(metrics) == 1:
            axes = [axes]

        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Calculate mean values for each mechanism and time point
            pivot_df = temporal_results.pivot_table(
                index='days', 
                columns='mechanism', 
                values=metric, 
                aggfunc='mean'
            )

            # Plot lines for each mechanism
            pivot_df.plot(ax=ax, marker='o', linewidth=2)

            # Add labels and grid
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{metric.capitalize()} Over Time")
            ax.grid(True, alpha=0.3)

            # Add legend for the last subplot only
            if i == len(metrics) - 1:
                ax.legend(title="Mechanism", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add x-axis label to the bottom subplot
        axes[-1].set_xlabel("Days")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_rate_of_decay_visualization(self, temporal_results, output_path):
        """Create a visualization of how quickly metrics decay over time."""
        # Get unique mechanisms
        mechanisms = temporal_results['mechanism'].unique()

        # Calculate rate of decay for hit_rate
        decay_data = []

        for mechanism in mechanisms:
            mech_data = temporal_results[temporal_results['mechanism'] == mechanism]

            # Group by user and calculate decay rate
            for user_id in mech_data['user_id'].unique():
                user_data = mech_data[mech_data['user_id'] == user_id]

                if len(user_data) >= 2:
                    # Sort by days
                    user_data = user_data.sort_values('days')

                    # Calculate initial and final hit rate
                    initial_hit_rate = user_data['hit_rate'].iloc[0]
                    final_hit_rate = user_data['hit_rate'].iloc[-1]

                    # Skip if initial hit rate is zero to avoid division by zero
                    if initial_hit_rate > 0:
                        max_days = user_data['days'].max()

                        # Decay rate per day
                        if max_days > 0:
                            decay_rate = (initial_hit_rate - final_hit_rate) / (initial_hit_rate * max_days)

                            # Add to results
                            decay_data.append({
                                'mechanism': mechanism,
                                'user_id': user_id,
                                'decay_rate': decay_rate,
                                'initial_hit_rate': initial_hit_rate,
                                'final_hit_rate': final_hit_rate
                            })

        if not decay_data:
            return

        # Convert to DataFrame
        decay_df = pd.DataFrame(decay_data)

        # Calculate average decay rate by mechanism
        avg_decay = decay_df.groupby('mechanism')['decay_rate'].mean().reset_index()
        avg_decay = avg_decay.sort_values('decay_rate')

        # Create bar chart
        plt.figure(figsize=(10, 6))

        # Create bars with colors indicating decay rate
        colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(avg_decay)))
        bars = plt.bar(avg_decay['mechanism'], avg_decay['decay_rate'], color=colors)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10)

        # Add labels and title
        plt.xlabel('Forgetting Mechanism')
        plt.ylabel('Average Decay Rate (per day)')
        plt.title('Rate of Performance Decay Over Time by Mechanism')

        # Add grid
        plt.grid(True, alpha=0.3, axis='y')

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_temporal_summary_stats(self, temporal_results):
        """Calculate summary statistics for temporal analysis."""
        # Get unique mechanisms
        mechanisms = temporal_results['mechanism'].unique()

        # Calculate statistics
        summary_stats = []

        for mechanism in mechanisms:
            mech_data = temporal_results[temporal_results['mechanism'] == mechanism]

            # Group by days
            grouped = mech_data.groupby('days')

            # Calculate mean metrics for each time point
            means = grouped[['hit_rate', 'serendipity', 'novelty', 'diversity']].mean()

            # Calculate initial values (at day 0)
            if 0.0 in means.index:
                initial_hit_rate = means.loc[0.0, 'hit_rate']
                initial_serendipity = means.loc[0.0, 'serendipity']
            else:
                initial_hit_rate = means['hit_rate'].iloc[0]
                initial_serendipity = means['serendipity'].iloc[0]

            # Calculate final values (at max days)
            max_day = means.index.max()
            final_hit_rate = means.loc[max_day, 'hit_rate']
            final_serendipity = means.loc[max_day, 'serendipity']

            # Calculate relative changes
            if initial_hit_rate > 0:
                hit_rate_change = (final_hit_rate - initial_hit_rate) / initial_hit_rate
            else:
                hit_rate_change = 0

            if initial_serendipity > 0:
                serendipity_change = (final_serendipity - initial_serendipity) / initial_serendipity
            else:
                serendipity_change = 0

            # Calculate consistency (standard deviation across time points)
            hit_rate_std = means['hit_rate'].std()
            serendipity_std = means['serendipity'].std()

            # Add to results
            summary_stats.append({
                'mechanism': mechanism,
                'initial_hit_rate': initial_hit_rate,
                'final_hit_rate': final_hit_rate,
                'hit_rate_change': hit_rate_change,
                'hit_rate_stability': 1.0 - (hit_rate_std / max(0.001, initial_hit_rate)),
                'initial_serendipity': initial_serendipity,
                'final_serendipity': final_serendipity,
                'serendipity_change': serendipity_change,
                'serendipity_stability': 1.0 - (serendipity_std / max(0.001, initial_serendipity)),
                'overall_stability': 1.0 - (hit_rate_std / max(0.001, initial_hit_rate)) * 0.7 - 
                                   (serendipity_std / max(0.001, initial_serendipity)) * 0.3
            })

        return summary_stats
    
    def generate_executive_summary(self, all_results, output_prefix="executive_summary"):
        """
        Generate an executive summary of all evaluation results.

        Args:
            all_results: Dictionary with different result DataFrames
            output_prefix: Prefix for output files

        Returns:
            Dictionary with executive summary
        """
        summary = {
            'overview': {},
            'mechanisms': [],
            'recommendations': [],
            'insights': []
        }

        # Create a directory for visualizations
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Process mechanism results
        if 'mechanisms' in all_results and not all_results['mechanisms'].empty:
            mechanism_df = all_results['mechanisms']

            # Get average metrics by mechanism
            if 'mechanism' in mechanism_df.columns:
                grouped = mechanism_df.groupby('mechanism')

                # Find metrics columns
                metric_cols = [col for col in mechanism_df.columns 
                             if any(col.startswith(m) for m in self.accuracy_metrics + self.beyond_accuracy_metrics)
                             and col not in ['mechanism', 'strategy', 'user_id', 'type']]

                if metric_cols:
                    # Calculate mean values
                    mean_metrics = grouped[metric_cols].mean().reset_index()

                    # Add to summary
                    for _, row in mean_metrics.iterrows():
                        mech_name = row['mechanism']

                        # Get mechanism type if available
                        mech_type = "Unknown"
                        if 'type' in mechanism_df.columns:
                            type_values = mechanism_df[mechanism_df['mechanism'] == mech_name]['type'].unique()
                            if len(type_values) > 0:
                                mech_type = type_values[0]

                        # Add metrics
                        mech_metrics = {}
                        for col in metric_cols:
                            mech_metrics[col] = row[col]

                        # Add to summary
                        summary['mechanisms'].append({
                            'name': mech_name,
                            'type': mech_type,
                            'metrics': mech_metrics
                        })

        # Process tradeoff results
        if 'tradeoffs' in all_results and not all_results['tradeoffs'].empty:
            tradeoffs_df = all_results['tradeoffs']

            if 'tradeoff_score' in tradeoffs_df.columns and len(tradeoffs_df) > 0:
                # Get best overall mechanism
                best_idx = tradeoffs_df['tradeoff_score'].idxmax()
                best_row = tradeoffs_df.iloc[best_idx]

                # Add to overview
                if 'mechanism' in best_row:
                    summary['overview']['best_overall_mechanism'] = best_row['mechanism']
                    summary['overview']['best_overall_score'] = best_row['tradeoff_score']

                    # Add recommendation
                    summary['recommendations'].append(
                        f"Best overall forgetting mechanism: {best_row['mechanism']} with balanced performance across all metrics."
                    )

        # Process temporal results
        if 'temporal' in all_results and not all_results['temporal'].empty:
            temporal_df = all_results['temporal']

            if 'hit_rate_stability' in temporal_df.columns and len(temporal_df) > 0:
                # Get most stable mechanism
                stable_idx = temporal_df['hit_rate_stability'].idxmax()
                stable_row = temporal_df.iloc[stable_idx]

                # Add to overview
                if 'mechanism' in stable_row:
                    summary['overview']['most_stable_mechanism'] = stable_row['mechanism']
                    summary['overview']['stability_score'] = stable_row['hit_rate_stability']

                    # Add recommendation
                    summary['recommendations'].append(
                        f"Most temporally stable mechanism: {stable_row['mechanism']} with consistent performance over time."
                    )

                # Get mechanism with least performance degradation
                if 'hit_rate_change' in temporal_df.columns:
                    least_degradation_idx = temporal_df['hit_rate_change'].idxmax()
                    least_degradation_row = temporal_df.iloc[least_degradation_idx]

                    # Add to overview
                    if 'mechanism' in least_degradation_row:
                        summary['overview']['least_degradation_mechanism'] = least_degradation_row['mechanism']
                        summary['overview']['degradation_score'] = least_degradation_row['hit_rate_change']

                        # Add insight
                        summary['insights'].append(
                            f"The {least_degradation_row['mechanism']} mechanism showed the least performance degradation over time, making it suitable for long-term user modeling."
                        )

        # Process segment results
        if 'segments' in all_results and not all_results['segments'].empty:
            segment_df = all_results['segments']

            # Get unique segments
            if 'segment' in segment_df.columns:
                segments = segment_df['segment'].unique()

                for segment in segments:
                    segment_data = segment_df[segment_df['segment'] == segment]

                    # Get best mechanism for this segment
                    if 'hit_rate@10' in segment_data.columns and len(segment_data) > 0:
                        best_idx = segment_data['hit_rate@10'].idxmax()

                        # Check if the index is valid before accessing it
                        if best_idx in segment_data.index:
                            best_row = segment_data.loc[best_idx]

                            # Add insight
                            if 'mechanism' in best_row:
                                summary['insights'].append(
                                    f"For users in the '{segment}' segment, the {best_row['mechanism']} mechanism performs best."
                                )

        # Create a comprehensive dashboard
        self._create_executive_dashboard(
            all_results,
            summary,
            os.path.join(vis_dir, f"{output_prefix}_dashboard.png")
        )

        # Save summary to JSON
        output_path = os.path.join(self.output_dir, f"{output_prefix}.json")
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary
    def _create_executive_dashboard(self, all_results, summary, output_path):
        """Create an executive dashboard visualization."""
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        
        # Create grid for subplots
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1.5, 1.5])
        
        # 1. Title and key stats
        ax_title = plt.subplot(gs[0, :])
        ax_title.axis('off')
        
        # Create title text
        title_text = "Executive Summary: Forgetting Mechanisms Evaluation\n\n"
        
        # Add best overall mechanism if available
        if 'overview' in summary and 'best_overall_mechanism' in summary['overview']:
            title_text += f"Best Overall Mechanism: {summary['overview']['best_overall_mechanism']}\n"
        
        # Add most stable mechanism if available
        if 'overview' in summary and 'most_stable_mechanism' in summary['overview']:
            title_text += f"Most Stable Mechanism: {summary['overview']['most_stable_mechanism']}\n"
        
        ax_title.text(0.5, 0.5, title_text, fontsize=16, ha='center', va='center')
        
        # 2. Top mechanisms comparison
        ax_top = plt.subplot(gs[1, 0])
        
        # Get mechanism data
        if 'mechanisms' in all_results and not all_results['mechanisms'].empty:
            mech_df = all_results['mechanisms']
            
            if 'mechanism' in mech_df.columns and 'hit_rate@10' in mech_df.columns:
                # Calculate average hit rate by mechanism
                grouped = mech_df.groupby('mechanism')['hit_rate@10'].mean().reset_index()
                
                # Sort by hit rate
                grouped = grouped.sort_values('hit_rate@10', ascending=False)
                
                # Plot top 5 mechanisms
                top_mechs = grouped.head(5)
                
                # Create bar chart
                bars = ax_top.bar(top_mechs['mechanism'], top_mechs['hit_rate@10'])
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax_top.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                             f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                
                # Add labels and title
                ax_top.set_ylabel('Hit Rate@10')
                ax_top.set_title('Top Mechanisms by Hit Rate')
                
                # Rotate x-axis labels
                ax_top.set_xticklabels(top_mechs['mechanism'], rotation=45, ha='right')
                
                # Add grid
                ax_top.grid(True, alpha=0.3, axis='y')
        
        # 3. Tradeoff visualization
        ax_tradeoff = plt.subplot(gs[1, 1])
        
        # Get tradeoff data
        if 'tradeoffs' in all_results and not all_results['tradeoffs'].empty:
            tradeoff_df = all_results['tradeoffs']
            
            if 'accuracy_score' in tradeoff_df.columns and 'beyond_accuracy_score' in tradeoff_df.columns:
                # Create scatter plot
                ax_tradeoff.scatter(
                    tradeoff_df['accuracy_score'],
                    tradeoff_df['beyond_accuracy_score'],
                    s=100,
                    alpha=0.7
                )
                
                # Add labels for each point
                for i, row in tradeoff_df.iterrows():
                    ax_tradeoff.annotate(
                        row['mechanism'],
                        (row['accuracy_score'], row['beyond_accuracy_score']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8
                    )
                
                # Add reference lines
                ax_tradeoff.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
                ax_tradeoff.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
                
                # Add labels and title
                ax_tradeoff.set_xlabel('Accuracy Score')
                ax_tradeoff.set_ylabel('Beyond-Accuracy Score')
                ax_tradeoff.set_title('Accuracy vs. Beyond-Accuracy Tradeoff')
                
                # Add grid
                ax_tradeoff.grid(True, alpha=0.3)
        
        # 4. Temporal analysis
        ax_temporal = plt.subplot(gs[2, 0])
        
        # Get temporal data
        if 'temporal' in all_results and not all_results['temporal'].empty:
            temporal_df = all_results['temporal']
            
            if 'mechanism' in temporal_df.columns and 'hit_rate_stability' in temporal_df.columns:
                # Sort by stability
                temp_df = temporal_df.sort_values('hit_rate_stability', ascending=False)
                
                # Create bar chart
                bars = ax_temporal.bar(temp_df['mechanism'], temp_df['hit_rate_stability'])
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax_temporal.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                
                # Add labels and title
                ax_temporal.set_ylabel('Hit Rate Stability')
                ax_temporal.set_title('Mechanisms by Temporal Stability')
                
                # Rotate x-axis labels
                ax_temporal.set_xticklabels(temp_df['mechanism'], rotation=45, ha='right')
                
                # Add grid
                ax_temporal.grid(True, alpha=0.3, axis='y')
        
        # 5. Segment analysis
        ax_segment = plt.subplot(gs[2, 1])
        
        # Get segment data
        if 'segments' in all_results and not all_results['segments'].empty:
            segment_df = all_results['segments']
            
            if 'segment' in segment_df.columns and 'mechanism' in segment_df.columns and 'hit_rate@10' in segment_df.columns:
                # Create pivot table
                pivot_df = segment_df.pivot_table(
                    index='segment',
                    columns='mechanism',
                    values='hit_rate@10',
                    aggfunc='mean'
                )
                
                # Create heatmap
                sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax_segment)
                
                # Add title
                ax_segment.set_title('Hit Rate by Segment and Mechanism')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    