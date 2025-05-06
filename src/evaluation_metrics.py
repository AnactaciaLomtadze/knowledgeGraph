#src/evaluation_metrics.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
from tqdm import tqdm
import math

class EvaluationMetrics:
    """
    Evaluation metrics for assessing the performance of forgetting mechanisms.
    
    This class provides a comprehensive set of metrics for evaluating how
    forgetting mechanisms affect recommendation quality, diversity, and other
    important aspects of the system.
    """
    def __init__(self, knowledge_graph, forgetting_mechanism=None):
        """
        Initialize the evaluation metrics for the forgetting mechanism.
        
        Args:
            knowledge_graph: The MovieLensKnowledgeGraph instance
            forgetting_mechanism: Optional ForgettingMechanism instance
        """
        self.kg = knowledge_graph
        self.fm = forgetting_mechanism
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('EvaluationMetrics')
    
    def measure_recommendation_diversity_after_forgetting(self, recommendations_before, recommendations_after):
        """
        Measure how diverse recommendations become after applying forgetting.
        
        Args:
            recommendations_before: List of movie IDs before forgetting
            recommendations_after: List of movie IDs after forgetting
            
        Returns:
            Dictionary with diversity metrics
        """
        # Compute genre diversity
        genre_vectors_before = []
        genre_vectors_after = []
        
        for movie_id in recommendations_before:
            if movie_id in self.kg.movie_features:
                genre_vectors_before.append(self.kg.movie_features[movie_id])
        
        for movie_id in recommendations_after:
            if movie_id in self.kg.movie_features:
                genre_vectors_after.append(self.kg.movie_features[movie_id])
        
        # If no valid movies, return defaults
        if not genre_vectors_before or not genre_vectors_after:
            return {
                'genre_diversity_before': 0,
                'genre_diversity_after': 0,
                'jaccard_similarity': 0,
                'new_item_percentage': 0
            }
        
        # Calculate genre diversity as average pairwise distance
        sim_matrix_before = cosine_similarity(genre_vectors_before)
        sim_matrix_after = cosine_similarity(genre_vectors_after)
        
        # Set diagonal to 0 to exclude self-similarity
        np.fill_diagonal(sim_matrix_before, 0)
        np.fill_diagonal(sim_matrix_after, 0)
        
        genre_diversity_before = 1 - np.mean(sim_matrix_before)
        genre_diversity_after = 1 - np.mean(sim_matrix_after)
        
        # Calculate Jaccard similarity between recommendation sets
        set_before = set(recommendations_before)
        set_after = set(recommendations_after)
        
        jaccard_similarity = len(set_before.intersection(set_after)) / len(set_before.union(set_after))
        
        # Calculate percentage of new items
        new_items = [item for item in recommendations_after if item not in recommendations_before]
        new_item_percentage = len(new_items) / len(recommendations_after)
        
        return {
            'genre_diversity_before': genre_diversity_before,
            'genre_diversity_after': genre_diversity_after,
            'jaccard_similarity': jaccard_similarity,
            'new_item_percentage': new_item_percentage
        }
    
    def calculate_temporal_relevance_score(self, recommendations, user_current_interests):
        """
        Calculate how well recommendations align with current user interests.
        
        Args:
            recommendations: List of movie IDs
            user_current_interests: Dict mapping genre indices to interest scores
            
        Returns:
            Average temporal relevance score
        """
        if not recommendations or not user_current_interests:
            return 0.0
        
        relevance_scores = []
        
        for movie_id in recommendations:
            if movie_id in self.kg.movie_features:
                genre_vector = self.kg.movie_features[movie_id]
                
                # Calculate relevance as dot product of genre vector and interest vector
                interest_vector = np.zeros(len(genre_vector))
                for genre_idx, score in user_current_interests.items():
                    if 0 <= genre_idx < len(interest_vector):
                        interest_vector[genre_idx] = score
                
                relevance = np.dot(genre_vector, interest_vector) / (np.sum(genre_vector) + 1e-10)
                relevance_scores.append(relevance)
        
        if not relevance_scores:
            return 0.0
        
        return np.mean(relevance_scores)
    
    def evaluate_catastrophic_forgetting_impact(self, model_performance_timeline):
        """
        Evaluate if forgetting causes catastrophic loss of important information.
        
        Args:
            model_performance_timeline: List of (time_point, performance_metric) tuples
            
        Returns:
            Dictionary with forgetting impact metrics
        """
        if not model_performance_timeline or len(model_performance_timeline) < 2:
            return {
                'max_performance_drop': 0,
                'stability_score': 1.0,
                'catastrophic_forgetting_detected': False
            }
        
        # Extract time points and performance values
        time_points, performance_values = zip(*model_performance_timeline)
        
        # Calculate performance changes
        performance_changes = [performance_values[i] - performance_values[i-1] 
                              for i in range(1, len(performance_values))]
        
        max_drop = min(0, min(performance_changes)) if performance_changes else 0
        
        # Calculate stability as inverse of standard deviation of performance
        stability = 1.0 / (np.std(performance_values) + 1e-10)
        
        # Detect catastrophic forgetting (sudden large drop)
        threshold = -0.2  # 20% drop is considered catastrophic
        catastrophic = any(change < threshold for change in performance_changes)
        
        return {
            'max_performance_drop': max_drop,
            'stability_score': stability,
            'catastrophic_forgetting_detected': catastrophic
        }
    
    def compute_memory_efficiency_metrics(self, graph_size_before, graph_size_after):
        """
        Compute metrics for memory efficiency after forgetting.
        
        Args:
            graph_size_before: Tuple of (nodes, edges) counts before forgetting
            graph_size_after: Tuple of (nodes, edges) counts after forgetting
            
        Returns:
            Dictionary with memory efficiency metrics
        """
        nodes_before, edges_before = graph_size_before
        nodes_after, edges_after = graph_size_after
        
        if nodes_before == 0 or edges_before == 0:
            return {
                'node_reduction_ratio': 0,
                'edge_reduction_ratio': 0,
                'memory_efficiency_gain': 0
            }
        
        node_reduction = (nodes_before - nodes_after) / nodes_before
        edge_reduction = (edges_before - edges_after) / edges_before
        
        # Memory efficiency is approximated as weighted sum of node and edge reduction
        # Edges typically consume more memory in graph representations
        memory_efficiency = 0.3 * node_reduction + 0.7 * edge_reduction
        
        return {
            'node_reduction_ratio': node_reduction,
            'edge_reduction_ratio': edge_reduction,
            'memory_efficiency_gain': memory_efficiency
        }
    
    def calculate_hit_rate_at_k(self, test_set, recommendations, k=10):
        """
        Calculate Hit Rate@K, which measures if a relevant item appears in top-K recommendations.
        
        Args:
            test_set: Set of movie IDs that are relevant (e.g., movies the user actually watched)
            recommendations: List of recommended movie IDs
            k: Number of top recommendations to consider
            
        Returns:
            1 if at least one item in the test set appears in top-k recommendations, 0 otherwise
        """
        if not test_set or not recommendations:
            return 0.0
        
        # Consider only top-k recommendations
        top_k_recommendations = recommendations[:k]
        
        # Check if at least one test item is in the recommendations
        for item in test_set:
            if item in top_k_recommendations:
                return 1.0
        
        return 0.0
    
    def calculate_precision_at_k(self, test_set, recommendations, k=10):
        """
        Calculate Precision@K, which is the proportion of recommended items that are relevant.
        
        Args:
            test_set: Set of movie IDs that are relevant (e.g., movies the user actually watched)
            recommendations: List of recommended movie IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score between 0 and 1
        """
        if not test_set or not recommendations:
            return 0.0
        
        # Consider only top-k recommendations
        top_k_recommendations = recommendations[:k]
        
        # Count relevant items in the recommendations
        relevant_items = [item for item in top_k_recommendations if item in test_set]
        
        return len(relevant_items) / min(k, len(top_k_recommendations))

    def calculate_recall_at_k(self, test_set, recommendations, k=10):
        """
        Calculate Recall@K, which is the proportion of relevant items that are recommended.

        Args:
            test_set: Set of movie IDs that are relevant (e.g., movies the user actually watched)
            recommendations: List of recommended movie IDs
            k: Number of top recommendations to consider

        Returns:
            Recall@K score between 0 and 1
        """
        if not test_set or not recommendations:
            return 0.0

        # Consider only top-k recommendations
        top_k_recommendations = recommendations[:k]

        # Count relevant items in the recommendations
        relevant_items = [item for item in top_k_recommendations if item in test_set]

        return len(relevant_items) / len(test_set) if len(test_set) > 0 else 0.0

    def calculate_ndcg_at_k(self, test_set, recommendations, k=10):
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) at K.
        NDCG takes into account the position of relevant items in the recommendation list.
        
        Args:
            test_set: Set of movie IDs that are relevant
            recommendations: List of recommended movie IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score between 0 and 1
        """
        if not test_set or not recommendations:
            return 0.0
        
        # Consider only top-k recommendations
        top_k_recommendations = recommendations[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recommendations):
            if item in test_set:
                # Use log base 2 as is standard for NDCG
                # Add 2 to i because i is 0-indexed and the formula uses 1-indexed positions
                dcg += 1.0 / math.log2(i + 2)
        
        # Calculate ideal DCG (IDCG)
        # In the ideal case, all relevant items appear at the top of the list
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
        Calculate Mean Reciprocal Rank (MRR), which is the average of reciprocal ranks
        of the first relevant item in the recommendations.

        Args:
            test_set: Set of movie IDs that are relevant (e.g., movies the user actually watched)
            recommendations: List of recommended movie IDs

        Returns:
            MRR score between 0 and 1
        """
        if not test_set or not recommendations:
            return 0.0

        # Find the rank of the first relevant item
        for i, item in enumerate(recommendations):
            if item in test_set:
                # +1 because ranks start from 1, not 0
                return 1.0 / (i + 1)

        return 0.0
    
    def calculate_serendipity(self, user_id, recommendations, k=10, popularity_threshold=0.8):
        """
        Calculate serendipity of recommendations.
        Serendipity measures how "surprising" yet relevant the recommendations are.
        
        Args:
            user_id: The user ID
            recommendations: List of recommended movie IDs
            k: Number of recommendations to consider
            popularity_threshold: Threshold to determine popular items (percentile)
            
        Returns:
            Serendipity score between 0 and 1
        """
        if not recommendations:
            return 0.0
            
        # Consider only top-k recommendations
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
            # Default to uniform preferences if user not found
            user_genres = np.ones(19) / 19
        
        serendipity_scores = []
        
        for movie_id in top_k_recommendations:
            # Calculate unexpectedness based on popularity
            if movie_id in movie_popularity:
                popularity = movie_popularity[movie_id]
                unexpectedness = 1.0 - min(1.0, popularity / popularity_cutoff)
            else:
                unexpectedness = 0.5  # Default if movie not found
            
            # Calculate relevance based on genre match
            if movie_id in self.kg.movie_features:
                movie_genres = self.kg.movie_features[movie_id]
                # Normalize user genres to non-negative values for relevance calculation
                norm_user_genres = user_genres - np.min(user_genres)
                if np.max(norm_user_genres) > 0:
                    norm_user_genres = norm_user_genres / np.max(norm_user_genres)
                relevance = np.sum(norm_user_genres * movie_genres) / (np.sum(movie_genres) + 1e-10)
            else:
                relevance = 0.0  # Default if movie not found
            
            # Serendipity is a product of unexpectedness and relevance
            serendipity = unexpectedness * relevance
            serendipity_scores.append(serendipity)
        
        # Average serendipity across all recommendations
        if not serendipity_scores:
            return 0.0
            
        return np.mean(serendipity_scores)
    
    def calculate_novelty(self, recommendations, k=10):
        """
        Calculate novelty of recommendations.
        Novelty measures how unusual or unknown the recommended items are in general.
        
        Args:
            recommendations: List of recommended movie IDs
            k: Number of recommendations to consider
            
        Returns:
            Novelty score between 0 and 1
        """
        if not recommendations:
            return 0.0
            
        # Consider only top-k recommendations
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
                novelty_scores.append(0.5)  # Default if movie not found
        
        if not novelty_scores:
            return 0.0
            
        return np.mean(novelty_scores)
    
    def calculate_coverage(self, recommendations_by_user, all_items=None):
        """
        Calculate catalog coverage.
        Coverage measures what percentage of items are ever recommended.
        
        Args:
            recommendations_by_user: Dict mapping user_id to list of recommendations
            all_items: Set of all available items. If None, use all movies in the dataset.
            
        Returns:
            Coverage percentage between 0 and 100
        """
        if not recommendations_by_user:
            return 0.0
            
        # Get all recommended items across all users
        recommended_items = set()
        for user_id, recs in recommendations_by_user.items():
            recommended_items.update(recs)
        
        # Get all available items if not provided
        if all_items is None:
            all_items = set(self.kg.movie_features.keys())
        
        # Calculate coverage
        if not all_items:
            return 0.0
            
        return 100.0 * len(recommended_items) / len(all_items)
    
    def evaluate_ranking_metrics(self, user_id, test_set, recommendations_before, recommendations_after, k_values=[5, 10, 20]):
        """
        Evaluate ranking-based metrics for recommendations before and after forgetting.

        Args:
            user_id: The user ID
            test_set: Set of movie IDs that are relevant (e.g., movies the user actually watched)
            recommendations_before: List of recommended movie IDs before forgetting
            recommendations_after: List of recommended movie IDs after forgetting
            k_values: List of k values to calculate metrics for

        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'user_id': user_id,
            'before': {},
            'after': {}
        }

        # Calculate metrics for each k value
        for k in k_values:
            # Before forgetting
            metrics['before'][f'hit_rate@{k}'] = self.calculate_hit_rate_at_k(test_set, recommendations_before, k)
            metrics['before'][f'precision@{k}'] = self.calculate_precision_at_k(test_set, recommendations_before, k)
            metrics['before'][f'recall@{k}'] = self.calculate_recall_at_k(test_set, recommendations_before, k)
            metrics['before'][f'ndcg@{k}'] = self.calculate_ndcg_at_k(test_set, recommendations_before, k)

            # After forgetting
            metrics['after'][f'hit_rate@{k}'] = self.calculate_hit_rate_at_k(test_set, recommendations_after, k)
            metrics['after'][f'precision@{k}'] = self.calculate_precision_at_k(test_set, recommendations_after, k)
            metrics['after'][f'recall@{k}'] = self.calculate_recall_at_k(test_set, recommendations_after, k)
            metrics['after'][f'ndcg@{k}'] = self.calculate_ndcg_at_k(test_set, recommendations_after, k)

        # Calculate MRR
        metrics['before']['mrr'] = self.calculate_mrr(test_set, recommendations_before)
        metrics['after']['mrr'] = self.calculate_mrr(test_set, recommendations_after)
        
        # Calculate serendipity
        metrics['before']['serendipity'] = self.calculate_serendipity(user_id, recommendations_before)
        metrics['after']['serendipity'] = self.calculate_serendipity(user_id, recommendations_after)
        
        # Calculate novelty
        metrics['before']['novelty'] = self.calculate_novelty(recommendations_before)
        metrics['after']['novelty'] = self.calculate_novelty(recommendations_after)
        
        # Calculate diversity
        diversity_metrics = self.measure_recommendation_diversity_after_forgetting(
            recommendations_before, recommendations_after)
        
        metrics['before']['diversity'] = diversity_metrics['genre_diversity_before']
        metrics['after']['diversity'] = diversity_metrics['genre_diversity_after']
        metrics['similarity'] = diversity_metrics['jaccard_similarity']
        metrics['new_item_percentage'] = diversity_metrics['new_item_percentage']

        return metrics

    def evaluate_forgetting_impact_on_multiple_users(self, user_ids, test_data, get_recommendations_fn, forgetting_fn, k_values=[5, 10, 20]):
        """
        Evaluate the impact of forgetting on recommendations for multiple users.

        Args:
            user_ids: List of user IDs to evaluate
            test_data: Dictionary mapping user_id to set of relevant movie IDs (test set)
            get_recommendations_fn: Function that returns recommendations for a user
            forgetting_fn: Function that applies forgetting mechanism for a user
            k_values: List of k values to calculate metrics for

        Returns:
            DataFrame with evaluation metrics for all users
        """
        all_metrics = []

        for user_id in tqdm(user_ids, desc="Evaluating users"):
            if user_id not in test_data:
                continue

            # Get recommendations before forgetting
            recommendations_before = get_recommendations_fn(user_id)

            # Apply forgetting mechanism
            forgetting_fn(user_id)

            # Get recommendations after forgetting
            recommendations_after = get_recommendations_fn(user_id)

            # Evaluate metrics
            metrics = self.evaluate_ranking_metrics(
                user_id, 
                test_data[user_id],
                recommendations_before,
                recommendations_after,
                k_values
            )

            all_metrics.append(metrics)

        # Convert to DataFrame for easier analysis
        metrics_df = pd.DataFrame()

        for metrics in all_metrics:
            user_id = metrics['user_id']
            user_row = {'user_id': user_id}

            # Flatten the metrics dictionary
            for condition in ['before', 'after']:
                for metric_name, value in metrics[condition].items():
                    user_row[f'{condition}_{metric_name}'] = value

            # Add similarity metrics
            if 'similarity' in metrics:
                user_row['jaccard_similarity'] = metrics['similarity']
            if 'new_item_percentage' in metrics:
                user_row['new_item_percentage'] = metrics['new_item_percentage']

            # Calculate differences
            for k in k_values:
                for metric in [f'hit_rate@{k}', f'precision@{k}', f'recall@{k}', f'ndcg@{k}']:
                    before_key = f'before_{metric}'
                    after_key = f'after_{metric}'
                    if before_key in user_row and after_key in user_row:
                        user_row[f'diff_{metric}'] = user_row[after_key] - user_row[before_key]

            # Calculate differences for other metrics
            for metric in ['mrr', 'serendipity', 'novelty', 'diversity']:
                before_key = f'before_{metric}'
                after_key = f'after_{metric}'
                if before_key in user_row and after_key in user_row:
                    user_row[f'diff_{metric}'] = user_row[after_key] - user_row[before_key]

            # Append to DataFrame
            metrics_df = pd.concat([metrics_df, pd.DataFrame([user_row])], ignore_index=True)

        return metrics_df

    def visualize_ranking_metrics(self, metrics_df, metric_name='hit_rate@10'):
        """
        Visualize the impact of forgetting on ranking metrics.

        Args:
            metrics_df: DataFrame with evaluation metrics from evaluate_forgetting_impact_on_multiple_users
            metric_name: The metric to visualize (without 'before_' or 'after_' prefix)

        Returns:
            None (displays plots)
        """
        # Set up the figure
        plt.figure(figsize=(15, 6))

        # Plot 1: Before vs After
        plt.subplot(1, 2, 1)
        before_key = f'before_{metric_name}'
        after_key = f'after_{metric_name}'

        # Get values
        before_values = metrics_df[before_key].values
        after_values = metrics_df[after_key].values

        # Create dataframe for seaborn
        plot_df = pd.DataFrame({
            'User': np.repeat(metrics_df['user_id'].values, 2),
            'Condition': ['Before'] * len(metrics_df) + ['After'] * len(metrics_df),
            'Value': np.concatenate([before_values, after_values])
        })

        # Plot
        sns.boxplot(x='Condition', y='Value', data=plot_df)
        plt.title(f'Impact of Forgetting on {metric_name}')
        plt.ylim(0, 1)

        # Plot 2: Differences
        plt.subplot(1, 2, 2)
        diff_key = f'diff_{metric_name}'

        # Sort users by difference
        sorted_df = metrics_df.sort_values(by=diff_key)

        # Plot
        plt.bar(range(len(sorted_df)), sorted_df[diff_key].values)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(f'Difference in {metric_name} After Forgetting')
        plt.xlabel('User (sorted by difference)')
        plt.ylabel('Difference (After - Before)')

        plt.tight_layout()
        plt.show()

        # Summary statistics
        avg_before = metrics_df[before_key].mean()
        avg_after = metrics_df[after_key].mean()
        avg_diff = metrics_df[diff_key].mean()

        print(f"Average {metric_name} before forgetting: {avg_before:.4f}")
        print(f"Average {metric_name} after forgetting: {avg_after:.4f}")
        print(f"Average difference: {avg_diff:.4f}")
        print(f"Users with improved {metric_name}: {sum(metrics_df[diff_key] > 0)} out of {len(metrics_df)}")
    
    def visualize_forgetting_impact_over_time(self, user_id, time_points=5, metrics=['hit_rate@10', 'diversity', 'serendipity']):
        """
        Visualize how forgetting impacts recommendation quality over time.
        
        Args:
            user_id: The user ID to analyze
            time_points: Number of time points to sample
            metrics: List of metrics to track
        """
        if self.fm is None:
            self.logger.error("Forgetting mechanism not provided. Cannot visualize impact over time.")
            return
        
        # Ensure user has a test set
        if user_id not in self.kg.user_profiles:
            self.logger.error(f"User {user_id} not found in user profiles.")
            return
        
        # Create a simple test set from user's rated movies (for demonstration)
        rated_movies = list(self.kg.user_profiles[user_id]['rated_movies'])
        if len(rated_movies) < 2:
            self.logger.error(f"User {user_id} has too few rated movies for meaningful analysis.")
            return
            
        # Use 20% of rated movies as test set
        np.random.seed(42)  # For reproducibility
        test_size = max(1, int(len(rated_movies) * 0.2))
        test_indices = np.random.choice(len(rated_movies), test_size, replace=False)
        test_set = set([rated_movies[i] for i in test_indices])
        
        # Save original memory strengths for this user
        original_strengths = {}
        for key, value in self.fm.memory_strength.items():
            if key[0] == user_id:
                original_strengths[key] = value
        
        # Create time points for analysis
        current_time = datetime.datetime.now().timestamp()
        time_deltas = [0, 7, 30, 90, 180]  # Days: now, 1 week, 1 month, 3 months, 6 months
        if time_points > len(time_deltas):
            # Generate intermediate time points
            time_deltas = np.linspace(0, 180, time_points)
        else:
            time_deltas = time_deltas[:time_points]
        
        # Track metrics over time
        results = []
        
        for days in tqdm(time_deltas, desc="Simulating time progression"):
            # Reset memory strengths to original values
            for key, value in original_strengths.items():
                self.fm.memory_strength[key] = value
            
            # Set last interaction time to simulate passage of time
            for key in original_strengths:
                if key in self.fm.last_interaction_time:
                    self.fm.last_interaction_time[key] = current_time - (days * 24 * 60 * 60)
            
            # Apply forgetting mechanism
            params = self.fm.personalize_forgetting_parameters(user_id)
            self.fm.create_hybrid_decay_function(
                user_id, 
                time_weight=params['time_weight'],
                usage_weight=params['usage_weight'],
                novelty_weight=params['novelty_weight']
            )
            
            # Get recommendations
            recommendations = self.kg.get_recommendations(user_id, method='hybrid')
            
            # Calculate metrics
            metric_values = {}
            
            for metric in metrics:
                if metric.startswith('hit_rate@'):
                    k = int(metric.split('@')[1])
                    metric_values[metric] = self.calculate_hit_rate_at_k(test_set, recommendations, k)
                elif metric.startswith('precision@'):
                    k = int(metric.split('@')[1])
                    metric_values[metric] = self.calculate_precision_at_k(test_set, recommendations, k)
                elif metric.startswith('recall@'):
                    k = int(metric.split('@')[1])
                    metric_values[metric] = self.calculate_recall_at_k(test_set, recommendations, k)
                elif metric.startswith('ndcg@'):
                    k = int(metric.split('@')[1])
                    metric_values[metric] = self.calculate_ndcg_at_k(test_set, recommendations, k)
                elif metric == 'mrr':
                    metric_values[metric] = self.calculate_mrr(test_set, recommendations)
                elif metric == 'serendipity':
                    metric_values[metric] = self.calculate_serendipity(user_id, recommendations)
                elif metric == 'novelty':
                    metric_values[metric] = self.calculate_novelty(recommendations)
                elif metric == 'diversity':
                    # If we don't have before recommendations, calculate diversity directly
                    genre_vectors = []
                    for movie_id in recommendations:
                        if movie_id in self.kg.movie_features:
                            genre_vectors.append(self.kg.movie_features[movie_id])
                    
                    if genre_vectors:
                        sim_matrix = cosine_similarity(genre_vectors)
                        np.fill_diagonal(sim_matrix, 0)
                        metric_values[metric] = 1 - np.mean(sim_matrix)
                    else:
                        metric_values[metric] = 0.0
            
            # Track memory strengths
            avg_memory_strength = np.mean([value for key, value in self.fm.memory_strength.items() if key[0] == user_id])
            
            results.append({
                'days': days,
                'avg_memory_strength': avg_memory_strength,
                **metric_values
            })
        
        # Restore original memory strengths
        for key, value in original_strengths.items():
            self.fm.memory_strength[key] = value
        
        # Create visualization
        results_df = pd.DataFrame(results)
        
        # Plot metrics over time
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Memory Strength
        plt.subplot(2, 1, 1)
        plt.plot(results_df['days'], results_df['avg_memory_strength'], 'o-', linewidth=2)
        plt.title(f'Average Memory Strength Decay Over Time for User {user_id}')
        plt.xlabel('Days')
        plt.ylabel('Average Memory Strength')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Metrics
        plt.subplot(2, 1, 2)
        for metric in metrics:
            if metric in results_df.columns:
                plt.plot(results_df['days'], results_df[metric], 'o-', linewidth=2, label=metric)
        
        plt.title(f'Impact of Forgetting on Recommendation Metrics Over Time')
        plt.xlabel('Days')
        plt.ylabel('Metric Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    def compare_with_traditional_recommenders(self, user_ids, test_data, traditional_recommenders, forgetting_recommenders):
        """
        Compare forgetting-based recommendations with traditional approaches.
        
        Args:
            user_ids: List of user IDs to evaluate
            test_data: Dictionary of test movie IDs
            traditional_recommenders: Dict mapping name to recommendation function
            forgetting_recommenders: Dict mapping name to recommendation function
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        # Evaluate all methods
        all_recommenders = {**traditional_recommenders, **forgetting_recommenders}
        
        for user_id in tqdm(user_ids, desc="Evaluating users"):
            if user_id not in test_data or not test_data[user_id]:
                continue
                
            test_set = test_data[user_id]
            
            for name, recommender in all_recommenders.items():
                try:
                    recommendations = recommender(user_id)
                    
                    # Calculate key metrics
                    metrics = {
                        'hit_rate@10': self.calculate_hit_rate_at_k(test_set, recommendations, 10),
                        'precision@10': self.calculate_precision_at_k(test_set, recommendations, 10),
                        'recall@10': self.calculate_recall_at_k(test_set, recommendations, 10),
                        'ndcg@10': self.calculate_ndcg_at_k(test_set, recommendations, 10),
                        'mrr': self.calculate_mrr(test_set, recommendations),
                        'serendipity': self.calculate_serendipity(user_id, recommendations),
                        'novelty': self.calculate_novelty(recommendations)
                    }
                    
                    results.append({
                        'user_id': user_id,
                        'recommender': name,
                        'category': 'Traditional' if name in traditional_recommenders else 'Forgetting',
                        **metrics
                    })
                except Exception as e:
                    self.logger.error(f"Error evaluating {name} for user {user_id}: {e}")
        
        return pd.DataFrame(results)