#src/experiment_runner.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import datetime 
import os
from tqdm import tqdm

from src.knowledge_graph import MovieLensKnowledgeGraph
from src.forgetting_mechanism import ForgettingMechanism
from src.evaluation_metrics import EvaluationMetrics
from src.visualization_tool import VisualizationTool

class ForgettingConfig:
    """Configuration class for forgetting mechanism experiments."""
    def __init__(self, **kwargs):
        self.data_path = kwargs.get('data_path', './ml-100k')
        self.decay_parameters = kwargs.get('decay_parameters', {
            'time_decay_rate': 0.1,
            'usage_threshold': 3,
            'time_weight': 0.4,
            'usage_weight': 0.3,
            'novelty_weight': 0.3,
            'forgetting_factor': 0.5
        })
        self.evaluation_metrics = kwargs.get('evaluation_metrics', 
                                          ['hit_rate', 'precision', 'recall', 'mrr', 'ndcg', 'serendipity', 'novelty', 'diversity'])
        self.k_values = kwargs.get('k_values', [5, 10, 20])
        self.test_ratio = kwargs.get('test_ratio', 0.2)
        self.seed = kwargs.get('seed', 42)
        self.output_dir = kwargs.get('output_dir', './results')
        self.num_users = kwargs.get('num_users', 50)  # Number of users to evaluate
        self.temporal_split = kwargs.get('temporal_split', True)  # Use temporal split instead of random
        self.test_days = kwargs.get('test_days', 30)  # Number of days at the end to use for testing
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
class ForgettingExperimentRunner:
    """Runs experiments with different forgetting mechanisms."""
    def __init__(self, config):
        """
        Initialize the experiment runner.
        
        Args:
            config: ForgettingConfig instance
        """
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ExperimentRunner')
        
        # Initialize components
        self.kg = None
        self.fm = None
        self.evaluator = None
        self.visualizer = None
        
        # Load data and initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize knowledge graph, forgetting mechanism, and evaluation components."""
        self.logger.info("Initializing knowledge graph...")
        self.kg = MovieLensKnowledgeGraph(data_path=self.config.data_path)
        
        if not self.kg.load_data():
            self.logger.error("Failed to load data. Exiting.")
            return False
            
        if not self.kg.build_knowledge_graph():
            self.logger.error("Failed to build knowledge graph. Exiting.")
            return False
        
        self.logger.info("Initializing forgetting mechanism...")
        self.fm = ForgettingMechanism(self.kg)
        
        self.logger.info("Initializing evaluation metrics...")
        self.evaluator = EvaluationMetrics(self.kg, self.fm)
        
        self.logger.info("Initializing visualization tool...")
        self.visualizer = VisualizationTool(self.kg, self.fm)
        
        self.logger.info("Initialization complete.")
        return True
    
    def prepare_train_test_data(self):
        """Prepare train-test splits for evaluation."""
        if self.config.temporal_split:
            self.logger.info(f"Creating temporal train-test split with {self.config.test_days} test days...")
            self.test_data = self.kg.create_temporal_train_test_split(test_days=self.config.test_days)
        else:
            # Use random split for traditional evaluation
            self.logger.info(f"Creating random train-test split with {self.config.test_ratio} test ratio...")
            np.random.seed(self.config.seed)
            
            # Get all users with enough ratings
            all_users = []
            for user_id in self.kg.ratings_df['user_id'].unique():
                user_ratings = self.kg.ratings_df[self.kg.ratings_df['user_id'] == user_id]
                if len(user_ratings) >= 5:  # Only include users with at least 5 ratings
                    all_users.append(user_id)
            
            self.test_data = {}
            
            # Process each user
            for user_id in all_users:
                user_ratings = self.kg.ratings_df[self.kg.ratings_df['user_id'] == user_id]
                
                # Split ratings into train and test
                test_size = max(1, int(len(user_ratings) * self.config.test_ratio))
                test_indices = np.random.choice(user_ratings.index, test_size, replace=False)
                
                user_test = user_ratings.loc[test_indices]
                user_train = user_ratings.drop(test_indices)
                
                # Store test movie IDs
                self.test_data[user_id] = set(user_test['movie_id'].values)
                
                # Update ratings
                # Remove test ratings from training set
                self.kg.ratings_df = self.kg.ratings_df.drop(test_indices)
            
            # Rebuild user profiles with updated ratings
            self.kg._build_user_profiles()
        
        # Get a sample of users for evaluation
        all_test_users = list(self.test_data.keys())
        
        if len(all_test_users) <= self.config.num_users:
            self.test_users = all_test_users
        else:
            self.test_users = np.random.choice(all_test_users, self.config.num_users, replace=False)
        
        self.logger.info(f"Prepared test data for {len(self.test_users)} users")
        return self.test_data, self.test_users
        
    def run_baseline_comparison(self):
        """
        Run a baseline comparison experiment.
        
        Compare forgetting-based recommendations with traditional approaches.
        
        Returns:
            DataFrame with results
        """
        self.logger.info("Running baseline comparison experiment...")
        
        # Prepare test data
        test_data, test_users = self.prepare_train_test_data()
        
        # Define traditional recommenders
        traditional_recommenders = {
            'Popular': lambda u: self._recommend_popular(u),
            'Content-Based': lambda u: self.kg.get_personalized_recommendations(u),
            'Graph-Based': lambda u: self.kg.get_graph_based_recommendations(u),
            'Hybrid': lambda u: self.kg.get_hybrid_recommendations(u)
        }
        
        # Define forgetting recommenders
        forgetting_recommenders = {
            'Time-Decay': lambda u: self._recommend_with_time_decay(u),
            'Ebbinghaus': lambda u: self._recommend_with_ebbinghaus(u),
            'Power-Law': lambda u: self._recommend_with_power_law(u),
            'Usage-Based': lambda u: self._recommend_with_usage_based(u),
            'Hybrid-Decay': lambda u: self._recommend_with_hybrid_decay(u),
            'Personalized': lambda u: self._recommend_with_personalized_decay(u)
        }
        
        # Run comparison
        results = self.evaluator.compare_with_traditional_recommenders(
            test_users, test_data, traditional_recommenders, forgetting_recommenders)
        
        # Save results
        results_path = os.path.join(self.config.output_dir, 'baseline_comparison_results.csv')
        results.to_csv(results_path, index=False)
        
        # Visualize results
        self.visualizer.visualize_comparative_evaluation(
            results, 
            metrics=['hit_rate@10', 'precision@10', 'ndcg@10', 'mrr', 'serendipity', 'novelty'],
            filename=os.path.join(self.config.output_dir, 'baseline_comparison.png')
        )
        
        self.logger.info(f"Baseline comparison complete. Results saved to {results_path}")
        return results
    
    def run_temporal_evaluation(self):
        """
        Run a temporal evaluation experiment.
        
        Investigate how recommendations change over time with forgetting.
        
        Returns:
            Dictionary with results for each user
        """
        self.logger.info("Running temporal evaluation experiment...")
        
        # Prepare test data
        test_data, test_users = self.prepare_train_test_data()
        
        # Sample a small number of users for detailed temporal analysis
        if len(test_users) > 5:
            sample_users = np.random.choice(test_users, 5, replace=False)
        else:
            sample_users = test_users
            
        temporal_results = {}
        
        for user_id in tqdm(sample_users, desc="Evaluating users"):
            self.logger.info(f"Running temporal evaluation for user {user_id}...")
            
            metrics_over_time = self.visualizer.visualize_temporal_dimension(
                user_id,
                metrics=['hit_rate', 'diversity', 'novelty', 'serendipity'],
                time_points=8,
                filename=os.path.join(self.config.output_dir, f'temporal_evaluation_user_{user_id}.png')
            )
            
            temporal_results[user_id] = metrics_over_time
            
            # Also visualize memory strength decay
            self.visualizer.visualize_forgetting_impact(
                user_id,
                time_period=180,  # 6 months
                visualization_type='memory_strength',
                filename=os.path.join(self.config.output_dir, f'memory_decay_user_{user_id}.png')
            )
            
            # Visualize recommendation changes
            self.visualizer.visualize_forgetting_impact(
                user_id,
                time_period=180,  # 6 months
                visualization_type='recommendations',
                filename=os.path.join(self.config.output_dir, f'recommendation_changes_user_{user_id}')
            )
        
        self.logger.info(f"Temporal evaluation complete for {len(sample_users)} users")
        return temporal_results
    
    def run_parameter_sensitivity(self):
        """
        Run a parameter sensitivity experiment.
        
        Investigate how different parameter settings affect recommendation quality.
        
        Returns:
            DataFrame with results
        """
        self.logger.info("Running parameter sensitivity experiment...")
        
        # Prepare test data
        test_data, test_users = self.prepare_train_test_data()
        
        # Sample a subset of users for efficiency
        if len(test_users) > 20:
            sample_users = np.random.choice(test_users, 20, replace=False)
        else:
            sample_users = test_users
        
        # Define parameter ranges to test
        time_weights = [0.2, 0.4, 0.6, 0.8]
        usage_weights = [0.2, 0.4, 0.6, 0.8]
        decay_rates = [0.05, 0.1, 0.2, 0.3]
        forgetting_factors = [0.3, 0.5, 0.7, 0.9]
        
        # Track results
        parameter_results = []
        
        # Test different time weights
        self.logger.info("Testing different time weights...")
        for time_weight in tqdm(time_weights, desc="Time weights"):
            # Ensure weights sum to 1
            usage_weight = (1.0 - time_weight) / 2
            novelty_weight = usage_weight
            
            params = {
                'time_weight': time_weight,
                'usage_weight': usage_weight,
                'novelty_weight': novelty_weight,
                'forgetting_factor': 0.5  # Fixed for this test
            }
            
            # Evaluate with these parameters
            for user_id in sample_users:
                # Get original recommendations
                original_recs = self.kg.get_recommendations(user_id, method='hybrid')
                
                # Apply forgetting with these parameters
                self.fm.create_hybrid_decay_function(
                    user_id,
                    time_weight=params['time_weight'],
                    usage_weight=params['usage_weight'],
                    novelty_weight=params['novelty_weight']
                )
                
                # Get forgetting-aware recommendations
                forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
                    'hybrid', params
                )
                forgetting_recs = forgetting_rec_fn(user_id)
                
                # Evaluate
                if user_id in test_data:
                    hit_rate = self.evaluator.calculate_hit_rate_at_k(test_data[user_id], forgetting_recs, 10)
                    precision = self.evaluator.calculate_precision_at_k(test_data[user_id], forgetting_recs, 10)
                    diversity = self.evaluator.measure_recommendation_diversity_after_forgetting(
                        original_recs, forgetting_recs)['genre_diversity_after']
                    novelty = self.evaluator.calculate_novelty(forgetting_recs)
                    serendipity = self.evaluator.calculate_serendipity(user_id, forgetting_recs)
                    
                    # Record results
                    parameter_results.append({
                        'user_id': user_id,
                        'parameter_type': 'time_weight',
                        'parameter_value': time_weight,
                        'hit_rate': hit_rate,
                        'precision': precision,
                        'diversity': diversity,
                        'novelty': novelty,
                        'serendipity': serendipity
                    })
        
        # Test different decay rates
        self.logger.info("Testing different decay rates...")
        for decay_rate in tqdm(decay_rates, desc="Decay rates"):
            for user_id in sample_users:
                # Get original recommendations
                original_recs = self.kg.get_recommendations(user_id, method='hybrid')
                
                # Apply time-based decay with this rate
                self.fm.implement_time_based_decay(user_id, decay_parameter=decay_rate)
                
                # Get recommendations
                forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
                    'hybrid', {
                        'decay_parameter': decay_rate,
                        'time_weight': 0.4,  
                        'usage_weight': 0.3,
                        'novelty_weight': 0.3,
                        'forgetting_factor': 0.5
                    }
                )
                forgetting_recs = forgetting_rec_fn(user_id)
                
                # Evaluate
                if user_id in test_data:
                    hit_rate = self.evaluator.calculate_hit_rate_at_k(test_data[user_id], forgetting_recs, 10)
                    precision = self.evaluator.calculate_precision_at_k(test_data[user_id], forgetting_recs, 10)
                    diversity = self.evaluator.measure_recommendation_diversity_after_forgetting(
                        original_recs, forgetting_recs)['genre_diversity_after']
                    novelty = self.evaluator.calculate_novelty(forgetting_recs)
                    serendipity = self.evaluator.calculate_serendipity(user_id, forgetting_recs)
                    
                    # Record results
                    parameter_results.append({
                        'user_id': user_id,
                        'parameter_type': 'decay_rate',
                        'parameter_value': decay_rate,
                        'hit_rate': hit_rate,
                        'precision': precision,
                        'diversity': diversity,
                        'novelty': novelty,
                        'serendipity': serendipity
                    })
        
        # Test different forgetting factors
        self.logger.info("Testing different forgetting factors...")
        for factor in tqdm(forgetting_factors, desc="Forgetting factors"):
            params = {
                'time_weight': 0.4,
                'usage_weight': 0.3,
                'novelty_weight': 0.3,
                'forgetting_factor': factor
            }
            
            for user_id in sample_users:
                # Get original recommendations
                original_recs = self.kg.get_recommendations(user_id, method='hybrid')
                
                # Apply hybrid decay with default weights
                self.fm.create_hybrid_decay_function(
                    user_id,
                    time_weight=params['time_weight'],
                    usage_weight=params['usage_weight'],
                    novelty_weight=params['novelty_weight']
                )
                
                # Get forgetting-aware recommendations with this factor
                forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
                    'hybrid', params
                )
                forgetting_recs = forgetting_rec_fn(user_id)
                
                # Evaluate
                if user_id in test_data:
                    hit_rate = self.evaluator.calculate_hit_rate_at_k(test_data[user_id], forgetting_recs, 10)
                    precision = self.evaluator.calculate_precision_at_k(test_data[user_id], forgetting_recs, 10)
                    diversity = self.evaluator.measure_recommendation_diversity_after_forgetting(
                        original_recs, forgetting_recs)['genre_diversity_after']
                    novelty = self.evaluator.calculate_novelty(forgetting_recs)
                    serendipity = self.evaluator.calculate_serendipity(user_id, forgetting_recs)
                    
                    # Record results
                    parameter_results.append({
                        'user_id': user_id,
                        'parameter_type': 'forgetting_factor',
                        'parameter_value': factor,
                        'hit_rate': hit_rate,
                        'precision': precision,
                        'diversity': diversity,
                        'novelty': novelty,
                        'serendipity': serendipity
                    })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(parameter_results)
        
        # Save results
        results_path = os.path.join(self.config.output_dir, 'parameter_sensitivity_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # Visualize results
        self._visualize_parameter_sensitivity(results_df)
        
        self.logger.info(f"Parameter sensitivity experiment complete. Results saved to {results_path}")
        return results_df
    
    def _visualize_parameter_sensitivity(self, results_df):
        """Visualize parameter sensitivity results."""
        # Group by parameter type and value
        metric_columns = ['hit_rate', 'precision', 'diversity', 'novelty', 'serendipity']
        
        for parameter_type in results_df['parameter_type'].unique():
            param_data = results_df[results_df['parameter_type'] == parameter_type]
            
            plt.figure(figsize=(15, 10))
            
            for i, metric in enumerate(metric_columns):
                plt.subplot(2, 3, i+1)
                
                # Group by parameter value and calculate mean and std
                grouped = param_data.groupby('parameter_value')[metric].agg(['mean', 'std']).reset_index()
                
                # Plot
                plt.errorbar(grouped['parameter_value'], grouped['mean'], yerr=grouped['std'], 
                           marker='o', markersize=8, capsize=6, linewidth=2)
                
                plt.xlabel(parameter_type.replace('_', ' ').title())
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'Impact of {parameter_type.replace("_", " ").title()} on {metric.replace("_", " ").title()}')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_dir, f'parameter_sensitivity_{parameter_type}.png'))
    
    def run_privacy_impact(self):
        """
        Run an experiment to measure the impact of "right to be forgotten" on recommendations.
        
        Returns:
            Dictionary with results
        """
        self.logger.info("Running privacy impact experiment...")
        
        # Prepare test data
        test_data, test_users = self.prepare_train_test_data()
        
        # Sample a subset of users for efficiency
        if len(test_users) > 20:
            sample_users = np.random.choice(test_users, 20, replace=False)
        else:
            sample_users = test_users
        
        # Run experiment for complete forgetting
        complete_forget_results = []
        
        for user_id in tqdm(sample_users, desc="Complete forgetting"):
            forget_result = self.fm.simulate_right_to_be_forgotten(user_id)
            complete_forget_results.append(forget_result)
        
        # Run experiment for partial forgetting (25% of items)
        partial_forget_results = []
        
        for user_id in tqdm(sample_users, desc="Partial forgetting"):
            # Get user's rated movies
            rated_movies = list(self.kg.user_profiles[user_id]['rated_movies']) if user_id in self.kg.user_profiles else []
            
            if rated_movies:
                # Forget 25% of rated movies
                forget_count = max(1, len(rated_movies) // 4)
                forget_items = np.random.choice(rated_movies, forget_count, replace=False)
                
                forget_result = self.fm.simulate_right_to_be_forgotten(user_id, movie_ids=forget_items)
                partial_forget_results.append(forget_result)
        
        # Visualize results
        self.visualizer.visualize_privacy_impact(
            complete_forget_results,
            filename=os.path.join(self.config.output_dir, 'privacy_impact_complete.png')
        )
        
        if partial_forget_results:
            self.visualizer.visualize_privacy_impact(
                partial_forget_results,
                filename=os.path.join(self.config.output_dir, 'privacy_impact_partial.png')
            )
        
        # Save results
        privacy_results = {
            'complete_forget': complete_forget_results,
            'partial_forget': partial_forget_results
        }
        
        self.logger.info("Privacy impact experiment complete.")
        return privacy_results
    
    def run_scalability_test(self):
        """
        Run a scalability test to measure computational efficiency.
        
        Returns:
            Dictionary with benchmarking results
        """
        self.logger.info("Running scalability test...")
        
        # Test with different numbers of users and interactions
        user_counts = [10, 50, 100, 200]
        interaction_counts = [1000, 5000, 10000, None]  # None means use all
        
        all_results = {}
        
        # Test with different user counts
        user_results = []
        for num_users in tqdm(user_counts, desc="Testing user scalability"):
            benchmark = self.fm.benchmark_scalability(num_users=num_users, repetitions=3)
            benchmark['num_users'] = num_users
            user_results.append(benchmark)
        
        all_results['user_scalability'] = user_results
        
        # Test with different interaction counts
        interaction_results = []
        for num_interactions in tqdm(interaction_counts, desc="Testing interaction scalability"):
            if num_interactions is not None:
                benchmark = self.fm.benchmark_scalability(num_interactions=num_interactions, repetitions=3)
                benchmark['num_interactions'] = num_interactions
                interaction_results.append(benchmark)
        
        all_results['interaction_scalability'] = interaction_results
        
        # Visualize results
        self._visualize_scalability_results(all_results)
        
        self.logger.info("Scalability test complete.")
        return all_results
    
    def _visualize_scalability_results(self, scalability_results):
        """Visualize scalability test results with robust error handling."""
        self.logger.info("Visualizing scalability results...")
        
        # Visualize user scalability
        if 'user_scalability' not in scalability_results:
            self.logger.warning("No user scalability results found")
            return
            
        user_results = scalability_results['user_scalability']
        
        if not user_results or not isinstance(user_results, list):
            self.logger.warning("Invalid user scalability results format")
            return
        
        # Filter out any non-dictionary results
        user_results = [r for r in user_results if isinstance(r, dict)]
        
        if user_results:
            try:
                plt.figure(figsize=(12, 8))
                
                # Extract data for plotting, with error handling
                user_counts = []
                for r in user_results:
                    if 'num_users' in r:
                        user_counts.append(r['num_users'])
                    elif 'metadata' in r and isinstance(r['metadata'], dict) and 'num_users' in r['metadata']:
                        user_counts.append(r['metadata']['num_users'])
                    else:
                        self.logger.warning(f"Missing num_users in result: {r}")
                
                if not user_counts:
                    self.logger.warning("No valid user counts found")
                    return
                    
                # Find all valid strategy names
                strategies = set()
                for r in user_results:
                    for key, value in r.items():
                        if key != 'metadata' and key != 'num_users' and isinstance(value, dict) and 'avg_time' in value:
                            strategies.add(key)
                
                # Plot each strategy
                for strategy in strategies:
                    strategy_times = []
                    valid_indices = []
                    
                    for i, r in enumerate(user_results):
                        if strategy in r and isinstance(r[strategy], dict) and 'avg_time' in r[strategy]:
                            strategy_times.append(r[strategy]['avg_time'])
                            valid_indices.append(i)
                    
                    if strategy_times:
                        # Only use user counts that have valid times for this strategy
                        valid_user_counts = [user_counts[i] for i in valid_indices]
                        # Sort by user count for proper line display
                        sorted_data = sorted(zip(valid_user_counts, strategy_times))
                        if sorted_data:
                            valid_user_counts, strategy_times = zip(*sorted_data)
                            plt.plot(valid_user_counts, strategy_times, 'o-', linewidth=2, label=strategy)
                
                plt.xlabel('Number of Users')
                plt.ylabel('Average Time (seconds)')
                plt.title('Scalability with Increasing Number of Users')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                output_path = os.path.join(self.config.output_dir, 'scalability_users.png')
                plt.savefig(output_path)
                self.logger.info(f"Saved user scalability visualization to {output_path}")
                plt.close()
            except Exception as e:
                self.logger.error(f"Error visualizing user scalability: {e}")
        
        # Visualize interaction scalability
        if 'interaction_scalability' not in scalability_results:
            self.logger.warning("No interaction scalability results found")
            return
            
        interaction_results = scalability_results['interaction_scalability']
        
        # Filter out any non-dictionary results
        interaction_results = [r for r in interaction_results if isinstance(r, dict)]
        
        if interaction_results:
            try:
                plt.figure(figsize=(12, 8))
                
                # Extract data for plotting, with error handling
                interaction_counts = []
                for r in interaction_results:
                    if 'num_interactions' in r:
                        interaction_counts.append(r['num_interactions'])
                    elif 'metadata' in r and isinstance(r['metadata'], dict) and 'num_interactions' in r['metadata']:
                        interaction_counts.append(r['metadata']['num_interactions'])
                    else:
                        self.logger.warning(f"Missing num_interactions in result: {r}")
                
                if not interaction_counts:
                    self.logger.warning("No valid interaction counts found")
                    return
                    
                # Find all valid strategy names
                strategies = set()
                for r in interaction_results:
                    for key, value in r.items():
                        if key != 'metadata' and key != 'num_interactions' and isinstance(value, dict) and 'avg_time' in value:
                            strategies.add(key)
                
                # Plot each strategy
                for strategy in strategies:
                    strategy_times = []
                    valid_indices = []
                    
                    for i, r in enumerate(interaction_results):
                        if strategy in r and isinstance(r[strategy], dict) and 'avg_time' in r[strategy]:
                            strategy_times.append(r[strategy]['avg_time'])
                            valid_indices.append(i)
                    
                    if strategy_times:
                        # Only use interaction counts that have valid times for this strategy
                        valid_interaction_counts = [interaction_counts[i] for i in valid_indices]
                        # Sort by interaction count for proper line display
                        sorted_data = sorted(zip(valid_interaction_counts, strategy_times))
                        if sorted_data:
                            valid_interaction_counts, strategy_times = zip(*sorted_data)
                            plt.plot(valid_interaction_counts, strategy_times, 'o-', linewidth=2, label=strategy)
                
                plt.xlabel('Number of Interactions')
                plt.ylabel('Average Time (seconds)')
                plt.title('Scalability with Increasing Number of Interactions')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                output_path = os.path.join(self.config.output_dir, 'scalability_interactions.png')
                plt.savefig(output_path)
                self.logger.info(f"Saved interaction scalability visualization to {output_path}")
                plt.close()
            except Exception as e:
                self.logger.error(f"Error visualizing interaction scalability: {e}")
        
    def run_user_segmentation(self):
        """
        Analyze how forgetting mechanisms perform for different user segments.
        
        Returns:
            Dictionary with results for each segment
        """
        self.logger.info("Running user segmentation analysis...")
        
        # Prepare test data
        test_data, test_users = self.prepare_train_test_data()
        
        # Define user segments
        segments = {
            'high_activity': [],   # Users with many ratings
            'low_activity': [],    # Users with few ratings
            'diverse_taste': [],   # Users with diverse genre preferences
            'focused_taste': [],   # Users with focused genre preferences
            'new_users': [],       # Users with recent activity
            'established_users': [] # Users with longer history
        }
        
        # Segment threshold values
        rating_threshold = np.median([len(self.kg.ratings_df[self.kg.ratings_df['user_id'] == u]) 
                                    for u in test_users])
        
        # Calculate genre diversity for each user
        genre_diversity = {}
        for user_id in test_users:
            if user_id in self.kg.user_profiles:
                genre_prefs = self.kg.user_profiles[user_id]['genre_preferences']
                # Higher values indicate more diverse tastes
                genre_diversity[user_id] = np.sum(genre_prefs * (1 - genre_prefs))
            else:
                genre_diversity[user_id] = 0.5  # Default if not found
        
        diversity_threshold = np.median(list(genre_diversity.values()))
        
        # Calculate recency for each user
        recency = {}
        for user_id in test_users:
            if user_id in self.kg.user_profiles:
                last_time = self.kg.user_profiles[user_id].get('last_rating_time', 0)
                # Higher values indicate more recent activity
                recency[user_id] = last_time
            else:
                recency[user_id] = 0  # Default if not found
        
        recency_threshold = np.median(list(recency.values()))
        
        # Assign users to segments
        for user_id in test_users:
            # Activity segments
            rating_count = len(self.kg.ratings_df[self.kg.ratings_df['user_id'] == user_id])
            if rating_count >= rating_threshold:
                segments['high_activity'].append(user_id)
            else:
                segments['low_activity'].append(user_id)
            
            # Taste diversity segments
            if genre_diversity.get(user_id, 0) >= diversity_threshold:
                segments['diverse_taste'].append(user_id)
            else:
                segments['focused_taste'].append(user_id)
            
            # User recency segments
            if recency.get(user_id, 0) >= recency_threshold:
                segments['new_users'].append(user_id)
            else:
                segments['established_users'].append(user_id)
        
        # Define forgetting strategies to evaluate
        strategies = {
            'Time-Decay': lambda u: self._recommend_with_time_decay(u),
            'Usage-Based': lambda u: self._recommend_with_usage_based(u),
            'Hybrid-Decay': lambda u: self._recommend_with_hybrid_decay(u),
            'Personalized': lambda u: self._recommend_with_personalized_decay(u)
        }
        
        # Evaluate each segment
        segment_results = {}
        
        for segment_name, segment_users in tqdm(segments.items(), desc="Evaluating segments"):
            self.logger.info(f"Evaluating {segment_name} segment with {len(segment_users)} users...")
            
            if not segment_users:
                continue
                
            # Sample users if segment is large
            if len(segment_users) > 20:
                segment_sample = np.random.choice(segment_users, 20, replace=False)
            else:
                segment_sample = segment_users
            
            # Evaluate each strategy on this segment
            strategy_results = {}
            
            for strategy_name, strategy_fn in strategies.items():
                self.logger.info(f"Evaluating {strategy_name} on {segment_name} segment...")
                
                results = []
                
                for user_id in segment_sample:
                    if user_id not in test_data:
                        continue
                        
                    # Get recommendations
                    recommendations = strategy_fn(user_id)
                    
                    # Evaluate
                    hit_rate = self.evaluator.calculate_hit_rate_at_k(test_data[user_id], recommendations, 10)
                    ndcg = self.evaluator.calculate_ndcg_at_k(test_data[user_id], recommendations, 10)
                    serendipity = self.evaluator.calculate_serendipity(user_id, recommendations)
                    
                    results.append({
                        'user_id': user_id,
                        'hit_rate': hit_rate,
                        'ndcg': ndcg,
                        'serendipity': serendipity
                    })
                
                # Aggregate results
                if results:
                    strategy_results[strategy_name] = {
                        'hit_rate': np.mean([r['hit_rate'] for r in results]),
                        'ndcg': np.mean([r['ndcg'] for r in results]),
                        'serendipity': np.mean([r['serendipity'] for r in results]),
                        'sample_size': len(results)
                    }
            
            segment_results[segment_name] = strategy_results
        
        # Visualize segment results
        self._visualize_segment_results(segment_results)
        
        self.logger.info("User segmentation analysis complete.")
        return segment_results
    
    def _visualize_segment_results(self, segment_results):
        """Visualize user segmentation results."""
        metrics = ['hit_rate', 'ndcg', 'serendipity']
        
        for metric in metrics:
            plt.figure(figsize=(15, 8))
            
            # Prepare data for grouped bar chart
            segments = []
            strategies = []
            values = []
            
            for segment, strategies_data in segment_results.items():
                for strategy, metrics_data in strategies_data.items():
                    segments.append(segment.replace('_', ' ').title())
                    strategies.append(strategy)
                    values.append(metrics_data[metric])
            
            # Create DataFrame for easier plotting
            df = pd.DataFrame({
                'Segment': segments,
                'Strategy': strategies,
                'Value': values
            })
            
            # Create grouped bar chart
            ax = sns.barplot(x='Segment', y='Value', hue='Strategy', data=df)
            
            plt.title(f'{metric.replace("_", " ").title()} by User Segment and Strategy')
            plt.xlabel('User Segment')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Strategy')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.config.output_dir, f'segment_results_{metric}.png'))
    
    def run_experiment(self, experiment_name):
        """Run a predefined experiment."""
        if experiment_name == 'baseline_comparison':
            return self.run_baseline_comparison()
        elif experiment_name == 'improved_baseline_comparison':
            return self.run_improved_baseline_comparison()
        elif experiment_name == 'temporal_evaluation':
            return self.run_temporal_evaluation()
        elif experiment_name == 'parameter_sensitivity':
            return self.run_parameter_sensitivity()
        elif experiment_name == 'parameter_optimization':
            return self.run_parameter_optimization()
        elif experiment_name == 'privacy_impact':
            return self.run_privacy_impact()
        elif experiment_name == 'scalability_test':
            return self.run_scalability_test()
        elif experiment_name == 'user_segmentation':
            return self.run_user_segmentation()
        else:
            self.logger.error(f"Unknown experiment: {experiment_name}")
            return None
        

    # Helper methods for recommendation strategies
    
    def _recommend_popular(self, user_id, n=10):
        """Get popular movie recommendations."""
        # Get movie popularity counts
        movie_counts = self.kg.ratings_df['movie_id'].value_counts()
        
        # Get movies already rated by the user
        if user_id in self.kg.user_profiles:
            rated_movies = self.kg.user_profiles[user_id]['rated_movies']
        else:
            rated_movies = set()
        
        # Get popular movies not rated by the user
        popular_movies = []
        for movie_id, count in movie_counts.items():
            if movie_id not in rated_movies:
                popular_movies.append(movie_id)
                if len(popular_movies) >= n:
                    break
        
        return popular_movies
    
    def _recommend_with_time_decay(self, user_id, n=10):
        """Get recommendations with time-based decay."""
        # Apply time-based decay
        self.fm.implement_time_based_decay(user_id, decay_parameter=0.1)
        
        # Get recommendations
        forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
            'hybrid', {
                'decay_parameter': 0.1,
                'time_weight': 0.7,
                'usage_weight': 0.2,
                'novelty_weight': 0.1,
                'forgetting_factor': 0.5
            }
        )
        
        return forgetting_rec_fn(user_id, n=n)
    
    def _recommend_with_ebbinghaus(self, user_id, n=10):
        """Get recommendations with Ebbinghaus forgetting curve."""
        # Apply Ebbinghaus forgetting curve
        self.fm.implement_ebbinghaus_forgetting_curve(user_id, retention=0.9, strength=1.0)
        
        # Get recommendations
        forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
            'hybrid', {
                'retention': 0.9,
                'strength': 1.0,
                'time_weight': 0.5,
                'usage_weight': 0.3,
                'novelty_weight': 0.2,
                'forgetting_factor': 0.5
            }
        )
        
        return forgetting_rec_fn(user_id, n=n)
    
    def _recommend_with_power_law(self, user_id, n=10):
        """Get recommendations with power law decay."""
        # Apply power law decay
        self.fm.implement_power_law_decay(user_id, decay_factor=0.75)
        
        # Get recommendations
        forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
            'hybrid', {
                'decay_factor': 0.75,
                'time_weight': 0.5,
                'usage_weight': 0.3,
                'novelty_weight': 0.2,
                'forgetting_factor': 0.5
            }
        )
        
        return forgetting_rec_fn(user_id, n=n)
    
    def _recommend_with_usage_based(self, user_id, n=10):
        """Get recommendations with usage-based decay."""
        # Apply usage-based decay
        self.fm.implement_usage_based_decay(user_id, interaction_threshold=3)
        
        # Get recommendations
        forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
            'hybrid', {
                'interaction_threshold': 3,
                'time_weight': 0.2,
                'usage_weight': 0.6,
                'novelty_weight': 0.2,
                'forgetting_factor': 0.5
            }
        )
        
        return forgetting_rec_fn(user_id, n=n)
    
    def _recommend_with_hybrid_decay(self, user_id, n=10):
        """Get recommendations with hybrid decay."""
        # Apply hybrid decay
        self.fm.create_hybrid_decay_function(
            user_id,
            time_weight=0.4,
            usage_weight=0.3,
            novelty_weight=0.3
        )
        
        # Get recommendations
        forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
            'hybrid', {
                'time_weight': 0.4,
                'usage_weight': 0.3,
                'novelty_weight': 0.3,
                'forgetting_factor': 0.5
            }
        )
        
        return forgetting_rec_fn(user_id, n=n)
    
    def _recommend_with_personalized_decay(self, user_id, n=10):
        """Get recommendations with personalized decay parameters."""
        # Get personalized parameters
        params = self.fm.personalize_forgetting_parameters(user_id)
        
        # Apply hybrid decay with personalized parameters
        self.fm.create_hybrid_decay_function(
            user_id,
            time_weight=params['time_weight'],
            usage_weight=params['usage_weight'],
            novelty_weight=params['novelty_weight']
        )
        
        # Get recommendations
        forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
            'hybrid', params
        )
        
        return forgetting_rec_fn(user_id, n=n)
    
    def run_improved_baseline_comparison(self):
        """
        Run a comparison between original and improved algorithms.
        """
        self.logger.info("Running improved baseline comparison experiment...")

        # Prepare test data
        test_data, test_users = self.prepare_train_test_data()

        # Define traditional recommenders
        traditional_recommenders = {
            'Popular': lambda u: self.kg._recommend_popular(u),
            'Content-Based': lambda u: self.kg.get_personalized_recommendations(u),
            'Graph-Based': lambda u: self.kg.get_graph_based_recommendations(u),
            'Original-Hybrid': lambda u: self.kg.get_hybrid_recommendations(u)
        }

        # Define improved recommenders
        improved_recommenders = {
            'Improved-Hybrid': lambda u: self.kg.get_improved_hybrid_recommendations(u),
            'Knowledge-Enhanced': lambda u: self.kg.get_knowledge_enhanced_recommendations(u),
            'Segment-Adaptive': lambda u: self.kg.get_segment_adaptive_recommendations(u),
            'Improved-Decay': lambda u: self._recommend_with_improved_decay(u)
        }

        # Combine all recommenders
        all_recommenders = {**traditional_recommenders, **improved_recommenders}

        # Run comparison
        results = self.evaluator.compare_with_traditional_recommenders(
            test_users, test_data, traditional_recommenders, improved_recommenders)

        # Save results
        results_path = os.path.join(self.config.output_dir, 'improved_comparison_results.csv')
        results.to_csv(results_path, index=False)

        # Visualize results
        self.visualizer.visualize_comparative_evaluation(
            results, 
            metrics=['hit_rate@10', 'precision@10', 'ndcg@10', 'mrr', 'serendipity', 'novelty'],
            filename=os.path.join(self.config.output_dir, 'improved_baseline_comparison.png')
        )

        self.logger.info(f"Improved baseline comparison complete. Results saved to {results_path}")
        return results

    def run_parameter_optimization(self):
        """
        Run parameter optimization using Bayesian Optimization.
        """
        self.logger.info("Running parameter optimization experiment...")

        # Prepare test data
        test_data, test_users = self.prepare_train_test_data()

        # Sample a subset of users for efficiency
        if len(test_users) > 20:
            sample_users = np.random.choice(test_users, 20, replace=False)
        else:
            sample_users = test_users

        # Run optimization
        best_params, best_hit_rate = self.fm.optimize_forgetting_parameters(
            sample_users, test_data, n_calls=30)

        # Log and save results
        self.logger.info(f"Optimization complete. Best hit rate: {best_hit_rate}")
        self.logger.info(f"Best parameters: {best_params}")

        # Save best parameters
        params_path = os.path.join(self.config.output_dir, 'optimized_parameters.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)

        self.logger.info(f"Optimized parameters saved to {params_path}")
        return best_params
    
    def run_decay_rate_comparison(config):
        """Run a focused comparison of different memory decay approaches."""
        logger = logging.getLogger(__name__)
        logger.info("Running memory decay rate comparison...")

        # Initialize experiment runner
        runner = ForgettingExperimentRunner(config)

        # Run the specific comparison
        results = runner.compare_decay_rates()

        # Generate visualization
        comparison_path = os.path.join(config.output_dir, 'decay_rate_comparison.png')

        # Make sure results is a DataFrame
        if not isinstance(results, pd.DataFrame):
            logger.error("Expected a DataFrame from compare_decay_rates()")
            results = pd.DataFrame()  # Create empty DataFrame to avoid further errors

        runner.visualizer.visualize_decay_comparison(results, filename=comparison_path)

        logger.info(f"Memory decay rate comparison completed and saved to {comparison_path}")
        return results

    def _recommend_with_improved_decay(self, user_id, n=10):
        """
        Get recommendations with improved decay.
        """
        # Apply improved decay
        self.fm.implement_improved_decay(user_id, short_term_decay=0.05, long_term_factor=0.3)

        # Get recommendations
        forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
            'hybrid', {
                'short_term_decay': 0.05,
                'long_term_factor': 0.3,
                'time_weight': 0.4,
                'usage_weight': 0.3,
                'novelty_weight': 0.3,
                'forgetting_factor': 0.5
            }
        )

        return forgetting_rec_fn(user_id, n=n)
    

    def run_comprehensive_evaluation(self):
        """
        Run a comprehensive evaluation of all methods.
        """
        self.logger.info("Running comprehensive evaluation...")

        # Prepare test data
        test_data, test_users = self.prepare_train_test_data()

        # Define all recommendation strategies to test
        all_recommenders = {
            # Traditional methods
            'Popular': lambda u: self.kg._recommend_popular(u),
            'Content-Based': lambda u: self.kg.get_personalized_recommendations(u),
            'Graph-Based': lambda u: self.kg.get_graph_based_recommendations(u),
            'Original-Hybrid': lambda u: self.kg.get_hybrid_recommendations(u),

            # Forgetting-based methods
            'Time-Decay': lambda u: self._recommend_with_time_decay(u),
            'Ebbinghaus': lambda u: self._recommend_with_ebbinghaus(u),
            'Power-Law': lambda u: self._recommend_with_power_law(u),
            'Usage-Based': lambda u: self._recommend_with_usage_based(u),
            'Hybrid-Decay': lambda u: self._recommend_with_hybrid_decay(u),

            # Improved methods
            'Improved-Hybrid': lambda u: self.kg.get_improved_hybrid_recommendations(u),
            'Knowledge-Enhanced': lambda u: self.kg.get_knowledge_enhanced_recommendations(u),
            'Segment-Adaptive': lambda u: self.kg.get_segment_adaptive_recommendations(u),
            'Improved-Decay': lambda u: self._recommend_with_improved_decay(u)
        }

        # Run evaluation with all metrics
        metrics = ['hit_rate@5', 'hit_rate@10', 'precision@10', 'recall@10', 
                   'ndcg@10', 'mrr', 'serendipity', 'novelty', 'diversity']

        # Sample a subset of users for efficiency
        if len(test_users) > 50:
            evaluation_users = np.random.choice(test_users, 50, replace=False)
        else:
            evaluation_users = test_users

        # Run the evaluation
        results_df = self.evaluator.evaluate_all_recommenders(
            evaluation_users, test_data, all_recommenders, metrics)

        # Save detailed results
        results_path = os.path.join(self.config.output_dir, 'comprehensive_evaluation_results.csv')
        results_df.to_csv(results_path, index=False)

        # Create comprehensive summary
        summary = self.evaluator.create_comprehensive_summary(
            results_df, output_prefix="comprehensive_evaluation", k=10, save_summary=True)

        # Generate visualizations
        self.visualizer.visualize_comprehensive_evaluation(
            results_df, 
            metrics=['hit_rate@10', 'ndcg@10', 'serendipity', 'diversity'],
            filename=os.path.join(self.config.output_dir, 'comprehensive_evaluation')
        )

        self.logger.info(f"Comprehensive evaluation complete. Results saved to {results_path}")
        return results_df

    def compare_decay_rates(self):
        """
        Run a focused comparison of different memory decay approaches.
        """
        self.logger.info("Comparing different memory decay rates...")

        # Define different decay parameters to test
        decay_params = [
            {'name': 'Original-Fast-Decay', 'decay_parameter': 0.1, 'method': 'time_based'},
            {'name': 'Original-Slow-Decay', 'decay_parameter': 0.03, 'method': 'time_based'},
            {'name': 'Ebbinghaus', 'retention': 0.9, 'strength': 1.0, 'method': 'ebbinghaus'},
            {'name': 'Power-Law', 'decay_factor': 0.75, 'method': 'power_law'},
            {'name': 'Two-Phase-Default', 'short_term_decay': 0.05, 'long_term_factor': 0.3, 'method': 'improved'},
            {'name': 'Two-Phase-Gradual', 'short_term_decay': 0.03, 'long_term_factor': 0.5, 'method': 'improved'},
            {'name': 'Adaptive', 'method': 'adaptive'}
        ]

        # Select a small set of test users
        user_ids = list(self.kg.user_profiles.keys())
        if len(user_ids) > 10:
            test_users = np.random.choice(user_ids, 10, replace=False)
        else:
            test_users = user_ids

        # Test each decay method
        results = []

        for user_id in test_users:
            user_results = {'user_id': user_id}

            # Apply each decay method and record memory strengths
            for params in decay_params:
                method = params['method']

                # Save original memory strengths
                original_strengths = {}
                for key, value in self.fm.memory_strength.items():
                    if key[0] == user_id:
                        original_strengths[key] = value

                # Apply appropriate decay method
                if method == 'time_based':
                    self.fm.implement_time_based_decay(user_id, decay_parameter=params['decay_parameter'])
                elif method == 'ebbinghaus':
                    self.fm.implement_ebbinghaus_forgetting_curve(user_id, retention=params['retention'], 
                                                                strength=params['strength'])
                elif method == 'power_law':
                    self.fm.implement_power_law_decay(user_id, decay_factor=params['decay_factor'])
                elif method == 'improved':
                    self.fm.implement_improved_decay(user_id, short_term_decay=params['short_term_decay'], 
                                                   long_term_factor=params['long_term_factor'])
                elif method == 'adaptive':
                    self.fm.implement_adaptive_time_decay(user_id)

                # Calculate average memory strength
                memory_values = []
                for (u_id, m_id), strength in self.fm.memory_strength.items():
                    if u_id == user_id:
                        memory_values.append(strength)

                avg_memory = np.mean(memory_values) if memory_values else 0
                user_results[params['name']] = avg_memory

                # Restore original memory strengths
                for key, value in original_strengths.items():
                    self.fm.memory_strength[key] = value

            results.append(user_results)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Save results
        results_path = os.path.join(self.config.output_dir, 'decay_rate_comparison.csv')
        results_df.to_csv(results_path, index=False)

        self.logger.info(f"Decay rate comparison complete. Results saved to {results_path}")
        return results_df

    def test_recommendation_evolution(self):
        """
        Test how recommendations evolve over time with different decay methods.
        """
        self.logger.info("Testing recommendation evolution over time...")

        # Define decay methods to test
        decay_methods = [
            {'name': 'Original-Decay', 'fn': lambda u: self.fm.implement_time_based_decay(u, decay_parameter=0.1)},
            {'name': 'Improved-Decay', 'fn': lambda u: self.fm.implement_improved_decay(u)},
            {'name': 'Adaptive-Decay', 'fn': lambda u: self.fm.implement_adaptive_time_decay(u)}
        ]

        # Select a small set of test users
        user_ids = list(self.kg.user_profiles.keys())
        if len(user_ids) > 5:
            test_users = np.random.choice(user_ids, 5, replace=False)
        else:
            test_users = user_ids

        # Define time points to test
        time_points = [0, 7, 30, 90, 180]  # days

        all_results = []

        for user_id in test_users:
            # Save original memory state
            original_strengths = {}
            original_times = {}
            for key, value in self.fm.memory_strength.items():
                if key[0] == user_id:
                    original_strengths[key] = value
                    if key in self.fm.last_interaction_time:
                        original_times[key] = self.fm.last_interaction_time[key]

            # Get initial recommendations
            initial_recs = self.kg.get_hybrid_recommendations(user_id)

            # Test each decay method
            for method in decay_methods:
                method_name = method['name']
                decay_fn = method['fn']

                for days in time_points:
                    # Simulate time passage
                    current_time = datetime.datetime.now().timestamp()
                    for key in original_strengths:
                        if key in self.fm.last_interaction_time:
                            self.fm.last_interaction_time[key] = current_time - (days * 24 * 60 * 60)

                    # Apply decay
                    decay_fn(user_id)

                    # Get recommendations
                    forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
                        'hybrid', {'forgetting_factor': 0.5}
                    )
                    recs = forgetting_rec_fn(user_id)

                    # Calculate similarity with initial recommendations
                    jaccard = len(set(initial_recs).intersection(set(recs))) / len(set(initial_recs).union(set(recs)))

                    # Store result
                    all_results.append({
                        'user_id': user_id,
                        'method': method_name,
                        'days': days,
                        'jaccard_similarity': jaccard
                    })

                    # Restore original memory state for next iteration
                    for key, value in original_strengths.items():
                        self.fm.memory_strength[key] = value
                    for key, value in original_times.items():
                        self.fm.last_interaction_time[key] = value

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)

        # Save results
        results_path = os.path.join(self.config.output_dir, 'recommendation_evolution.csv')
        results_df.to_csv(results_path, index=False)

        # Create visualization
        self.visualizer.visualize_recommendation_evolution(
            results_df,
            filename=os.path.join(self.config.output_dir, 'recommendation_evolution.png')
        )

        self.logger.info(f"Recommendation evolution test complete. Results saved to {results_path}")
        return results_df