#src/forgetting_mechanism.py
import numpy as np
import datetime
import math
import logging
from collections import defaultdict
import pickle
import gzip
import time
from skopt import gp_minimize
from skopt.space import Real, Integer
        

class ForgettingMechanism:
    """
    Implements various forgetting mechanisms for recommendation systems.
    
    This class provides methods to simulate memory decay over time,
    allowing for more dynamic and temporally-aware recommendations.
    """
    def __init__(self, knowledge_graph):
        """
        Initialize the forgetting mechanism for a knowledge graph.
        
        Args:
            knowledge_graph: The MovieLensKnowledgeGraph instance
        """
        self.kg = knowledge_graph
        self.memory_strength = {}  # Maps (user_id, movie_id) to memory strength
        self.last_interaction_time = {}  # Maps (user_id, movie_id) to last interaction timestamp
        self.interaction_counts = defaultdict(int)  # Maps (user_id, movie_id) to interaction count
        self.user_activity_patterns = {}  # Maps user_id to activity pattern metrics
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ForgettingMechanism')
        
        # Initialize memory strengths from existing graph data
        self._initialize_memory_strengths()
    
    def _initialize_memory_strengths(self):
        """Initialize memory strengths from existing ratings data."""
        self.logger.info("Initializing memory strengths from ratings data...")
        
        if self.kg.ratings_df is None:
            self.logger.warning("No ratings data available for initialization")
            return
            
        for _, rating in self.kg.ratings_df.iterrows():
            user_id = rating['user_id']
            movie_id = rating['movie_id']
            rating_value = rating['rating']
            timestamp = rating['timestamp']
            
            # Initial memory strength is based on the rating value (normalized to [0,1])
            memory_strength = rating_value / 5.0
            
            self.memory_strength[(user_id, movie_id)] = memory_strength
            self.last_interaction_time[(user_id, movie_id)] = timestamp
            self.interaction_counts[(user_id, movie_id)] += 1
        
        self.logger.info(f"Initialized memory strengths for {len(self.memory_strength)} user-item pairs")
    
    def implement_time_based_decay(self, user_id, decay_parameter=0.1):
        """
        Implement time-based decay for a user's memories.
        
        Args:
            user_id: The user ID to apply decay to
            decay_parameter: Controls how quickly memories decay (smaller values = slower decay)
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        # Apply decay to all movies the user has interacted with
        for (u_id, movie_id), strength in self.memory_strength.items():
            if u_id == user_id:
                last_time = self.last_interaction_time.get((u_id, movie_id), 0)
                time_diff = current_time - last_time
                
                # Exponential decay formula: strength * e^(-decay_parameter * time_diff)
                # Time difference is in seconds, convert to days for more reasonable decay
                days_diff = time_diff / (24 * 60 * 60)
                decayed_strength = strength * math.exp(-decay_parameter * days_diff)
                
                # Update memory strength
                self.memory_strength[(u_id, movie_id)] = max(0.001, decayed_strength)  # Prevent complete forgetting
                user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        return user_memories
    
    def implement_ebbinghaus_forgetting_curve(self, user_id, retention=0.9, strength=1.0):
        """
        Implement the classic Ebbinghaus forgetting curve: R = e^(-t/S)
        where R is retention, t is time, and S is strength of memory.
        
        Args:
            user_id: The user ID to apply decay to
            retention: Base retention rate
            strength: Parameter controlling memory strength
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        for (u_id, movie_id), memory_strength in self.memory_strength.items():
            if u_id == user_id:
                last_time = self.last_interaction_time.get((u_id, movie_id), 0)
                time_diff = (current_time - last_time) / (24 * 60 * 60)  # days
                
                # Adjust strength based on rating value
                if (u_id, movie_id) in self.last_interaction_time:
                    rating_data = self.kg.ratings_df[
                        (self.kg.ratings_df['user_id'] == u_id) & 
                        (self.kg.ratings_df['movie_id'] == movie_id)
                    ]
                    if not rating_data.empty:
                        rating = rating_data.iloc[0]['rating']
                        individual_strength = strength * (rating / 5.0) # Adjust by rating
                    else:
                        individual_strength = strength
                else:
                    individual_strength = strength
                
                # Classic Ebbinghaus formula
                new_strength = retention * np.exp(-time_diff / individual_strength)
                self.memory_strength[(u_id, movie_id)] = max(0.001, new_strength)
                user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        return user_memories
    
    def implement_power_law_decay(self, user_id, decay_factor=0.75):
        """
        Implement power law decay, which better models long-term forgetting.
        Follows the form: S(t) = S(0) * (1 + t)^(-decay_factor)
        
        Args:
            user_id: The user ID to apply decay to
            decay_factor: Power law exponent controlling decay rate
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        for (u_id, movie_id), initial_strength in self.memory_strength.items():
            if u_id == user_id:
                last_time = self.last_interaction_time.get((u_id, movie_id), 0)
                time_diff = current_time - last_time
                
                # Convert to days and add 1 to avoid division by zero
                days_diff = (time_diff / (24 * 60 * 60)) + 1
                
                # Power law decay
                decayed_strength = initial_strength * (days_diff ** (-decay_factor))
                
                # Update memory strength
                self.memory_strength[(u_id, movie_id)] = max(0.001, decayed_strength)
                user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        return user_memories
    
    def implement_usage_based_decay(self, user_id, interaction_threshold=3):
        """
        Implement usage-based decay where less frequently accessed items decay faster.
        
        Args:
            user_id: The user ID to apply decay to
            interaction_threshold: Number of interactions below which memory decays faster
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        user_memories = {}
        
        for (u_id, movie_id), strength in self.memory_strength.items():
            if u_id == user_id:
                interaction_count = self.interaction_counts.get((u_id, movie_id), 0)
                
                # Apply stronger decay to less frequently accessed items
                if interaction_count < interaction_threshold:
                    usage_decay_factor = 0.8  # Stronger decay for less used items
                else:
                    usage_decay_factor = 0.95  # Weaker decay for frequently used items
                
                # Apply usage-based decay
                decayed_strength = strength * usage_decay_factor
                self.memory_strength[(u_id, movie_id)] = max(0.001, decayed_strength)
                user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        return user_memories
    
    def implement_step_function_decay(self, user_id, time_thresholds=[7, 30, 90], decay_factors=[0.9, 0.7, 0.5, 0.2]):
        """
        Implement step-function decay where memory strength decreases in discrete steps
        based on time elapsed since last interaction.
        
        Args:
            user_id: The user ID to apply decay to
            time_thresholds: List of time thresholds in days
            decay_factors: List of decay factors for each time bin (should be len(time_thresholds) + 1)
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        if len(decay_factors) != len(time_thresholds) + 1:
            self.logger.warning("Number of decay factors should be one more than time thresholds")
            return user_memories
            
        for (u_id, movie_id), initial_strength in self.memory_strength.items():
            if u_id == user_id:
                last_time = self.last_interaction_time.get((u_id, movie_id), 0)
                time_diff = current_time - last_time
                days_diff = time_diff / (24 * 60 * 60)
                
                # Determine which time bin this item falls into
                bin_index = len(time_thresholds)  # Default to last bin
                for i, threshold in enumerate(time_thresholds):
                    if days_diff <= threshold:
                        bin_index = i
                        break
                
                # Apply corresponding decay factor
                decay_factor = decay_factors[bin_index]
                decayed_strength = initial_strength * decay_factor
                
                # Update memory strength
                self.memory_strength[(u_id, movie_id)] = max(0.001, decayed_strength)
                user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        return user_memories
    
    def create_hybrid_decay_function(self, user_id, time_weight=0.4, usage_weight=0.3, novelty_weight=0.3):
        """
        Create a hybrid decay function that combines time-based, usage-based, and novelty-based decay.
        
        Args:
            user_id: The user ID to apply decay to
            time_weight: Weight for time-based decay
            usage_weight: Weight for usage-based decay
            novelty_weight: Weight for novelty-based decay
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        # Get all movies the user has interacted with
        user_movies = [(m_id, strength) for (u_id, m_id), strength in self.memory_strength.items() if u_id == user_id]
        
        # Calculate average rating time to determine movie novelty
        avg_timestamp = sum(self.last_interaction_time.get((user_id, m_id), 0) for m_id, _ in user_movies) / max(1, len(user_movies))
        
        for movie_id, strength in user_movies:
            # Time-based component
            last_time = self.last_interaction_time.get((user_id, movie_id), 0)
            time_diff = current_time - last_time
            days_diff = time_diff / (24 * 60 * 60)
            time_decay = math.exp(-0.05 * days_diff)  # Slower decay rate
            
            # Usage-based component
            interaction_count = self.interaction_counts.get((user_id, movie_id), 0)
            usage_factor = min(1.0, interaction_count / 5.0)  # Normalize to [0,1]
            
            # Novelty-based component
            movie_timestamp = self.last_interaction_time.get((user_id, movie_id), 0)
            novelty_factor = 1.0 if movie_timestamp > avg_timestamp else 0.8
            
            # Combine all factors
            hybrid_factor = (time_weight * time_decay + 
                             usage_weight * usage_factor + 
                             novelty_weight * novelty_factor)
            
            # Apply decay
            new_strength = strength * hybrid_factor
            self.memory_strength[(user_id, movie_id)] = max(0.001, min(1.0, new_strength))
            user_memories[movie_id] = self.memory_strength[(user_id, movie_id)]
        
        return user_memories
    
    def implement_improved_decay(self, user_id, short_term_decay=0.05, long_term_factor=0.3):
        """
        Implement two-phase memory decay with both short and long-term components.
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}

        for (u_id, movie_id), strength in self.memory_strength.items():
            if u_id == user_id:
                last_time = self.last_interaction_time.get((u_id, movie_id), 0)
                time_diff = current_time - last_time
                days_diff = time_diff / (24 * 60 * 60)

                # Two-phase decay: rapid initial decline followed by slower long-term decay
                short_term = (1 - long_term_factor) * math.exp(-short_term_decay * days_diff)
                long_term = long_term_factor * math.exp(-short_term_decay * days_diff * 0.1)

                decayed_strength = strength * (short_term + long_term)
                self.memory_strength[(u_id, movie_id)] = max(0.05, decayed_strength)  # Higher floor
                user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]

        return user_memories

    def _calculate_genre_diversity(self, movie_ids):
        """
        Helper method to calculate genre diversity for a list of movie IDs.
        
        Args:
            movie_ids: List of movie IDs to calculate diversity for
            
        Returns:
            Diversity score between 0 and 1 (higher values mean more diverse)
        """
        if not movie_ids:
            return 0
                
        genre_vectors = []
        for movie_id in movie_ids:
            if movie_id in self.kg.movie_features:
                genre_vectors.append(self.kg.movie_features[movie_id])
        
        if not genre_vectors:
            return 0
                
        # Need to import here for the cosine_similarity function
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(genre_vectors)
        np.fill_diagonal(sim_matrix, 0)
        
        return 1 - np.mean(sim_matrix)
    
    def personalize_forgetting_parameters(self, user_id):
        """
        Personalize forgetting mechanism parameters based on user characteristics.

        Args:
            user_id: The user ID

        Returns:
            Dictionary of personalized parameters for the hybrid decay function
        """
        # Get user activity level
        user_ratings = [strength for (u_id, m_id), strength in self.memory_strength.items() if u_id == user_id]
        activity_level = len(user_ratings)

        # Get user movie preferences diversity
        user_movies = [m_id for (u_id, m_id), _ in self.memory_strength.items() if u_id == user_id]
        diversity = self._calculate_genre_diversity(user_movies)

        # Adjust weights based on user characteristics
        if activity_level > 50:  # High activity user
            if diversity > 0.6:  # Diverse taste
                # For active users with diverse taste, emphasize novelty
                time_weight = 0.3
                usage_weight = 0.3
                novelty_weight = 0.4
            else:  # Focused taste
                # For active users with focused taste, emphasize usage
                time_weight = 0.3
                usage_weight = 0.5
                novelty_weight = 0.2
        else:  # Low activity user
            if diversity > 0.6:  # Diverse taste
                # For casual users with diverse taste, balanced approach
                time_weight = 0.4
                usage_weight = 0.3
                novelty_weight = 0.3
            else:  # Focused taste
                # For casual users with focused taste, emphasize time decay
                time_weight = 0.5
                usage_weight = 0.3
                novelty_weight = 0.2

        # Return parameters that match the signature of create_hybrid_decay_function
        return {
            'time_weight': time_weight,
            'usage_weight': usage_weight,
            'novelty_weight': novelty_weight
        }
    
    def dynamic_half_life_adjustment(self, user_profile):
        """
        Dynamically adjust the half-life of the forgetting curve based on user profile.
        
        Args:
            user_profile: The user profile dictionary from KG or a user_id
            
        Returns:
            Dictionary mapping movie_id to adjusted half-life values
        """
        if isinstance(user_profile, int):
            user_id = user_profile
            if user_id in self.kg.user_profiles:
                user_profile = self.kg.user_profiles[user_id]
            else:
                return {}
        
        # Get genre preferences from user profile
        genre_preferences = user_profile.get('genre_preferences', np.zeros(19))
        
        # Scale the genre preferences to get positive values
        min_pref = np.min(genre_preferences)
        max_pref = np.max(genre_preferences)
        if max_pref == min_pref:
            scaled_preferences = np.ones_like(genre_preferences) * 0.5
        else:
            scaled_preferences = (genre_preferences - min_pref) / (max_pref - min_pref + 1e-10)
        
        half_lives = {}
        
        # For each movie the user has rated
        for movie_id in user_profile.get('rated_movies', set()):
            if movie_id in self.kg.movie_features:
                # Get movie genre features
                movie_genres = self.kg.movie_features[movie_id]
                
                # Calculate relevance to user preferences
                genre_match = np.sum(scaled_preferences * movie_genres) / (np.sum(movie_genres) + 1e-10)
                
                # Adjust half-life based on genre match
                # Movies that match user preferences have longer half-lives
                base_half_life = 30  # Base half-life in days
                adjusted_half_life = base_half_life * (1 + genre_match)
                
                half_lives[movie_id] = adjusted_half_life
        
        return half_lives
    
    def apply_forgetting_to_recommendations(self, user_id, recommendation_scores, forgetting_factor=0.5):
        """
        Improved implementation to apply forgetting to recommendations with better balancing.
        """
        adjusted_scores = {}

        for movie_id, score in recommendation_scores.items():
            memory_strength = self.memory_strength.get((user_id, movie_id), 1.0)

            # Balance between familiar items (high memory) and novel items (low memory)
            # Instead of just boosting novelty, create a proper balance
            if memory_strength > 0.7:
                # Familiar items get slight penalty
                adjustment = -0.1 * forgetting_factor
            elif memory_strength < 0.3:
                # Novel items get boost
                adjustment = 0.3 * forgetting_factor
            else:
                # Items in the middle get smaller adjustments
                adjustment = (0.5 - memory_strength) * forgetting_factor

            adjusted_scores[movie_id] = score * (1.0 + adjustment)

        return adjusted_scores
    
    def integrate_forgetting_mechanism_into_recommendation_pipeline(self, recommendation_algorithm, forgetting_parameters):
        """
        Integrate forgetting mechanism into the recommendation pipeline.
        
        Args:
            recommendation_algorithm: Function that returns recommendation scores for a user
            forgetting_parameters: Dictionary of forgetting parameters
            
        Returns:
            Function that generates recommendations with forgetting mechanism applied
        """
        def forgetting_aware_recommendations(user_id, n=10):
            # Get personalized forgetting parameters if not provided
            if user_id not in self.user_activity_patterns:
                user_params = self.personalize_forgetting_parameters(user_id)
            else:
                user_params = forgetting_parameters

            # Ensure all required parameters are present with defaults
            params = {
                'time_weight': user_params.get('time_weight', 0.4),
                'usage_weight': user_params.get('usage_weight', 0.3),
                'novelty_weight': user_params.get('novelty_weight', 0.3),
                'forgetting_factor': user_params.get('forgetting_factor', 0.5)
            }

            # Apply hybrid decay to update memory strengths
            self.create_hybrid_decay_function(
                user_id, 
                time_weight=params['time_weight'],
                usage_weight=params['usage_weight'],
                novelty_weight=params['novelty_weight']
            )
            
            # Get base recommendations
            if recommendation_algorithm == 'personalized':
                movie_ids = self.kg.get_personalized_recommendations(user_id, n=n*2)
                
                # Create scores dictionary (normalized to 0-1)
                scores = {mid: (n*2 - i) / (n*2) for i, mid in enumerate(movie_ids)}
                
            elif recommendation_algorithm == 'graph_based':
                movie_ids = self.kg.get_graph_based_recommendations(user_id, n=n*2)
                scores = {mid: (n*2 - i) / (n*2) for i, mid in enumerate(movie_ids)}
                
            elif recommendation_algorithm == 'hybrid':
                movie_ids = self.kg.get_hybrid_recommendations(user_id, n=n*2)
                scores = {mid: (n*2 - i) / (n*2) for i, mid in enumerate(movie_ids)}
                
            else:
                # Custom recommendation algorithm that returns scores
                scores = recommendation_algorithm(user_id)
            
            # Apply forgetting mechanism to adjust scores
            adjusted_scores = self.apply_forgetting_to_recommendations(
                user_id, 
                scores, 
                forgetting_factor=params.get('forgetting_factor', 0.5)
            )
            
            # Sort by adjusted scores and return top n
            sorted_recommendations = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)[:n]
            return [movie_id for movie_id, _ in sorted_recommendations]
        
        return forgetting_aware_recommendations

    def simulate_right_to_be_forgotten(self, user_id, movie_ids=None):
        """
        Simulate a GDPR right to be forgotten request by completely removing
        user-movie interactions from the knowledge graph.

        Args:
            user_id: The user ID requesting to be forgotten
            movie_ids: Optional list of specific movie IDs to forget (if None, forget all)

        Returns:
            Impact metrics on recommendation quality
        """
        # Store original recommendations for comparison
        original_recs = self.kg.get_personalized_recommendations(user_id)

        # Store original data
        original_ratings = self.kg.ratings_df.copy()

        # Remove interactions
        if movie_ids is None:
            # Remove all user's ratings
            self.kg.ratings_df = self.kg.ratings_df[self.kg.ratings_df['user_id'] != user_id]

            # Also remove from memory strength
            keys_to_remove = []
            for key in self.memory_strength:
                if key[0] == user_id:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.memory_strength[key]
                if key in self.last_interaction_time:
                    del self.last_interaction_time[key]
                if key in self.interaction_counts:
                    del self.interaction_counts[key]

            # Remove from user profiles
            if user_id in self.kg.user_profiles:
                del self.kg.user_profiles[user_id]
        else:
            # Remove specific ratings
            self.kg.ratings_df = self.kg.ratings_df[
                ~((self.kg.ratings_df['user_id'] == user_id) & 
                  (self.kg.ratings_df['movie_id'].isin(movie_ids)))
            ]

            # Remove from memory strength
            for movie_id in movie_ids:
                key = (user_id, movie_id)
                if key in self.memory_strength:
                    del self.memory_strength[key]
                if key in self.last_interaction_time:
                    del self.last_interaction_time[key]
                if key in self.interaction_counts:
                    del self.interaction_counts[key]

            # Update user profile
            if user_id in self.kg.user_profiles:
                user_ratings = self.kg.ratings_df[self.kg.ratings_df['user_id'] == user_id]
                rated_movies = set(user_ratings['movie_id'].values)

                if rated_movies:
                    genre_vectors = [self.kg.movie_features.get(mid, np.zeros(19)) for mid in rated_movies]
                    genre_preferences = np.mean(genre_vectors, axis=0)
                else:
                    genre_preferences = np.zeros(19)

                self.kg.user_profiles[user_id]['rated_movies'] = rated_movies
                self.kg.user_profiles[user_id]['genre_preferences'] = genre_preferences

        # Get new recommendations
        if user_id in self.kg.user_profiles:
            new_recs = self.kg.get_personalized_recommendations(user_id)
        else:
            new_recs = []

        # Calculate impact if possible
        if new_recs:
            from collections import defaultdict
            # Create a simple diversity metric
            genre_diversity_before = self._calculate_genre_diversity(original_recs)
            genre_diversity_after = self._calculate_genre_diversity(new_recs)

            jaccard_similarity = len(set(original_recs).intersection(set(new_recs))) / \
                                len(set(original_recs).union(set(new_recs))) if original_recs and new_recs else 0

            new_items = [item for item in new_recs if item not in original_recs]
            new_item_percentage = len(new_items) / len(new_recs) if new_recs else 0

            impact = {
                'genre_diversity_before': genre_diversity_before,
                'genre_diversity_after': genre_diversity_after,
                'jaccard_similarity': jaccard_similarity,
                'new_item_percentage': new_item_percentage
            }
        else:
            impact = {
                'genre_diversity_before': 0,
                'genre_diversity_after': 0,
                'jaccard_similarity': 0,
                'new_item_percentage': 0,
                'complete_forget': True
            }

        # Restore original data
        self.kg.ratings_df = original_ratings

        # Rebuild user profiles
        self.kg._build_user_profiles()

        # Reinitialize memory strengths
        self._initialize_memory_strengths()

        # FIX: Use shape[0] to get the count of filtered items
        forgotten_count = 0
        if movie_ids is not None:
            forgotten_count = len(movie_ids)
        else:
            user_ratings = original_ratings[original_ratings['user_id'] == user_id]
            forgotten_count = user_ratings.shape[0]

        return {
            'user_id': user_id,
            'forgotten_items': forgotten_count,
            'impact_metrics': impact
        }
    
    def serialize_and_store_memory_state(self, file_path, compression_level=0):
        """
        Serialize and store the current memory state.
        
        Args:
            file_path: Path to store the memory state
            compression_level: 0-9 compression level (0=none, 9=max)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                'memory_strength': self.memory_strength,
                'last_interaction_time': self.last_interaction_time,
                'interaction_counts': self.interaction_counts,
                'user_activity_patterns': self.user_activity_patterns
            }
            
            if compression_level > 0:
                with gzip.open(file_path, 'wb', compresslevel=compression_level) as f:
                    pickle.dump(data, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            return True
        except Exception as e:
            self.logger.error(f"Error storing memory state: {e}")
            return False
    
    def load_and_restore_memory_state(self, file_path, validation_check=True):
        """
        Load and restore a previously saved memory state.
        
        Args:
            file_path: Path to the stored memory state
            validation_check: Whether to validate the loaded data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to load as gzipped first
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            except:
                # If not gzipped, try normal pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            # Validation check
            if validation_check:
                required_keys = ['memory_strength', 'last_interaction_time', 
                                'interaction_counts', 'user_activity_patterns']
                
                if not all(key in data for key in required_keys):
                    self.logger.error("Invalid memory state file: missing required data")
                    return False
            
            # Restore state
            self.memory_strength = data['memory_strength']
            self.last_interaction_time = data['last_interaction_time']
            self.interaction_counts = data['interaction_counts']
            self.user_activity_patterns = data['user_activity_patterns']
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading memory state: {e}")
            return False
    
    def benchmark_scalability(self, num_users=None, num_interactions=None, repetitions=3):
        """
        Benchmark the scalability of forgetting mechanisms.
        
        Args:
            num_users: Number of users to test with (if None, use all)
            num_interactions: Number of interactions to test with (if None, use all)
            repetitions: Number of times to repeat each test
            
        Returns:
            Dictionary with benchmarking results
        """
        # Prepare subset of data if needed
        if num_users is not None or num_interactions is not None:
            original_ratings = self.kg.ratings_df.copy()
            
            if num_users is not None:
                user_ids = list(self.kg.ratings_df['user_id'].unique())
                if num_users < len(user_ids):
                    selected_users = user_ids[:num_users]
                    self.kg.ratings_df = self.kg.ratings_df[self.kg.ratings_df['user_id'].isin(selected_users)]
            
            if num_interactions is not None and len(self.kg.ratings_df) > num_interactions:
                self.kg.ratings_df = self.kg.ratings_df.sample(num_interactions)
            
            # Reinitialize with subset
            self._initialize_memory_strengths()
        
        # Define strategies to benchmark
        strategies = {
            'time_based': lambda u: self.implement_time_based_decay(u),
            'ebbinghaus': lambda u: self.implement_ebbinghaus_forgetting_curve(u),
            'power_law': lambda u: self.implement_power_law_decay(u),
            'usage_based': lambda u: self.implement_usage_based_decay(u),
            'step_function': lambda u: self.implement_step_function_decay(u),
            'hybrid': lambda u: self.create_hybrid_decay_function(u)
        }
        
        # Run benchmarks
        results = defaultdict(list)
        user_ids = list(self.kg.ratings_df['user_id'].unique())
        
        self.logger.info(f"Benchmarking with {len(user_ids)} users and {len(self.kg.ratings_df)} interactions")
        
        for strategy_name, strategy_fn in strategies.items():
            self.logger.info(f"Benchmarking {strategy_name} strategy...")
            
            for _ in range(repetitions):
                start_time = time.time()
                
                # Apply to all users
                for user_id in user_ids:
                    strategy_fn(user_id)
                
                end_time = time.time()
                results[strategy_name].append(end_time - start_time)
        
        # Restore original data if needed
        if num_users is not None or num_interactions is not None:
            self.kg.ratings_df = original_ratings
            self._initialize_memory_strengths()
        
        # Process results
        benchmark_results = {}
        for strategy_name, times in results.items():
            benchmark_results[strategy_name] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times)
            }
        
        benchmark_results['metadata'] = {
            'num_users': len(user_ids),
            'num_interactions': len(self.kg.ratings_df),
            'repetitions': repetitions
        }
        
        return benchmark_results
    
    def implement_adaptive_time_decay(self, user_id):
        """
        Implement decay with time windows that adapt to user behavior patterns.
        """
        # Analyze user rating pattern to determine appropriate time windows
        user_ratings = self.kg.ratings_df[self.kg.ratings_df['user_id'] == user_id]

        if user_ratings.empty:
            return {}

        # Sort by timestamp
        user_ratings = user_ratings.sort_values('timestamp')

        # Calculate time gaps between consecutive ratings
        timestamps = user_ratings['timestamp'].values
        gaps = np.diff(timestamps) / (24 * 60 * 60)  # Convert to days

        if len(gaps) > 0:
            median_gap = np.median(gaps)
        else:
            median_gap = 30  # Default

        # Set decay rate based on user's typical rating frequency
        if median_gap < 7:  # Active user, rates weekly
            decay_rate = 0.03  # Slower decay
        elif median_gap < 30:  # Regular user, rates monthly
            decay_rate = 0.05
        else:  # Infrequent user
            decay_rate = 0.08  # Faster decay

        # Apply decay with personalized rate
        return self.implement_time_based_decay(user_id, decay_parameter=decay_rate)
    
    def optimize_forgetting_parameters(self, user_ids, test_data, n_calls=20):
        """
        Use Bayesian optimization to find best forgetting parameters.
        Requires skopt package.
        """

        # Define the search space
        space = [
            Real(0.01, 0.2, name='decay_parameter'),
            Real(0.1, 0.9, name='time_weight'),
            Real(0.1, 0.9, name='usage_weight'),
            Real(0.1, 0.9, name='novelty_weight'),
            Real(0.1, 0.9, name='forgetting_factor')
        ]

        # Define the objective function
        def objective(params):
            decay_param, time_w, usage_w, novelty_w, forget_f = params

            # Normalize weights to sum to 1
            total = time_w + usage_w + novelty_w
            time_w /= total
            usage_w /= total
            novelty_w /= total

            # Test performance with these parameters
            hit_rates = []

            for user_id in user_ids:
                if user_id not in test_data:
                    continue

                # Apply forgetting with these parameters
                self.fm.create_hybrid_decay_function(
                    user_id, 
                    time_weight=time_w,
                    usage_weight=usage_w,
                    novelty_weight=novelty_w
                )

                # Get recommendations
                forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
                    'hybrid', {
                        'decay_parameter': decay_param,
                        'time_weight': time_w,
                        'usage_weight': usage_w,
                        'novelty_weight': novelty_w,
                        'forgetting_factor': forget_f
                    }
                )
                recommendations = forgetting_rec_fn(user_id)

                # Calculate hit rate
                hit_rate = self.evaluator.calculate_hit_rate_at_k(test_data[user_id], recommendations, 10)
                hit_rates.append(hit_rate)

            # Return negative mean hit rate (for minimization)
            return -np.mean(hit_rates)

        # Run Bayesian optimization
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

        # Return best parameters
        best_params = {
            'decay_parameter': result.x[0],
            'time_weight': result.x[1] / (result.x[1] + result.x[2] + result.x[3]),
            'usage_weight': result.x[2] / (result.x[1] + result.x[2] + result.x[3]),
            'novelty_weight': result.x[3] / (result.x[1] + result.x[2] + result.x[3]),
            'forgetting_factor': result.x[4]
        }

        return best_params, -result.fun  # Return best params and best hit rate