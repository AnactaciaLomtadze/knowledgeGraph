#src/context_forgetting.py
import numpy as np
from collections import defaultdict
import datetime
import logging
import math

class ContextAwareForgettingMechanism:
    """
    Enhanced forgetting mechanism that considers the contextual importance of information.
    
    This class extends the basic ForgettingMechanism with context-aware capabilities,
    such as preserving memories of significant items (award-winning movies, etc.) longer
    and adapting forgetting rates based on item properties.
    """
    def __init__(self, knowledge_graph):
        """
        Initialize the context-aware forgetting mechanism.
        
        Args:
            knowledge_graph: The MovieLensKnowledgeGraph instance
        """
        self.kg = knowledge_graph
        self.memory_strength = {}  # Maps (user_id, movie_id) to memory strength
        self.last_interaction_time = {}  # Maps (user_id, movie_id) to last interaction timestamp
        self.interaction_counts = defaultdict(int)  # Maps (user_id, movie_id) to interaction count
        self.user_activity_patterns = {}  # Maps user_id to activity pattern metrics
        self.item_significance = {}  # Maps movie_id to significance score
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ContextAwareForgettingMechanism')
        
        # Initialize memory strengths from existing graph data
        self._initialize_memory_strengths()
        
        # Calculate item significance scores
        self._calculate_item_significance()
    
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
    
    def _calculate_item_significance(self):
        """
        Calculate significance scores for items based on their properties.
        
        Significance factors include:
        - Awards received
        - Box office performance
        - Critical acclaim
        - Cultural importance
        """
        self.logger.info("Calculating item significance scores...")
        
        # Initialize with default significance (0.5)
        for movie_id in self.kg.movie_features:
            self.item_significance[movie_id] = 0.5
        
        # Enhance significance based on graph properties
        for movie_id in self.item_significance:
            movie_node = f"movie_{movie_id}"
            
            if movie_node not in self.kg.G:
                continue
            
            significance = 0.5  # Base significance
            
            # Check for awards
            award_count = 0
            for neighbor in self.kg.G.neighbors(movie_node):
                if neighbor.startswith("award_"):
                    award_count += 1
            
            # Award bonus (up to 0.4 additional significance)
            award_bonus = min(0.4, award_count * 0.1)
            significance += award_bonus
            
            # Check for box office success
            box_office = self.kg.G.nodes[movie_node].get('box_office')
            if box_office:
                try:
                    # Normalize to a value between 0 and 0.2
                    box_office_value = float(box_office)
                    box_office_bonus = min(0.2, box_office_value / 1000000000 * 0.2)  # Assume 1B+ is max
                    significance += box_office_bonus
                except (ValueError, TypeError):
                    pass
            
            # Check for director significance
            for neighbor in self.kg.G.neighbors(movie_node):
                if neighbor.startswith("director_"):
                    # Check if this director has other highly-rated movies
                    director_rating_sum = 0
                    director_movie_count = 0
                    
                    for connected_movie in self.kg.G.neighbors(neighbor):
                        if connected_movie.startswith("movie_") and connected_movie != movie_node:
                            # Get average rating of this movie
                            connected_movie_id = int(connected_movie.split("_")[1])
                            movie_ratings = self.kg.ratings_df[self.kg.ratings_df['movie_id'] == connected_movie_id]
                            if not movie_ratings.empty:
                                avg_rating = movie_ratings['rating'].mean()
                                director_rating_sum += avg_rating
                                director_movie_count += 1
                    
                    if director_movie_count > 0:
                        director_avg_rating = director_rating_sum / director_movie_count
                        # Director bonus (up to 0.2 additional significance)
                        director_bonus = min(0.2, (director_avg_rating - 2.5) / 2.5 * 0.2)
                        significance += max(0, director_bonus)
            
            # Cap significance at 1.0
            self.item_significance[movie_id] = min(1.0, significance)
        
        self.logger.info(f"Calculated significance scores for {len(self.item_significance)} movies")
    
    def implement_context_aware_decay(self, user_id, decay_parameter=0.1):
        """
        Implement context-aware decay that considers item significance.
        
        Args:
            user_id: The user ID to apply decay to
            decay_parameter: Base decay parameter (adjusted by item significance)
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        # Apply decay to all movies the user has interacted with
        for (u_id, movie_id), strength in self.memory_strength.items():
            if u_id == user_id:
                # Get significance factor
                significance = self.item_significance.get(movie_id, 0.5)
                
                # Adjust decay rate based on significance
                # Higher significance = slower decay
                adjusted_decay = decay_parameter * (1.0 - significance * 0.8)
                
                # Apply time-based decay
                last_time = self.last_interaction_time.get((u_id, movie_id), 0)
                time_diff = current_time - last_time
                days_diff = time_diff / (24 * 60 * 60)
                
                # Exponential decay with adjusted parameter
                decayed_strength = strength * math.exp(-adjusted_decay * days_diff)
                
                # Update memory strength with a minimum floor
                # Significant items never completely fade from memory
                min_strength = 0.001 + significance * 0.1
                self.memory_strength[(u_id, movie_id)] = max(min_strength, decayed_strength)
                
                user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        return user_memories
    
    def implement_context_aware_ebbinghaus(self, user_id, base_retention=0.9):
        """
        Implement context-aware Ebbinghaus forgetting curve.
        
        Args:
            user_id: The user ID to apply decay to
            base_retention: Base retention rate (adjusted by item significance)
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        # Apply to all movies the user has interacted with
        for (u_id, movie_id), memory_strength in self.memory_strength.items():
            if u_id == user_id:
                # Get significance factor
                significance = self.item_significance.get(movie_id, 0.5)
                
                # Adjust retention based on significance
                # Higher significance = higher retention
                adjusted_retention = base_retention + significance * 0.1
                
                # Adjust strength based on rating and significance
                rating_data = self.kg.ratings_df[
                    (self.kg.ratings_df['user_id'] == u_id) & 
                    (self.kg.ratings_df['movie_id'] == movie_id)
                ]
                
                if not rating_data.empty:
                    rating = rating_data.iloc[0]['rating']
                    base_strength = rating / 5.0
                else:
                    base_strength = 0.5
                
                # Higher significance = stronger memory
                individual_strength = base_strength * (1.0 + significance)
                
                # Apply Ebbinghaus forgetting curve
                last_time = self.last_interaction_time.get((u_id, movie_id), 0)
                time_diff = (current_time - last_time) / (24 * 60 * 60)  # days
                
                new_strength = adjusted_retention * np.exp(-time_diff / individual_strength)
                
                # Significant items have higher minimum strength
                min_strength = 0.001 + significance * 0.1
                self.memory_strength[(u_id, movie_id)] = max(min_strength, new_strength)
                
                user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        return user_memories

    def implement_frequency_significance_decay(self, user_id):
        """
        Implement decay based on both usage frequency and item significance.
        
        Args:
            user_id: The user ID to apply decay to
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        user_memories = {}
        
        for (u_id, movie_id), strength in self.memory_strength.items():
            if u_id == user_id:
                # Get interaction count
                interaction_count = self.interaction_counts.get((u_id, movie_id), 0)
                
                # Get significance factor
                significance = self.item_significance.get(movie_id, 0.5)
                
                # Calculate frequency factor (more interactions = stronger memory)
                frequency_factor = min(1.0, interaction_count / 5.0)
                
                # Combined factor considers both frequency and significance
                combined_factor = 0.5 * frequency_factor + 0.5 * significance
                
                # Apply decay based on combined factor
                # Higher combined factor = slower decay
                decay_rate = 0.9 + combined_factor * 0.1  # Between 0.9 and 1.0
                
                # Apply decay
                decayed_strength = strength * decay_rate
                
                # Significant and frequently accessed items have higher floor
                min_strength = 0.001 + combined_factor * 0.1
                self.memory_strength[(u_id, movie_id)] = max(min_strength, decayed_strength)
                
                user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        return user_memories

    def implement_event_based_forgetting(self, user_id, event_type, affected_ids=None):
        """
        Implement forgetting triggered by specific events.
        
        Args:
            user_id: The user ID to apply forgetting to
            event_type: Type of event triggering forgetting ('award_announcement', 'movie_release', etc.)
            affected_ids: Optional list of movie IDs directly affected by the event
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        user_memories = {}
        
        if event_type == 'award_announcement':
            # Award announcements boost memory of award-winning movies
            # and slightly decay memory of non-winning movies in the same category
            
            if affected_ids:
                # Boost memory of award-winning movies
                for movie_id in affected_ids:
                    key = (user_id, movie_id)
                    if key in self.memory_strength:
                        # Boost memory by up to 30%
                        self.memory_strength[key] = min(1.0, self.memory_strength[key] * 1.3)
                        user_memories[movie_id] = self.memory_strength[key]
                        
                        # Update significance score
                        self.item_significance[movie_id] = min(1.0, self.item_significance.get(movie_id, 0.5) + 0.2)
                
                # Slightly decay other movies (forgetting due to attention shift)
                for (u_id, movie_id), strength in self.memory_strength.items():
                    if u_id == user_id and movie_id not in affected_ids:
                        # Mild decay (5%)
                        self.memory_strength[(u_id, movie_id)] = strength * 0.95
                        user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        elif event_type == 'movie_release':
            # New movie releases may cause slight forgetting of older movies
            
            # Apply mild decay to all existing memories
            for (u_id, movie_id), strength in self.memory_strength.items():
                if u_id == user_id:
                    # Get movie age
                    movie_node = f"movie_{movie_id}"
                    
                    # Older movies decay more with new releases
                    if movie_node in self.kg.G and 'release_date' in self.kg.G.nodes[movie_node]:
                        release_date = self.kg.G.nodes[movie_node]['release_date']
                        try:
                            # Parse release date
                            if isinstance(release_date, str):
                                release_year = int(release_date.split('-')[0])
                                current_year = datetime.datetime.now().year
                                
                                # Calculate age factor (older = more decay)
                                age_years = max(0, current_year - release_year)
                                age_factor = min(0.2, age_years * 0.01)  # Up to 20% additional decay for 20+ year old movies
                                
                                decay_rate = 0.95 - age_factor
                            else:
                                decay_rate = 0.95
                        except:
                            decay_rate = 0.95
                    else:
                        decay_rate = 0.95
                    
                    # Apply decay
                    self.memory_strength[(u_id, movie_id)] = strength * decay_rate
                    user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        return user_memories

    def create_context_aware_hybrid_decay(self, user_id, time_weight=0.3, usage_weight=0.2, 
                                        significance_weight=0.3, novelty_weight=0.2):
        """
        Create a hybrid decay function that incorporates context awareness.
        
        Args:
            user_id: The user ID to apply decay to
            time_weight: Weight for time-based decay
            usage_weight: Weight for usage-based decay
            significance_weight: Weight for significance-based factors
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
            
            # Get significance factor
            significance = self.item_significance.get(movie_id, 0.5)
            
            # Adjust time decay by significance (significant items decay slower)
            adjusted_decay_rate = 0.05 * (1.0 - significance * 0.5)  # Between 0.025 and 0.05
            time_decay = math.exp(-adjusted_decay_rate * days_diff)
            
            # Usage-based component
            interaction_count = self.interaction_counts.get((user_id, movie_id), 0)
            usage_factor = min(1.0, interaction_count / 5.0)  # Normalize to [0,1]
            
            # Novelty-based component
            movie_timestamp = self.last_interaction_time.get((user_id, movie_id), 0)
            novelty_factor = 1.0 if movie_timestamp > avg_timestamp else 0.8
            
            # Combine all factors with weights
            hybrid_factor = (
                time_weight * time_decay + 
                usage_weight * usage_factor + 
                significance_weight * significance + 
                novelty_weight * novelty_factor
            )
            
            # Apply decay
            new_strength = strength * hybrid_factor
            
            # Significant items have higher minimum strength
            min_strength = 0.001 + significance * 0.1
            self.memory_strength[(user_id, movie_id)] = max(min_strength, min(1.0, new_strength))
            
            user_memories[movie_id] = self.memory_strength[(user_id, movie_id)]
        
        return user_memories


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
                significance_weight = 0.3
            else:  # Focused taste
                # For active users with focused taste, emphasize usage and significance
                time_weight = 0.3
                usage_weight = 0.5
                novelty_weight = 0.2
                significance_weight = 0.4
        else:  # Low activity user
            if diversity > 0.6:  # Diverse taste
                # For casual users with diverse taste, balanced approach
                time_weight = 0.4
                usage_weight = 0.3
                novelty_weight = 0.3
                significance_weight = 0.3
            else:  # Focused taste
                # For casual users with focused taste, emphasize time decay
                time_weight = 0.5
                usage_weight = 0.3
                novelty_weight = 0.2
                significance_weight = 0.2

        # Return parameters for the context-aware hybrid decay function
        return {
            'time_weight': time_weight,
            'usage_weight': usage_weight,
            'novelty_weight': novelty_weight,
            'significance_weight': significance_weight,
            'forgetting_factor': 0.5
        }
        
    def _calculate_genre_diversity(self, movie_ids):
        """Helper method to calculate genre diversity for a list of movie IDs."""
        if not movie_ids:
            return 0.0
                
        genre_vectors = []
        for movie_id in movie_ids:
            if movie_id in self.kg.movie_features:
                genre_vectors.append(self.kg.movie_features[movie_id])
        
        if len(genre_vectors) < 2:
            return 0.0
                
        # Calculate genre diversity using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(genre_vectors)
        np.fill_diagonal(sim_matrix, 0)
        
        return 1.0 - np.mean(sim_matrix)
    def integrate_context_aware_forgetting(self, recommendation_algorithm='hybrid', 
                                         forgetting_parameters=None, 
                                         context_factors=None):
        """
        Integrate context-aware forgetting into the recommendation pipeline.

        Args:
            recommendation_algorithm: Algorithm type or function for recommendations
            forgetting_parameters: Parameters for the forgetting mechanism
            context_factors: Dictionary of weights for context factors

        Returns:
            Function that generates recommendations with context-aware forgetting
        """
        if forgetting_parameters is None:
            forgetting_parameters = {
                'time_weight': 0.3,
                'usage_weight': 0.2,
                'significance_weight': 0.3,
                'novelty_weight': 0.2,
                'forgetting_factor': 0.5
            }

        if context_factors is None:
            context_factors = {
                'award_impact': 0.2,
                'director_impact': 0.15,
                'box_office_impact': 0.1,
                'age_impact': 0.05
            }

        def context_aware_recommendations(user_id, n=10):
            # Get personalized parameters if needed
            if user_id not in self.user_activity_patterns:
                user_params = self.personalize_forgetting_parameters(user_id)

                # Add significance weight if not present
                if 'significance_weight' not in user_params:
                    user_params['significance_weight'] = forgetting_parameters['significance_weight']
            else:
                user_params = forgetting_parameters

            # Apply context-aware hybrid decay
            self.create_context_aware_hybrid_decay(
                user_id, 
                time_weight=user_params.get('time_weight', 0.3),
                usage_weight=user_params.get('usage_weight', 0.2),
                significance_weight=user_params.get('significance_weight', 0.3),
                novelty_weight=user_params.get('novelty_weight', 0.2)
            )

            # Get base recommendations
            if isinstance(recommendation_algorithm, str):
                if recommendation_algorithm == 'personalized':
                    base_recs = self.kg.get_personalized_recommendations(user_id, n=n*2)
                    scores = {mid: (n*2 - i) / (n*2) for i, mid in enumerate(base_recs)}
                elif recommendation_algorithm == 'graph_based':
                    base_recs = self.kg.get_graph_based_recommendations(user_id, n=n*2)
                    scores = {mid: (n*2 - i) / (n*2) for i, mid in enumerate(base_recs)}
                elif recommendation_algorithm == 'hybrid':
                    base_recs = self.kg.get_hybrid_recommendations(user_id, n=n*2)
                    scores = {mid: (n*2 - i) / (n*2) for i, mid in enumerate(base_recs)}
                elif recommendation_algorithm == 'path_based':
                    base_recs = self.kg.get_path_based_recommendations(user_id, n=n*2)
                    scores = {mid: (n*2 - i) / (n*2) for i, mid in enumerate(base_recs)}
                elif recommendation_algorithm == 'multi_aspect':
                    base_recs = self.kg.get_multi_aspect_recommendations(user_id, n=n*2)
                    scores = {mid: (n*2 - i) / (n*2) for i, mid in enumerate(base_recs)}
                else:
                    base_recs = self.kg.get_hybrid_recommendations(user_id, n=n*2)
                    scores = {mid: (n*2 - i) / (n*2) for i, mid in enumerate(base_recs)}
            else:
                # Custom recommendation function
                scores = recommendation_algorithm(user_id)

            # Adjust scores based on context factors
            adjusted_scores = {}

            for movie_id, score in scores.items():
                # Get memory strength
                memory_strength = self.memory_strength.get((user_id, movie_id), 1.0)

                # Get item significance
                significance = self.item_significance.get(movie_id, 0.5)

                # Apply forgetting factor (boosts novel items)
                forgetting_boost = (1.0 - memory_strength) * user_params.get('forgetting_factor', 0.5)

                # Apply significance factor
                significance_boost = significance * context_factors.get('award_impact', 0.2)

                # Get additional context factors if available
                movie_node = f"movie_{movie_id}"

                director_boost = 0.0
                box_office_boost = 0.0
                age_penalty = 0.0

                if movie_node in self.kg.G:
                    # Director impact
                    for neighbor in self.kg.G.neighbors(movie_node):
                        if neighbor.startswith("director_"):
                            director_boost = context_factors.get('director_impact', 0.15)
                            break
                        
                    # Box office impact
                    if 'box_office' in self.kg.G.nodes[movie_node]:
                        try:
                            box_office = float(self.kg.G.nodes[movie_node]['box_office'])
                            box_office_boost = min(0.1, box_office / 1000000000 * 0.1) * context_factors.get('box_office_impact', 0.1)
                        except (ValueError, TypeError):
                            pass
                        
                    # Age penalty (newer movies get slight boost)
                    if 'release_date' in self.kg.G.nodes[movie_node]:
                        release_date = self.kg.G.nodes[movie_node]['release_date']
                        try:
                            if isinstance(release_date, str):
                                release_year = int(release_date.split('-')[0])
                                current_year = datetime.datetime.now().year

                                age_years = max(0, current_year - release_year)
                                age_penalty = min(0.05, age_years * 0.005) * context_factors.get('age_impact', 0.05)
                        except:
                            pass
                        
                # Combine all factors
                context_adjustment = significance_boost + director_boost + box_office_boost - age_penalty

                # Apply adjustments to score
                adjusted_scores[movie_id] = score * (1.0 + forgetting_boost + context_adjustment)

            # Sort by adjusted score and return top n
            recommendations = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)[:n]
            return [movie_id for movie_id, _ in recommendations]

        return context_aware_recommendations