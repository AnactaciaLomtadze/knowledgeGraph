import os
import sys
import numpy as np
sys.path.append('.')  # Add project root to path

# Add the missing method to the class
with open('src/context_forgetting.py', 'r') as f:
    code = f.read()

# Add the personalize_forgetting_parameters method before the integrate_context_aware_forgetting method
method_code = '''
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
'''

# Find the position to insert the method (before integrate_context_aware_forgetting)
insert_position = code.find('    def integrate_context_aware_forgetting')
if insert_position != -1:
    modified_code = code[:insert_position] + method_code + code[insert_position:]
    
    # Save the modified file
    with open('src/context_forgetting.py', 'w') as f:
        f.write(modified_code)
    
    print("Added personalize_forgetting_parameters method to ContextAwareForgettingMechanism")
else:
    print("Couldn't find the insertion point in context_forgetting.py")
