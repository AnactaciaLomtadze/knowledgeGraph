import os
import sys
sys.path.append('.')  # Add project root to path

from src.evaluation_metrics import EvaluationMetrics

# Define a subclass that adds the missing method
class FixedEvaluationMetrics(EvaluationMetrics):
    def calculate_diversity(self, recommendations, k=10):
        """
        Calculate diversity of recommendations based on genre similarity.
        
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
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(genre_vectors)
        import numpy as np
        np.fill_diagonal(sim_matrix, 0)  # Exclude self-similarities
        
        # Diversity is the inverse of average similarity
        diversity = 1.0 - np.mean(sim_matrix)
        
        return diversity

# Now modify the compare_forgetting.py file
with open('scripts/compare_forgetting.py', 'r') as f:
    code = f.read()

# Replace import statement
code = code.replace(
    'from src.evaluation_metrics import EvaluationMetrics',
    'from fix_evaluation_metrics import FixedEvaluationMetrics as EvaluationMetrics'
)

# Save the modified file
with open('scripts/compare_forgetting.py', 'w') as f:
    f.write(code)

print("Added calculate_diversity method and updated compare_forgetting.py")
