#src/knowledge_graph.py
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json
import datetime
import logging
from tqdm import tqdm

class MovieLensKnowledgeGraph:
    """
    A knowledge graph representation of MovieLens data with recommendations functionality.
    
    This class builds a graph-based representation of user-movie interactions,
    movie-movie relationships based on similarity, and provides methods for
    generating personalized recommendations.
    """
    def __init__(self, data_path='./ml-100k'):
        """
        Initialize the MovieLens Knowledge Graph.
        
        Args:
            data_path: Path to the MovieLens dataset
        """
        self.data_path = data_path
        self.G = nx.Graph()  
        self.user_profiles = {}
        self.movie_features = {}
        self.ratings_df = None
        self.users_df = None
        self.movies_df = None
        self.similarity_matrix = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('MovieLensKG')
        
    def load_data(self):
        """Load the MovieLens dataset."""
        try:
            ratings_path = os.path.join(self.data_path, 'u.data')
            self.ratings_df = pd.read_csv(
                ratings_path, 
                sep='\t', 
                names=['user_id', 'movie_id', 'rating', 'timestamp']
            )
            
            users_path = os.path.join(self.data_path, 'u.user')
            self.users_df = pd.read_csv(
                users_path,
                sep='|',
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
            )
            
            movies_path = os.path.join(self.data_path, 'u.item')
            self.movies_df = pd.read_csv(
                movies_path,
                sep='|',
                encoding='latin-1',
                names=['movie_id', 'title', 'release_date', 'video_release_date', 
                      'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                      'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                      'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                      'Thriller', 'War', 'Western']
            )
            
            self.logger.info(f"Loaded {len(self.ratings_df)} ratings from {self.ratings_df['user_id'].nunique()} users on {self.ratings_df['movie_id'].nunique()} movies")
            return True
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
        
    def build_knowledge_graph(self):
        """Build the knowledge graph from the MovieLens data."""
        if self.ratings_df is None:
            if not self.load_data():
                self.logger.error("Cannot build knowledge graph without data")
                return False
        
        self.logger.info("Building knowledge graph...")
        
        # Add user nodes
        for _, user in tqdm(self.users_df.iterrows(), total=len(self.users_df), desc="Adding user nodes"):
            self.G.add_node(
                f"user_{user['user_id']}", 
                type='user',
                age=user['age'],
                gender=user['gender'],
                occupation=user['occupation']
            )
        
        # Add movie nodes
        for _, movie in tqdm(self.movies_df.iterrows(), total=len(self.movies_df), desc="Adding movie nodes"):
            genre_features = movie[5:].values.astype(int)  # All genre columns
            
            self.G.add_node(
                f"movie_{movie['movie_id']}", 
                type='movie',
                title=movie['title'],
                release_date=movie['release_date']
            )
            
            self.movie_features[movie['movie_id']] = genre_features
    
        # Add user-movie edges (ratings)
        for _, rating in tqdm(self.ratings_df.iterrows(), total=len(self.ratings_df), desc="Adding rating edges"):
            user_id = rating['user_id']
            movie_id = rating['movie_id']
            rating_value = rating['rating']
            timestamp = rating['timestamp']
            
            rating_time = datetime.datetime.fromtimestamp(timestamp)

            self.G.add_edge(
                f"user_{user_id}", 
                f"movie_{movie_id}", 
                weight=rating_value,
                timestamp=timestamp,
                rating_time=rating_time
            )
        
        # Add movie-movie similarity edges
        self._add_movie_similarity_edges()
        
        # Build user profiles for recommendations
        self._build_user_profiles()
        
        self.logger.info(f"Knowledge graph built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return True
        
    def _add_movie_similarity_edges(self, threshold=0.7):
        """
        Add movie-movie edges based on genre similarity.
        
        Args:
            threshold: Minimum similarity score to create an edge
        """
        self.logger.info("Adding movie similarity edges...")
        
        movie_ids = list(self.movie_features.keys())
        feature_matrix = np.array([self.movie_features[mid] for mid in movie_ids])
        
        # Calculate cosine similarity matrix between movies
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
        # Add edges for sufficiently similar movies
        edge_count = 0
        for i in tqdm(range(len(movie_ids)), desc="Calculating movie similarities"):
            for j in range(i+1, len(movie_ids)):
                similarity = self.similarity_matrix[i, j]
                if similarity >= threshold:
                    self.G.add_edge(
                        f"movie_{movie_ids[i]}", 
                        f"movie_{movie_ids[j]}", 
                        weight=similarity,
                        relation_type='similar'
                    )
                    edge_count += 1
        
        self.logger.info(f"Added {edge_count} movie similarity edges")
    
    def _build_user_profiles(self):
        """Build user profiles based on their ratings."""
        self.logger.info("Building user profiles...")
        
        user_ratings = self.ratings_df.groupby('user_id')
        
        for user_id, ratings in tqdm(user_ratings, total=len(user_ratings), desc="Building user profiles"):
            # Calculate average rating for this user
            avg_rating = ratings['rating'].mean()
            
            # Get all movies rated by this user and their ratings
            rated_movies = ratings[['movie_id', 'rating']].values
            
            # Calculate genre preferences
            genre_preferences = np.zeros(19)
            genre_counts = np.zeros(19)
            
            for movie_id, rating in rated_movies:
                if movie_id in self.movie_features:
                    # Get movie genres
                    movie_genres = self.movie_features[movie_id]
                    
                    # Normalize rating relative to user's average
                    normalized_rating = rating - avg_rating
                    
                    # Update genre preferences
                    for i, has_genre in enumerate(movie_genres):
                        if has_genre:
                            genre_preferences[i] += normalized_rating
                            genre_counts[i] += 1
          
            # Avoid division by zero
            genre_counts[genre_counts == 0] = 1
            
            # Calculate average preference for each genre
            genre_preferences = genre_preferences / genre_counts
            
            # Store user profile
            self.user_profiles[user_id] = {
                'avg_rating': avg_rating,
                'genre_preferences': genre_preferences,
                'rated_movies': set(ratings['movie_id'].values),
                'rating_count': len(ratings),
                'last_rating_time': ratings['timestamp'].max()
            }
    
    def get_personalized_recommendations(self, user_id, n=10):
        """
        Get personalized movie recommendations for a user based on content similarity.
        
        Args:
            user_id: The user ID to generate recommendations for
            n: Number of recommendations to return
            
        Returns:
            A list of recommended movie IDs
        """
        if user_id not in self.user_profiles:
            self.logger.warning(f"User {user_id} not found in profiles")
            return []
        
        user_profile = self.user_profiles[user_id]
        rated_movies = user_profile['rated_movies']
        genre_preferences = user_profile['genre_preferences']
        
        movie_scores = []
        
        for movie_id, features in self.movie_features.items():
            if movie_id not in rated_movies:
                # Calculate content-based score
                content_score = np.dot(genre_preferences, features)
                
                # Apply popularity factor
                movie_rating_count = len(self.ratings_df[self.ratings_df['movie_id'] == movie_id])
                popularity_factor = np.log1p(movie_rating_count) / 10
                
                # Combine scores
                final_score = content_score + popularity_factor
                
                movie_scores.append((movie_id, final_score))
        
        # Sort by score and return top n
        recommendations = sorted(movie_scores, key=lambda x: x[1], reverse=True)[:n]
        return [movie_id for movie_id, _ in recommendations]
    
    def get_graph_based_recommendations(self, user_id, n=10, depth=2):
        """
        Get recommendations using graph traversal.
        
        Args:
            user_id: The user ID to generate recommendations for
            n: Number of recommendations to return
            depth: How many hops to traverse in the graph
            
        Returns:
            A list of recommended movie IDs
        """
        if f"user_{user_id}" not in self.G:
            self.logger.warning(f"User {user_id} not found in graph")
            return []
        
        user_node = f"user_{user_id}"
        rated_movies = set()
        candidate_scores = defaultdict(float)
        
        # Get movies already rated by user
        for neighbor in self.G.neighbors(user_node):
            if neighbor.startswith("movie_"):
                movie_id = int(neighbor.split("_")[1])
                rated_movies.add(movie_id)
        
        # Initialize BFS
        paths = [(user_node, [])]
        visited = {user_node}
        
        # Explore graph up to specified depth
        for _ in range(depth):
            new_paths = []
            for node, path in paths:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [(node, neighbor)]
                        new_paths.append((neighbor, new_path))
                        
                        # If we found a movie node
                        if neighbor.startswith("movie_"):
                            movie_id = int(neighbor.split("_")[1])
                            if movie_id not in rated_movies:
                                # Calculate path score based on edge weights
                                path_weight = 1.0
                                for i in range(len(new_path)):
                                    n1, n2 = new_path[i]
                                    edge_weight = self.G.edges[n1, n2].get('weight', 1.0)
                                    path_weight *= edge_weight
                                
                                path_score = path_weight / len(new_path)
                                candidate_scores[movie_id] += path_score
            
            paths = new_paths
        
        # Sort by score and return top n
        recommendations = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [movie_id for movie_id, _ in recommendations]
    
    def get_hybrid_recommendations(self, user_id, n=10, alpha=0.5):
        """
        Get hybrid recommendations combining content-based and graph-based approaches.
        
        Args:
            user_id: The user ID to generate recommendations for
            n: Number of recommendations to return
            alpha: Weight for content-based recommendations (1-alpha for graph-based)
            
        Returns:
            A list of recommended movie IDs
        """
        # Get content-based recommendations
        content_recs = self.get_personalized_recommendations(user_id, n=n*2)
        content_scores = {movie_id: (n*2 - i)/n*2 for i, movie_id in enumerate(content_recs)}
        
        # Get graph-based recommendations
        graph_recs = self.get_graph_based_recommendations(user_id, n=n*2)
        graph_scores = {movie_id: (n*2 - i)/n*2 for i, movie_id in enumerate(graph_recs)}
        
        # Combine scores
        combined_scores = defaultdict(float)
        all_movies = set(content_scores.keys()).union(set(graph_scores.keys()))
        
        for movie_id in all_movies:
            content_score = content_scores.get(movie_id, 0)
            graph_score = graph_scores.get(movie_id, 0)
            combined_scores[movie_id] = alpha * content_score + (1 - alpha) * graph_score
        
        # Sort and return top n
        recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [movie_id for movie_id, _ in recommendations]
    
    def visualize_subgraph(self, center_node, depth=1, filename=None):
        """
        Visualize a subgraph around a specific node.
        
        Args:
            center_node: Center node (e.g., "user_12" or "movie_456")
            depth: How many hops to include
            filename: If specified, save figure to this file
        """
        nodes = {center_node}
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.G.neighbors(node))
            nodes.update(new_nodes)
        
        subgraph = self.G.subgraph(nodes)
        
        node_colors = []
        for node in subgraph.nodes():
            if node.startswith('user'):
                node_colors.append('skyblue')
            else:
                node_colors.append('lightgreen')
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2)
        nx.draw_networkx_labels(subgraph, pos, font_size=8)
        plt.title(f"Subgraph around {center_node}")
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
            
    def create_temporal_train_test_split(self, test_days=30):
        """
        Create train-test split based on time (train on earlier data, test on later data).
        
        Args:
            test_days: Number of days at the end to use as test set
            
        Returns:
            Dictionary mapping user_id to set of test movie_ids
        """
        if self.ratings_df is None:
            if not self.load_data():
                return {}
                
        # Sort ratings by timestamp
        self.ratings_df = self.ratings_df.sort_values('timestamp')
        
        # Calculate cutoff timestamp
        min_timestamp = self.ratings_df['timestamp'].min()
        max_timestamp = self.ratings_df['timestamp'].max()
        
        # Get approx time range in days
        time_range_seconds = max_timestamp - min_timestamp
        time_range_days = time_range_seconds / (24 * 60 * 60)
        
        # Adjust if requested test period is longer than available data
        test_ratio = min(1.0, test_days / time_range_days)
        cutoff_timestamp = min_timestamp + (1 - test_ratio) * time_range_seconds
        
        # Split data
        train_data = self.ratings_df[self.ratings_df['timestamp'] <= cutoff_timestamp]
        test_data = self.ratings_df[self.ratings_df['timestamp'] > cutoff_timestamp]
        
        self.logger.info(f"Temporal split: {len(train_data)} training and {len(test_data)} testing ratings")
        
        # Create test set mapping
        test_set = {}
        for user_id in test_data['user_id'].unique():
            if user_id in train_data['user_id'].unique():  # Only include users in both sets
                test_set[user_id] = set(test_data[test_data['user_id'] == user_id]['movie_id'])
        
        # Update knowledge graph with training data only
        self.ratings_df = train_data
        
        # Rebuild user profiles with training data only
        self._build_user_profiles()
        
        return test_set
    
    def get_recommendations(self, user_id, method='hybrid', n=10):
        """
        Get recommendations using the specified method.
        
        Args:
            user_id: The user ID to generate recommendations for
            method: 'content', 'graph', or 'hybrid'
            n: Number of recommendations to return
            
        Returns:
            A list of recommended movie IDs
        """
        if method == 'content':
            return self.get_personalized_recommendations(user_id, n)
        elif method == 'graph':
            return self.get_graph_based_recommendations(user_id, n)
        elif method == 'hybrid':
            return self.get_hybrid_recommendations(user_id, n)
        else:
            self.logger.warning(f"Unknown recommendation method: {method}, using hybrid")
            return self.get_hybrid_recommendations(user_id, n)
        
    def load_external_knowledge(self, enriched_movies_file):
        """Load enriched movie data from external knowledge sources."""
        try:
            self.logger.info(f"Loading external knowledge from {enriched_movies_file}")
            self.enriched_movies_df = pd.read_csv(enriched_movies_file)

            # Convert string-encoded lists back to lists
            for col in ['directors', 'actors', 'genres', 'awards']:
                if col in self.enriched_movies_df.columns:
                    self.enriched_movies_df[col] = self.enriched_movies_df[col].apply(
                        lambda x: json.loads(x) if isinstance(x, str) and pd.notna(x) else [])

            self.logger.info(f"Loaded external knowledge for {len(self.enriched_movies_df)} movies")
            return True
        except Exception as e:
            self.logger.error(f"Error loading external knowledge: {e}")
            return False

    def build_knowledge_graph_with_external_data(self):
        """Build an enhanced knowledge graph including external knowledge."""
        if not self.build_knowledge_graph():
            return False

        if not hasattr(self, 'enriched_movies_df') or self.enriched_movies_df is None:
            self.logger.warning("No external knowledge data loaded. Using basic knowledge graph.")
            return True

        self.logger.info("Enhancing knowledge graph with external data...")

        # Add new node types
        actor_nodes = {}
        director_nodes = {}
        award_nodes = {}

        # Process each enriched movie
        for _, movie in tqdm(self.enriched_movies_df.iterrows(), total=len(self.enriched_movies_df), 
                          desc="Adding external knowledge"):
            movie_id = movie['movie_id']
            movie_node = f"movie_{movie_id}"

            # Skip if movie node doesn't exist in the graph
            if movie_node not in self.G:
                continue
            
            # Add external IDs to movie node
            if pd.notna(movie['dbpedia_uri']):
                self.G.nodes[movie_node]['dbpedia_uri'] = movie['dbpedia_uri']

            if pd.notna(movie['wikidata_id']):
                self.G.nodes[movie_node]['wikidata_id'] = movie['wikidata_id']

            # Add directors
            for director in movie['directors']:
                director_id = director.get('uri', '').split('/')[-1]
                if not director_id:
                    continue

                director_node = f"director_{director_id}"

                # Add director node if not already in graph
                if director_node not in director_nodes:
                    self.G.add_node(
                        director_node,
                        type='director',
                        name=director.get('name', 'Unknown'),
                        uri=director.get('uri')
                    )
                    director_nodes[director_node] = True

                # Add directed_by edge
                self.G.add_edge(
                    movie_node,
                    director_node,
                    relation_type='directed_by'
                )

            # Add actors
            for actor in movie['actors']:
                actor_id = actor.get('uri', '').split('/')[-1]
                if not actor_id:
                    continue

                actor_node = f"actor_{actor_id}"

                # Add actor node if not already in graph
                if actor_node not in actor_nodes:
                    self.G.add_node(
                        actor_node,
                        type='actor',
                        name=actor.get('name', 'Unknown'),
                        uri=actor.get('uri')
                    )
                    actor_nodes[actor_node] = True

                # Add acted_in edge
                self.G.add_edge(
                    movie_node,
                    actor_node,
                    relation_type='acted_in'
                )

            # Add awards
            for award in movie['awards']:
                award_id = award.get('uri', '').split('/')[-1]
                if not award_id:
                    continue

                award_node = f"award_{award_id}"

                # Add award node if not already in graph
                if award_node not in award_nodes:
                    self.G.add_node(
                        award_node,
                        type='award',
                        name=award.get('name', 'Unknown'),
                        uri=award.get('uri')
                    )
                    award_nodes[award_node] = True

                # Add received_award edge
                self.G.add_edge(
                    movie_node,
                    award_node,
                    relation_type='received_award'
                )

            # Add budget, box office, runtime if available
            if pd.notna(movie['budget']):
                self.G.nodes[movie_node]['budget'] = movie['budget']

            if pd.notna(movie['box_office']):
                self.G.nodes[movie_node]['box_office'] = movie['box_office']

            if pd.notna(movie['runtime']):
                self.G.nodes[movie_node]['runtime'] = movie['runtime']

        self.logger.info(f"Enhanced knowledge graph now has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        self.logger.info(f"Added {len(director_nodes)} directors, {len(actor_nodes)} actors, and {len(award_nodes)} awards")

        return True  

    def get_path_based_recommendations(self, user_id, n=10, max_path_length=3):
        """
        Get recommendations based on paths through the knowledge graph.

        This method considers paths like:
        user -> movie -> actor -> movie
        user -> movie -> director -> movie

        Args:
            user_id: The user ID to generate recommendations for
            n: Number of recommendations to return
            max_path_length: Maximum path length to consider

        Returns:
            A list of recommended movie IDs
        """
        if f"user_{user_id}" not in self.G:
            self.logger.warning(f"User {user_id} not found in graph")
            return []

        user_node = f"user_{user_id}"
        rated_movies = set()
        candidate_scores = defaultdict(float)

        # Get movies already rated by user
        for neighbor in self.G.neighbors(user_node):
            if neighbor.startswith("movie_"):
                movie_id = int(neighbor.split("_")[1])
                rated_movies.add(movie_id)

        # Initialize BFS
        paths = [(user_node, [])]
        visited = {user_node}

        # Explore graph up to specified path length
        for _ in range(max_path_length):
            new_paths = []
            for node, path in paths:
                for neighbor in self.G.neighbors(node):
                    # Skip already visited nodes for this path
                    if neighbor in [p[1] for p in path]:
                        continue

                    new_path = path + [(node, neighbor)]
                    new_paths.append((neighbor, new_path))

                    # If we found a movie node that the user hasn't rated
                    if neighbor.startswith("movie_"):
                        movie_id = int(neighbor.split("_")[1])
                        if movie_id not in rated_movies:
                            # Calculate path score
                            path_score = self._calculate_path_score(new_path)
                            candidate_scores[movie_id] += path_score

                paths = new_paths

        # Sort by score and return top n
        recommendations = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [movie_id for movie_id, _ in recommendations]

    def _calculate_path_score(self, path):
        """
        Calculate the score for a path in the knowledge graph.
        
        Args:
            path: List of (node1, node2) tuples representing edges in the path
            
        Returns:
            Path score (higher is better)
        """
        path_score = 1.0
        path_length = len(path)
        
        # Path score decreases with length
        path_score *= (1.0 / path_length)
        
        # Different weights for different types of connections
        for i in range(path_length):
            node1, node2 = path[i]
            
            # Check the relationship type
            edge_data = self.G.get_edge_data(node1, node2) or {}
            relation_type = edge_data.get('relation_type', '')
            
            # Higher weights for more significant relationships
            if relation_type == 'acted_in':
                path_score *= 1.2
            elif relation_type == 'directed_by':
                path_score *= 1.5
            elif relation_type == 'received_award':
                path_score *= 2.0
            elif relation_type == 'similar':
                path_score *= 1.3
            
            # Consider edge weight if it exists
            if 'weight' in edge_data:
                path_score *= edge_data['weight']
        
        return path_score
    
    def get_multi_aspect_recommendations(self, user_id, n=10, aspects=None):
        """
        Get recommendations considering multiple aspects of the knowledge graph.
        
        Args:
            user_id: The user ID to generate recommendations for
            n: Number of recommendations to return
            aspects: Dictionary with aspect weights, e.g., {'content': 0.3, 'actor': 0.3, 'director': 0.2, 'award': 0.2}
            
        Returns:
            A list of recommended movie IDs
        """
        if aspects is None:
            aspects = {
                'content': 0.4,  # Content-based similarity
                'actor': 0.3,    # Actor-based connections
                'director': 0.2, # Director-based connections
                'award': 0.1     # Award-based significance
            }
        
        # Ensure user exists
        if f"user_{user_id}" not in self.G:
            self.logger.warning(f"User {user_id} not found in graph")
            return []
        
        if user_id not in self.user_profiles:
            self.logger.warning(f"User {user_id} not found in profiles")
            return []
            
        user_profile = self.user_profiles[user_id]
        rated_movies = user_profile['rated_movies']
        
        # Get all candidate movies (not rated by the user)
        candidate_movies = set()
        for movie_id in self.movie_features.keys():
            if movie_id not in rated_movies:
                candidate_movies.add(movie_id)
        
        # Calculate scores for each aspect
        content_scores = self._calculate_content_scores(user_id, candidate_movies)
        actor_scores = self._calculate_actor_scores(user_id, candidate_movies)
        director_scores = self._calculate_director_scores(user_id, candidate_movies)
        award_scores = self._calculate_award_scores(candidate_movies)
        
        # Normalize scores for each aspect
        content_scores = self._normalize_scores(content_scores)
        actor_scores = self._normalize_scores(actor_scores)
        director_scores = self._normalize_scores(director_scores)
        award_scores = self._normalize_scores(award_scores)
        
        # Combine scores using aspect weights
        final_scores = {}
        for movie_id in candidate_movies:
            final_scores[movie_id] = (
                aspects['content'] * content_scores.get(movie_id, 0) +
                aspects['actor'] * actor_scores.get(movie_id, 0) +
                aspects['director'] * director_scores.get(movie_id, 0) +
                aspects['award'] * award_scores.get(movie_id, 0)
            )
        
        # Sort by score and return top n
        recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [movie_id for movie_id, _ in recommendations]
    
    def _calculate_content_scores(self, user_id, candidate_movies):
        """Calculate content-based similarity scores."""
        scores = {}
        user_profile = self.user_profiles[user_id]
        genre_preferences = user_profile['genre_preferences']
        
        for movie_id in candidate_movies:
            if movie_id in self.movie_features:
                # Get movie genres
                movie_genres = self.movie_features[movie_id]
                
                # Calculate content score as dot product of preferences and genres
                score = np.dot(genre_preferences, movie_genres)
                
                # Apply popularity factor
                movie_rating_count = len(self.ratings_df[self.ratings_df['movie_id'] == movie_id])
                popularity_factor = np.log1p(movie_rating_count) / 10
                
                scores[movie_id] = score + popularity_factor
        
        return scores
    
    def _calculate_actor_scores(self, user_id, candidate_movies):
        """Calculate scores based on actors the user may like."""
        scores = {}
        user_node = f"user_{user_id}"
        
        # Get movies the user has rated highly
        rated_movies = []
        for neighbor in self.G.neighbors(user_node):
            if neighbor.startswith("movie_"):
                edge_data = self.G.get_edge_data(user_node, neighbor)
                rating = edge_data.get('weight', 0)
                if rating >= 4.0:  # Consider only highly rated movies
                    rated_movies.append(neighbor)
        
        # Find actors in those movies
        preferred_actors = set()
        for movie_node in rated_movies:
            for neighbor in self.G.neighbors(movie_node):
                if neighbor.startswith("actor_"):
                    preferred_actors.add(neighbor)
        
        # Score candidate movies based on shared actors
        for movie_id in candidate_movies:
            movie_node = f"movie_{movie_id}"
            
            if movie_node not in self.G:
                continue
                
            movie_actors = set()
            for neighbor in self.G.neighbors(movie_node):
                if neighbor.startswith("actor_"):
                    movie_actors.add(neighbor)
            
            # Score based on number of shared actors
            shared_actors = preferred_actors.intersection(movie_actors)
            scores[movie_id] = len(shared_actors)
        
        return scores
    
    def _calculate_director_scores(self, user_id, candidate_movies):
        """Calculate scores based on directors the user may like."""
        scores = {}
        user_node = f"user_{user_id}"
        
        # Get movies the user has rated highly
        rated_movies = []
        for neighbor in self.G.neighbors(user_node):
            if neighbor.startswith("movie_"):
                edge_data = self.G.get_edge_data(user_node, neighbor)
                rating = edge_data.get('weight', 0)
                if rating >= 4.0:  # Consider only highly rated movies
                    rated_movies.append(neighbor)
        
        # Find directors of those movies
        preferred_directors = set()
        for movie_node in rated_movies:
            for neighbor in self.G.neighbors(movie_node):
                if neighbor.startswith("director_"):
                    preferred_directors.add(neighbor)
        
        # Score candidate movies based on shared directors
        for movie_id in candidate_movies:
            movie_node = f"movie_{movie_id}"
            
            if movie_node not in self.G:
                continue
                
            movie_directors = set()
            for neighbor in self.G.neighbors(movie_node):
                if neighbor.startswith("director_"):
                    movie_directors.add(neighbor)
            
            # Score based on number of shared directors (higher weight than actors)
            shared_directors = preferred_directors.intersection(movie_directors)
            scores[movie_id] = 2.0 * len(shared_directors)  # Directors have higher impact
        
        return scores
    
    def _calculate_award_scores(self, candidate_movies):
        """Calculate scores based on awards received by movies."""
        scores = {}
        
        for movie_id in candidate_movies:
            movie_node = f"movie_{movie_id}"
            
            if movie_node not in self.G:
                continue
                
            award_count = 0
            for neighbor in self.G.neighbors(movie_node):
                if neighbor.startswith("award_"):
                    award_count += 1
            
            # Score based on number of awards
            scores[movie_id] = min(1.0, award_count / 5.0)  # Cap at 1.0 for 5 or more awards
        
        return scores
    
    def _normalize_scores(self, scores):
        """Normalize scores to range [0, 1]."""
        if not scores:
            return {}
            
        max_score = max(scores.values())
        min_score = min(scores.values())
        
        if max_score == min_score:
            return {k: 1.0 for k in scores}
        
        return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}
    
    def get_improved_hybrid_recommendations(self, user_id, n=10, popularity_weight=0.4):
        """
        Improved hybrid recommendations incorporating popularity more effectively.
        """
        # Get personalized content-based recommendations
        content_recs = self.get_personalized_recommendations(user_id, n=n*2)
        content_scores = {mid: (n*2 - i)/(n*2) for i, mid in enumerate(content_recs)}

        # Get popular recommendations
        popular_recs = self._recommend_popular(user_id, n=n*2)
        popular_scores = {mid: (n*2 - i)/(n*2) for i, mid in enumerate(popular_recs)}

        # Get memory-adjusted scores
        memory_strengths = {}
        if hasattr(self, 'fm') and self.fm is not None:
            memory_strengths = {mid: self.fm.memory_strength.get((user_id, mid), 0.5) 
                               for mid in set(content_recs).union(set(popular_recs))}
        else:
            memory_strengths = {mid: 0.5 for mid in set(content_recs).union(set(popular_recs))}

        # Combine scores with proper weighting
        combined_scores = {}
        all_movies = set(content_scores.keys()).union(set(popular_scores.keys()))

        for movie_id in all_movies:
            content_score = content_scores.get(movie_id, 0)
            popular_score = popular_scores.get(movie_id, 0)
            memory_strength = memory_strengths.get(movie_id, 0.5)

            # Calculate novelty factor (inverse of memory strength)
            novelty_factor = 1.0 - memory_strength

            # Weighted combination
            combined_scores[movie_id] = (
                (1 - popularity_weight) * content_score + 
                popularity_weight * popular_score +
                0.2 * novelty_factor  # Boost for novel items
            )

        # Sort and return top n
        recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [movie_id for movie_id, _ in recommendations]
    
    def get_knowledge_enhanced_recommendations(self, user_id, n=10):
        """
        Enhanced recommendations using external knowledge.
        """
        # Get base personalized recommendations
        base_recs = self.get_personalized_recommendations(user_id, n=n*2)
        base_scores = {mid: (n*2 - i)/(n*2) for i, mid in enumerate(base_recs)}

        # Enhance scores with external knowledge
        enhanced_scores = {}
        for movie_id, score in base_scores.items():
            movie_node = f"movie_{movie_id}"
            knowledge_boost = 0.0

            # Check for awards (quality signal)
            if movie_node in self.G:
                award_count = sum(1 for neighbor in self.G.neighbors(movie_node) 
                                 if neighbor.startswith("award_"))
                knowledge_boost += min(0.3, award_count * 0.1)  # Cap at 0.3

            # Check for well-regarded directors
            director_boost = 0.0
            if movie_node in self.G:
                for neighbor in self.G.neighbors(movie_node):
                    if neighbor.startswith("director_"):
                        # Count how many highly-rated movies this director has
                        director_rating_sum = 0
                        director_movie_count = 0

                        for connected_movie in self.G.neighbors(neighbor):
                            if connected_movie.startswith("movie_") and connected_movie != movie_node:
                                # Get average rating of this movie
                                connected_movie_id = int(connected_movie.split("_")[1])
                                movie_ratings = self.ratings_df[self.ratings_df['movie_id'] == connected_movie_id]
                                if not movie_ratings.empty:
                                    avg_rating = movie_ratings['rating'].mean()
                                    director_rating_sum += avg_rating
                                    director_movie_count += 1

                        if director_movie_count > 0:
                            director_avg_rating = director_rating_sum / director_movie_count
                            # Director bonus (up to 0.2 additional boost)
                            director_boost = max(0, min(0.2, (director_avg_rating - 3.0) / 2.0))
                        break  # Only consider the first director
                    
            knowledge_boost += director_boost

            # Apply knowledge boost to score
            enhanced_scores[movie_id] = score * (1.0 + knowledge_boost)

        # Sort and return top n
        recommendations = sorted(enhanced_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [movie_id for movie_id, _ in recommendations]
    
    def get_segment_adaptive_recommendations(self, user_id, n=10):
        """
        Adaptive recommendations that change strategy based on user segment.
        """
        # Determine user segment
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        activity_level = len(user_ratings)

        # Calculate diversity of taste
        if user_id in self.user_profiles:
            genre_prefs = self.user_profiles[user_id]['genre_preferences'] 
            # Higher values indicate more diverse tastes
            diversity = np.sum(genre_prefs * (1 - genre_prefs))
        else:
            diversity = 0.5  # Default

        # Get recency
        if user_id in self.user_profiles:
            last_time = self.user_profiles[user_id].get('last_rating_time', 0)
            current_time = datetime.datetime.now().timestamp()
            recency = (current_time - last_time) / (24 * 60 * 60)  # days
        else:
            recency = 100  # Default to somewhat old

        # Adapt strategy based on segment
        if activity_level > 50:  # High activity user
            if diversity > 0.6:  # Diverse taste
                # Active diverse users: balance popularity with novelty
                return self.get_improved_hybrid_recommendations(user_id, n, popularity_weight=0.3)
            else:  # Focused taste
                # Active focused users: emphasize knowledge-based recommendations
                return self.get_knowledge_enhanced_recommendations(user_id, n)
        else:  # Low activity user
            if recency < 30:  # New user
                # New casual users: emphasize popular items
                return self._recommend_popular(user_id, n)
            else:  # Established user
                # Established casual users: standard personalization
                return self.get_personalized_recommendations(user_id, n)
            
    def _recommend_popular(self, user_id, n=10):
        """Get popular movie recommendations."""
        # Get movie popularity counts
        movie_counts = self.ratings_df['movie_id'].value_counts()
        
        # Get movies already rated by the user
        if user_id in self.user_profiles:
            rated_movies = self.user_profiles[user_id]['rated_movies']
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