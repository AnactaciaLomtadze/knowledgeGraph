#src/visualization_tool.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

class VisualizationTool:
    """
    Visualization tools for forgetting mechanisms in recommendation systems.
    
    This class provides methods to visualize the impact of forgetting on
    recommendations, memory strength, and graph structure.
    """
    def __init__(self, knowledge_graph, forgetting_mechanism=None):
        """
        Initialize the visualization tool.
        
        Args:
            knowledge_graph: The MovieLensKnowledgeGraph instance
            forgetting_mechanism: Optional ForgettingMechanism instance
        """
        self.kg = knowledge_graph
        self.fm = forgetting_mechanism
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('VisualizationTool')
        
    def visualize_forgetting_impact(self, user_id, time_period=30, visualization_type='memory_strength', filename=None):
        """
        Visualize the impact of forgetting mechanism on a user's memory over time.
        
        Args:
            user_id: The user ID to visualize
            time_period: Number of days to simulate
            visualization_type: Type of visualization ('memory_strength', 'recommendations', 'graph')
            filename: If specified, save figure to this file
        """
        if self.fm is None:
            self.logger.error("Forgetting mechanism not provided. Cannot visualize forgetting impact.")
            return
        
        if visualization_type == 'memory_strength':
            self._visualize_memory_strength_decay(user_id, time_period, filename)
        elif visualization_type == 'recommendations':
            self._visualize_recommendation_changes(user_id, time_period, filename)
        elif visualization_type == 'graph':
            self._visualize_graph_changes(user_id, filename)
        else:
            self.logger.error(f"Unknown visualization type: {visualization_type}")
    
    def _visualize_memory_strength_decay(self, user_id, time_period, filename=None):
        """Visualize how memory strength decays over time for a user's top movies."""
        # Get user's rated movies
        user_ratings = self.kg.ratings_df[self.kg.ratings_df['user_id'] == user_id]
        if user_ratings.empty:
            self.logger.error(f"No ratings found for user {user_id}")
            return
        
        # Get top 5 movies by rating
        top_movies = user_ratings.sort_values('rating', ascending=False).head(5)
        movie_ids = top_movies['movie_id'].values
        movie_titles = []
        
        for mid in movie_ids:
            movie_data = self.kg.movies_df[self.kg.movies_df['movie_id'] == mid]
            if not movie_data.empty:
                title = movie_data['title'].values[0]
                # Truncate long titles
                if len(title) > 20:
                    title = title[:17] + "..."
                movie_titles.append(title)
            else:
                movie_titles.append(f"Movie {mid}")
        
        # Simulate memory decay over time
        days = list(range(time_period + 1))
        memory_strengths = {mid: [] for mid in movie_ids}
        
        # Save original memory strengths
        original_strengths = {(user_id, mid): self.fm.memory_strength.get((user_id, mid), 0.5) 
                             for mid in movie_ids}
        
        # Get forgetting parameters
        params = self.fm.personalize_forgetting_parameters(user_id)
        
        # Simulate memory decay for each day
        for day in tqdm(days, desc="Simulating memory decay"):
            if day > 0:
                # Simulate advancing time
                current_time = datetime.now() - timedelta(days=(time_period - day))
                
                # Update last interaction time for simulation
                for mid in movie_ids:
                    self.fm.last_interaction_time[(user_id, mid)] = current_time.timestamp() - day * 86400
                
                # Apply hybrid decay
                self.fm.create_hybrid_decay_function(
                    user_id, 
                    time_weight=params['time_weight'],
                    usage_weight=params['usage_weight'],
                    novelty_weight=params['novelty_weight']
                )
            
            # Record memory strengths
            for mid in movie_ids:
                memory_strengths[mid].append(self.fm.memory_strength.get((user_id, mid), 0.5))
        
        # Restore original memory strengths
        for (u_id, mid), strength in original_strengths.items():
            self.fm.memory_strength[(u_id, mid)] = strength
        
        # Plot memory strength decay
        plt.figure(figsize=(12, 8))
        for i, mid in enumerate(movie_ids):
            plt.plot(days, memory_strengths[mid], marker='o', linewidth=2, label=movie_titles[i])
        
        plt.title(f"Memory Strength Decay Over Time for User {user_id}")
        plt.xlabel("Days")
        plt.ylabel("Memory Strength")
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
    
    def _visualize_recommendation_changes(self, user_id, time_period, filename=None):
        """Visualize how recommendations change over time with forgetting."""
        # Initialize forgetting-aware recommendation function
        params = self.fm.personalize_forgetting_parameters(user_id)
        forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
            'hybrid', params
        )
        
        # Get initial recommendations without forgetting
        initial_recs = self.kg.get_recommendations(user_id, method='hybrid', n=10)
        
        # Get initial recommendation titles
        initial_rec_titles = []
        for mid in initial_recs:
            movie_data = self.kg.movies_df[self.kg.movies_df['movie_id'] == mid]
            if not movie_data.empty:
                title = movie_data['title'].values[0]
                # Truncate long titles
                if len(title) > 30:
                    title = title[:27] + "..."
                initial_rec_titles.append(title)
            else:
                initial_rec_titles.append(f"Movie {mid}")
        
        # Sample points over time
        sample_days = [0, time_period // 4, time_period // 2, 3 * time_period // 4, time_period]
        
        # Save original memory strengths
        user_movies = [(u_id, m_id) for (u_id, m_id) in self.fm.memory_strength.keys() if u_id == user_id]
        original_strengths = {key: self.fm.memory_strength[key] for key in user_movies}
        original_times = {key: self.fm.last_interaction_time.get(key, 0) for key in user_movies}
        
        # Track recommendation changes
        rec_changes = []
        
        for day in tqdm(sample_days, desc="Simulating recommendation changes"):
            # Simulate advancing time for all user-movie interactions
            current_time = datetime.now().timestamp()
            for u_m_key in user_movies:
                original_time = original_times[u_m_key]
                self.fm.last_interaction_time[u_m_key] = original_time - (time_period - day) * 86400
            
            # Get recommendations with forgetting at this point
            forgetting_recs = forgetting_rec_fn(user_id, n=10)
            
            # Get recommendation titles
            forgetting_rec_titles = []
            for mid in forgetting_recs:
                movie_data = self.kg.movies_df[self.kg.movies_df['movie_id'] == mid]
                if not movie_data.empty:
                    title = movie_data['title'].values[0]
                    # Truncate long titles
                    if len(title) > 30:
                        title = title[:27] + "..."
                    forgetting_rec_titles.append(title)
                else:
                    forgetting_rec_titles.append(f"Movie {mid}")
            
            # Record recommendation changes
            rec_changes.append({
                'day': day,
                'recommendations': forgetting_rec_titles
            })
        
        # Restore original memory strengths and times
        for key in user_movies:
            self.fm.memory_strength[key] = original_strengths[key]
            self.fm.last_interaction_time[key] = original_times[key]
        
        # Plot recommendation changes
        plt.figure(figsize=(14, 10))
        
        # Create a visualization showing how recommendations change
        plt.subplot(1, 2, 1)
        plt.title(f"Initial Recommendations for User {user_id}")
        plt.barh(range(len(initial_rec_titles)), [1]*len(initial_rec_titles), color='skyblue')
        plt.yticks(range(len(initial_rec_titles)), initial_rec_titles)
        plt.xlabel('Recommendation Score')
        
        # Create a visualization showing final recommendations
        plt.subplot(1, 2, 2)
        plt.title(f"Recommendations After {time_period} Days")
        plt.barh(range(len(rec_changes[-1]['recommendations'])), 
                [1]*len(rec_changes[-1]['recommendations']), color='lightgreen')
        plt.yticks(range(len(rec_changes[-1]['recommendations'])), 
                  rec_changes[-1]['recommendations'])
        plt.xlabel('Recommendation Score')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{filename}_comparison.png")
        else:
            plt.show()
        
        # Create a table showing the evolution of recommendations
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.axis('off')
        ax.axis('tight')
        
        # Create table data
        table_data = []
        for i in range(len(rec_changes)):
            row = [f"Day {rec_changes[i]['day']}"] + rec_changes[i]['recommendations'][:5]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, 
                        colLabels=["Time"] + [f"Rec {i+1}" for i in range(5)],
                        loc='center', cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.title(f"Evolution of Recommendations Over Time for User {user_id}")
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{filename}_evolution.png")
        else:
            plt.show()
    
    def _visualize_graph_changes(self, user_id, filename=None):
        """Visualize how the knowledge graph changes with forgetting mechanism."""
        if self.fm is None:
            self.logger.error("Forgetting mechanism not provided. Cannot visualize graph changes.")
            return
        
        # Create a copy of the current graph
        G_before = self.kg.G.copy()
        
        # Apply forgetting to user's interactions
        params = self.fm.personalize_forgetting_parameters(user_id)
        self.fm.create_hybrid_decay_function(
            user_id, 
            time_weight=params['time_weight'],
            usage_weight=params['usage_weight'],
            novelty_weight=params['novelty_weight']
        )
        
        # Create a modified graph with edge weights adjusted by memory strength
        G_after = self.kg.G.copy()
        
        # Adjust edge weights based on memory strength
        user_node = f"user_{user_id}"
        for movie_node in G_after.neighbors(user_node):
            if movie_node.startswith("movie_"):
                movie_id = int(movie_node.split("_")[1])
                memory_strength = self.fm.memory_strength.get((user_id, movie_id), 0.5)
                
                # Update edge weight based on memory strength
                edge_data = G_after.get_edge_data(user_node, movie_node)
                if edge_data and 'weight' in edge_data:
                    G_after[user_node][movie_node]['weight'] = edge_data['weight'] * memory_strength
        
        # Extract subgraph around user
        nodes_before = {user_node}
        nodes_after = {user_node}
        
        # Get 1-hop neighborhood
        for node in self.kg.G.neighbors(user_node):
            nodes_before.add(node)
            nodes_after.add(node)
        
        # Get 2-hop neighborhood (movie-movie connections)
        for movie_node in list(nodes_before):
            if movie_node.startswith("movie_"):
                for neighbor in self.kg.G.neighbors(movie_node):
                    if neighbor.startswith("movie_"):
                        nodes_before.add(neighbor)
                        nodes_after.add(neighbor)
        
        # Create subgraphs
        subgraph_before = G_before.subgraph(nodes_before)
        subgraph_after = G_after.subgraph(nodes_after)
        
        # Visualize both graphs
        plt.figure(figsize=(18, 8))
        
        # Plot original graph
        plt.subplot(1, 2, 1)
        pos_before = nx.spring_layout(subgraph_before, seed=42)
        
        # Draw nodes with different colors
        node_colors = []
        for node in subgraph_before.nodes():
            if node == user_node:
                node_colors.append('red')
            elif node.startswith('user'):
                node_colors.append('skyblue')
            else:
                node_colors.append('lightgreen')
        
        # Draw edges with width proportional to weight
        edge_widths = [subgraph_before[u][v].get('weight', 1.0) * 1.5 for u, v in subgraph_before.edges()]
        
        nx.draw_networkx_nodes(subgraph_before, pos_before, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(subgraph_before, pos_before, width=edge_widths, alpha=0.5)
        nx.draw_networkx_labels(subgraph_before, pos_before, font_size=8)
        plt.title(f"User {user_id} Subgraph Before Forgetting")
        plt.axis('off')
        
        # Plot graph after forgetting
        plt.subplot(1, 2, 2)
        # Use same layout for comparison
        pos_after = pos_before
        
        # Draw nodes with different colors (same as before)
        node_colors_after = []
        for node in subgraph_after.nodes():
            if node == user_node:
                node_colors_after.append('red')
            elif node.startswith('user'):
                node_colors_after.append('skyblue')
            else:
                node_colors_after.append('lightgreen')
        
        # Draw edges with width proportional to weight after forgetting
        edge_widths_after = [subgraph_after[u][v].get('weight', 1.0) * 1.5 for u, v in subgraph_after.edges()]
        
        nx.draw_networkx_nodes(subgraph_after, pos_after, node_color=node_colors_after, alpha=0.8)
        nx.draw_networkx_edges(subgraph_after, pos_after, width=edge_widths_after, alpha=0.5)
        nx.draw_networkx_labels(subgraph_after, pos_after, font_size=8)
        plt.title(f"User {user_id} Subgraph After Forgetting")
        plt.axis('off')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{filename}_subgraphs.png")
        else:
            plt.show()
        
        # Visualize edge weight changes in a heatmap
        shared_edges = []
        weights_before = []
        weights_after = []
        
        # Get edges connected to the user node
        user_movies_before = [(user_node, m) for m in subgraph_before.neighbors(user_node) if m.startswith("movie_")]
        
        for edge in user_movies_before:
            u, v = edge
            movie_id = int(v.split("_")[1])
            
            # Get movie title
            movie_data = self.kg.movies_df[self.kg.movies_df['movie_id'] == movie_id]
            if not movie_data.empty:
                movie_title = movie_data['title'].values[0]
                # Truncate long titles
                if len(movie_title) > 20:
                    movie_title = movie_title[:17] + "..."
            else:
                movie_title = f"Movie {movie_id}"
            
            w_before = subgraph_before[u][v].get('weight', 1.0)
            w_after = subgraph_after[u][v].get('weight', 1.0)
            
            shared_edges.append(movie_title)
            weights_before.append(w_before)
            weights_after.append(w_after)
        
        # Create a comparison bar chart of edge weights
        plt.figure(figsize=(14, 8))
        x = np.arange(len(shared_edges))
        width = 0.35
        
        plt.bar(x - width/2, weights_before, width, label='Before Forgetting', color='skyblue')
        plt.bar(x + width/2, weights_after, width, label='After Forgetting', color='lightgreen')
        
        plt.xlabel('Movies')
        plt.ylabel('Edge Weight')
        plt.title(f'Edge Weight Changes Due to Forgetting for User {user_id}')
        plt.xticks(x, shared_edges, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{filename}_edge_weights.png")
        else:
            plt.show()
    
    def visualize_memory_distribution(self, user_id=None, filename=None):
        """
        Visualize distribution of memory strengths across users or for a specific user.
        
        Args:
            user_id: Optional specific user to visualize
            filename: If specified, save figure to this file
        """
        if self.fm is None:
            self.logger.error("Forgetting mechanism not provided. Cannot visualize memory distribution.")
            return
        
        memory_strengths = []
        user_ids = []
        movie_ratings = []
        
        if user_id is not None:
            # Get data for specific user
            for (u_id, m_id), strength in self.fm.memory_strength.items():
                if u_id == user_id:
                    memory_strengths.append(strength)
                    
                    # Get rating if available
                    rating_data = self.kg.ratings_df[
                        (self.kg.ratings_df['user_id'] == u_id) & 
                        (self.kg.ratings_df['movie_id'] == m_id)
                    ]
                    
                    if not rating_data.empty:
                        rating = rating_data.iloc[0]['rating']
                    else:
                        rating = 0
                        
                    movie_ratings.append(rating)
            
            # Scatter plot of memory strength vs. rating
            plt.figure(figsize=(10, 6))
            plt.scatter(movie_ratings, memory_strengths, alpha=0.6, s=100)
            
            # Add trendline
            if movie_ratings and memory_strengths:
                z = np.polyfit(movie_ratings, memory_strengths, 1)
                p = np.poly1d(z)
                plt.plot([min(movie_ratings), max(movie_ratings)], 
                        [p(min(movie_ratings)), p(max(movie_ratings))], 
                        "r--", linewidth=2)
            
            plt.xlabel('Movie Rating')
            plt.ylabel('Memory Strength')
            plt.title(f'Memory Strength vs. Rating for User {user_id}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if filename:
                plt.savefig(f"{filename}_user_{user_id}.png")
            else:
                plt.show()
        else:
            # Get data for all users
            for (u_id, _), strength in self.fm.memory_strength.items():
                memory_strengths.append(strength)
                user_ids.append(u_id)
            
            # Create a boxplot of memory strengths by user
            plt.figure(figsize=(14, 8))
            df = pd.DataFrame({'user_id': user_ids, 'memory_strength': memory_strengths})
            
            # If too many users, sample a subset
            if df['user_id'].nunique() > 20:
                user_sample = np.random.choice(df['user_id'].unique(), 20, replace=False)
                sample_df = df[df['user_id'].isin(user_sample)]
            else:
                sample_df = df
            
            # Sort users by median memory strength for better visualization
            user_medians = sample_df.groupby('user_id')['memory_strength'].median().sort_values()
            sorted_users = user_medians.index.tolist()
            
            # Create order for the boxplot
            sample_df['user_id'] = pd.Categorical(sample_df['user_id'], categories=sorted_users)
            
            # Create boxplot
            sns.boxplot(x='user_id', y='memory_strength', data=sample_df.sort_values('user_id'))
            plt.xlabel('User ID')
            plt.ylabel('Memory Strength')
            plt.title('Distribution of Memory Strengths Across Users')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if filename:
                plt.savefig(f"{filename}_all_users.png")
            else:
                plt.show()
            
            # Also create a histogram of all memory strengths
            plt.figure(figsize=(10, 6))
            sns.histplot(memory_strengths, bins=20, kde=True)
            plt.xlabel('Memory Strength')
            plt.ylabel('Count')
            plt.title('Distribution of All Memory Strengths')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if filename:
                plt.savefig(f"{filename}_histogram.png")
            else:
                plt.show()
    
    def compare_forgetting_strategies(self, user_id, n=10, filename=None):
        """
        Compare different forgetting strategies for a user.
        
        Args:
            user_id: The user ID to analyze
            n: Number of recommendations to consider
            filename: If specified, save figure to this file
        """
        if self.fm is None:
            self.logger.error("Forgetting mechanism not provided. Cannot compare strategies.")
            return
        
        # Store original memory strengths
        original_strengths = {}
        for key, value in self.fm.memory_strength.items():
            if key[0] == user_id:
                original_strengths[key] = value
        
        # Get baseline recommendations (no forgetting)
        baseline_recs = self.kg.get_recommendations(user_id, method='hybrid', n=n)
        
        # Get baseline recommendation titles
        baseline_titles = []
        for mid in baseline_recs:
            movie_data = self.kg.movies_df[self.kg.movies_df['movie_id'] == mid]
            if not movie_data.empty:
                title = movie_data['title'].values[0]
                # Truncate long titles
                if len(title) > 30:
                    title = title[:27] + "..."
                baseline_titles.append(title)
            else:
                baseline_titles.append(f"Movie {mid}")
        
        # Define strategies to compare
        strategies = {
            'Time-based': {
                'decay_parameter': 0.1,
                'time_weight': 0.7,
                'usage_weight': 0.2,
                'novelty_weight': 0.1,
                'forgetting_factor': 0.5
            },
            'Usage-based': {
                'interaction_threshold': 3,
                'time_weight': 0.2,
                'usage_weight': 0.7,
                'novelty_weight': 0.1,
                'forgetting_factor': 0.5
            },
            'Ebbinghaus': {
                'retention': 0.9,
                'strength': 1.0,
                'time_weight': 0.5,
                'usage_weight': 0.3,
                'novelty_weight': 0.2,
                'forgetting_factor': 0.5
            },
            'Power-law': {
                'decay_factor': 0.75,
                'time_weight': 0.5,
                'usage_weight': 0.3,
                'novelty_weight': 0.2,
                'forgetting_factor': 0.5
            },
            'Step Function': {
                'time_weight': 0.5,
                'usage_weight': 0.3,
                'novelty_weight': 0.2,
                'forgetting_factor': 0.5
            },
            'Personalized': self.fm.personalize_forgetting_parameters(user_id)
        }
        
        all_recommendations = {}
        
        # Apply each strategy and get recommendations
        for strategy_name, params in tqdm(strategies.items(), desc="Testing strategies"):
            # Apply strategy
            if strategy_name == 'Time-based':
                self.fm.implement_time_based_decay(user_id, params['decay_parameter'])
            elif strategy_name == 'Usage-based':
                self.fm.implement_usage_based_decay(user_id, params['interaction_threshold'])
            elif strategy_name == 'Ebbinghaus':
                self.fm.implement_ebbinghaus_forgetting_curve(user_id, params['retention'], params['strength'])
            elif strategy_name == 'Power-law':
                self.fm.implement_power_law_decay(user_id, params['decay_factor'])
            elif strategy_name == 'Step Function':
                self.fm.implement_step_function_decay(user_id)
            elif strategy_name == 'Personalized':
                self.fm.create_hybrid_decay_function(
                    user_id,
                    time_weight=params['time_weight'],
                    usage_weight=params['usage_weight'],
                    novelty_weight=params['novelty_weight']
                )
            
            # Get recommendations after applying the strategy
            forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
                'hybrid', params
            )
            recs = forgetting_rec_fn(user_id, n=n)
            
            # Get recommendation titles
            rec_titles = []
            for mid in recs:
                movie_data = self.kg.movies_df[self.kg.movies_df['movie_id'] == mid]
                if not movie_data.empty:
                    title = movie_data['title'].values[0]
                    # Truncate long titles
                    if len(title) > 30:
                        title = title[:27] + "..."
                    rec_titles.append(title)
                else:
                    rec_titles.append(f"Movie {mid}")
            
            all_recommendations[strategy_name] = rec_titles
            
            # Reset memory strengths to original values
            for key, value in original_strengths.items():
                self.fm.memory_strength[key] = value
        
        # Create a visualization to compare strategies
        plt.figure(figsize=(15, 10))
        
        # Create a set of all unique recommendations
        all_unique_recs = set()
        for recs in all_recommendations.values():
            all_unique_recs.update(recs)
        all_unique_recs.update(baseline_titles)
        all_unique_recs = list(all_unique_recs)
        
        # Create a matrix of recommendation ranks
        strategy_names = list(strategies.keys()) + ['Baseline']
        all_recommendations['Baseline'] = baseline_titles
        
        # Create a heatmap
        heatmap_data = np.zeros((len(all_unique_recs), len(strategy_names)))
        
        for i, strategy in enumerate(strategy_names):
            recs = all_recommendations[strategy]
            for j, title in enumerate(all_unique_recs):
                if title in recs:
                    rank = recs.index(title) + 1
                    heatmap_data[j, i] = n - rank + 1  # Higher score for higher rank
                else:
                    heatmap_data[j, i] = 0
        
        # Sort rows by total score
        row_sums = np.sum(heatmap_data, axis=1)
        sorted_indices = np.argsort(-row_sums)
        heatmap_data = heatmap_data[sorted_indices]
        all_unique_recs = [all_unique_recs[i] for i in sorted_indices]
        
        # Create heatmap
        plt.figure(figsize=(12, len(all_unique_recs) * 0.4))
        ax = sns.heatmap(heatmap_data, cmap='YlGnBu', 
                         xticklabels=strategy_names, 
                         yticklabels=all_unique_recs,
                         linewidths=.5)
        
        plt.title(f'Recommendation Comparison Across Forgetting Strategies for User {user_id}')
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{filename}_heatmap.png")
        else:
            plt.show()
            
        # Calculate and display recommendation overlap measures
        plt.figure(figsize=(12, 8))
        
        jaccard_similarities = []
        labels = []
        
        baseline_set = set(baseline_titles)
        
        for strategy in strategy_names:
            if strategy != 'Baseline':
                strategy_set = set(all_recommendations[strategy])
                jaccard = len(baseline_set.intersection(strategy_set)) / len(baseline_set.union(strategy_set))
                jaccard_similarities.append(jaccard)
                labels.append(strategy)
        
        # Sort by similarity for better visualization
        sorted_indices = np.argsort(jaccard_similarities)
        sorted_jaccard = [jaccard_similarities[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        
        plt.bar(range(len(sorted_jaccard)), sorted_jaccard, color='skyblue')
        plt.xlabel('Forgetting Strategy')
        plt.ylabel('Jaccard Similarity with Baseline')
        plt.title('Recommendation Overlap with Baseline (No Forgetting)')
        plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=45, ha='right')
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"{filename}_similarity.png")
        else:
            plt.show()
        
        return all_recommendations
    
    def visualize_temporal_dimension(self, user_id, metrics=['hit_rate', 'diversity', 'novelty'], 
                                     time_points=5, max_days=180, filename=None):
        """
        Visualize how different metrics evolve over time with forgetting.
        
        Args:
            user_id: The user ID to analyze
            metrics: List of metrics to track
            time_points: Number of time points to sample
            max_days: Maximum number of days to simulate
            filename: If specified, save figure to this file
        """
        if self.fm is None:
            self.logger.error("Forgetting mechanism not provided. Cannot visualize temporal dimension.")
            return
        
        # Ensure user exists
        if user_id not in self.kg.user_profiles:
            self.logger.error(f"User {user_id} not found in user profiles.")
            return
            
        # Create a simple test set from the most recent half of user's rated movies
        user_ratings = self.kg.ratings_df[self.kg.ratings_df['user_id'] == user_id]
        if len(user_ratings) < 2:
            self.logger.error(f"User {user_id} has too few ratings for temporal analysis.")
            return
            
        # Sort by timestamp
        user_ratings = user_ratings.sort_values('timestamp')
        num_ratings = len(user_ratings)
        
        # Use the first 70% as training, the last 30% as testing
        split_point = int(num_ratings * 0.7)
        training_ratings = user_ratings.iloc[:split_point]
        test_ratings = user_ratings.iloc[split_point:]
        
        test_set = set(test_ratings['movie_id'])
        
        # Save original memory strengths for this user
        original_strengths = {}
        original_times = {}
        for key, value in self.fm.memory_strength.items():
            if key[0] == user_id:
                original_strengths[key] = value
                if key in self.fm.last_interaction_time:
                    original_times[key] = self.fm.last_interaction_time[key]
        
        # Create time points for analysis
        days = np.linspace(0, max_days, time_points)
        
        # Initialize metrics tracking
        metrics_over_time = {metric: [] for metric in metrics}
        
        # Get personalized parameters
        params = self.fm.personalize_forgetting_parameters(user_id)
        
        # Simulate each time point
        for day in tqdm(days, desc="Simulating time points"):
            # Reset memory strengths
            for key, value in original_strengths.items():
                self.fm.memory_strength[key] = value
            
            # Set interaction times to simulate time passage
            current_time = datetime.now().timestamp()
            for key in original_strengths:
                # Set time to be (current - day*seconds_in_day)
                self.fm.last_interaction_time[key] = current_time - (day * 24 * 60 * 60)
            
            # Apply forgetting
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
            recommendations = forgetting_rec_fn(user_id, n=10)
            
            # Calculate metrics
            # Implement metrics calculations here
            from sklearn.metrics.pairwise import cosine_similarity
            
            for metric in metrics:
                if metric == 'hit_rate':
                    # Calculate hit rate
                    hit = 0
                    for rec in recommendations[:10]:
                        if rec in test_set:
                            hit = 1
                            break
                    metrics_over_time[metric].append(hit)
                
                elif metric == 'diversity':
                    # Calculate diversity
                    genre_vectors = []
                    for movie_id in recommendations:
                        if movie_id in self.kg.movie_features:
                            genre_vectors.append(self.kg.movie_features[movie_id])
                    
                    if len(genre_vectors) >= 2:  # Need at least 2 movies to calculate diversity
                        sim_matrix = cosine_similarity(genre_vectors)
                        np.fill_diagonal(sim_matrix, 0)
                        diversity = 1 - np.mean(sim_matrix)
                    else:
                        diversity = 0
                    
                    metrics_over_time[metric].append(diversity)
                
                elif metric == 'novelty':
                    # Calculate novelty based on popularity
                    novelty_scores = []
                    for movie_id in recommendations:
                        popularity = len(self.kg.ratings_df[self.kg.ratings_df['movie_id'] == movie_id])
                        # Normalize by max popularity
                        max_popularity = self.kg.ratings_df['movie_id'].value_counts().max()
                        normalized_pop = popularity / max_popularity
                        novelty = 1 - normalized_pop  # Less popular = more novel
                        novelty_scores.append(novelty)
                    
                    avg_novelty = np.mean(novelty_scores) if novelty_scores else 0
                    metrics_over_time[metric].append(avg_novelty)
                
                elif metric == 'serendipity':
                    # Calculate serendipity (novelty + relevance)
                    serendipity_scores = []
                    
                    # Get user preferences
                    if user_id in self.kg.user_profiles:
                        user_prefs = self.kg.user_profiles[user_id]['genre_preferences']
                    else:
                        user_prefs = np.ones(19) / 19  # Uniform if not available
                    
                    for movie_id in recommendations:
                        # Calculate unexpectedness (novelty)
                        popularity = len(self.kg.ratings_df[self.kg.ratings_df['movie_id'] == movie_id])
                        max_popularity = self.kg.ratings_df['movie_id'].value_counts().max()
                        unexpectedness = 1 - (popularity / max_popularity)
                        
                        # Calculate relevance
                        if movie_id in self.kg.movie_features:
                            movie_genres = self.kg.movie_features[movie_id]
                            # Ensure user_prefs is non-negative for relevance calculation
                            norm_prefs = user_prefs - np.min(user_prefs)
                            if np.max(norm_prefs) > 0:
                                norm_prefs = norm_prefs / np.max(norm_prefs)
                            
                            relevance = np.dot(norm_prefs, movie_genres) / (np.sum(movie_genres) + 1e-10)
                        else:
                            relevance = 0
                        
                        # Serendipity combines both
                        serendipity = unexpectedness * relevance
                        serendipity_scores.append(serendipity)
                    
                    avg_serendipity = np.mean(serendipity_scores) if serendipity_scores else 0
                    metrics_over_time[metric].append(avg_serendipity)
                    
                else:
                    # Unknown metric
                    self.logger.warning(f"Unknown metric: {metric}")
                    metrics_over_time[metric].append(0)
        
        # Restore original memory strengths
        for key, value in original_strengths.items():
            self.fm.memory_strength[key] = value
        
        # Restore original interaction times
        for key, value in original_times.items():
            self.fm.last_interaction_time[key] = value
        
        # Visualize results
        plt.figure(figsize=(12, 8))
        
        for metric, values in metrics_over_time.items():
            plt.plot(days, values, 'o-', linewidth=2, label=metric)
        
        plt.xlabel('Days')
        plt.ylabel('Metric Value')
        plt.title(f'Evolution of Recommendation Metrics Over Time for User {user_id}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        
        return metrics_over_time
    def visualize_comparative_evaluation(self, results_df, metrics=['hit_rate@10', 'precision@10', 'novelty', 'serendipity'],
                                         filename=None):
        """
        Visualize comparative evaluation results.

        Args:
            results_df: DataFrame with evaluation results
            metrics: List of metrics to visualize
            filename: If specified, save figure to this file
        """
        if 'recommender' not in results_df.columns or 'category' not in results_df.columns:
            self.logger.error("Results DataFrame must contain 'recommender' and 'category' columns.")
            return

        # Check which metrics are available in the DataFrame
        available_metrics = [col for col in results_df.columns if col in metrics]

        if not available_metrics:
            self.logger.error(f"None of the requested metrics {metrics} found in DataFrame.")
            return

        # Set up the figure
        n_metrics = len(available_metrics)
        fig_height = 6 * n_metrics
        fig, axs = plt.subplots(n_metrics, 1, figsize=(12, fig_height))

        # If only one metric, axs isn't an array
        if n_metrics == 1:
            axs = [axs]

        # Calculate and plot results for each metric
        for i, metric in enumerate(available_metrics):
            # Group by recommender and calculate mean and std
            grouped = results_df.groupby('recommender')[metric].agg(['mean', 'std']).reset_index()

            # Sort by mean value for better visualization
            grouped = grouped.sort_values('mean', ascending=False)

            # Add category information
            grouped['category'] = grouped['recommender'].map(
                results_df.set_index('recommender')['category'].to_dict())

            # Create colors based on category
            colors = ['skyblue' if cat == 'Traditional' else 'lightgreen' for cat in grouped['category']]

            # Create bar plot
            bars = axs[i].bar(range(len(grouped)), grouped['mean'], yerr=grouped['std'], 
                             color=colors, capsize=10)

            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axs[i].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{height:.3f}', ha='center', va='bottom')

            # Add category indicators
            for j, (cat, rec) in enumerate(zip(grouped['category'], grouped['recommender'])):
                color = 'blue' if cat == 'Traditional' else 'green'
                axs[i].text(j, -0.05, cat[0], color=color, ha='center', fontweight='bold')

            axs[i].set_title(f'Comparison of {metric.capitalize()}')
            axs[i].set_xlabel('Recommender')
            axs[i].set_ylabel(metric.capitalize())
            axs[i].set_xticks(range(len(grouped)))
            axs[i].set_xticklabels(grouped['recommender'], rotation=45, ha='right')
            axs[i].set_ylim(0, grouped['mean'].max() * 1.15)
            axs[i].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

        # Also create a radar chart comparing the top performing methods
        self.create_radar_chart(results_df, metrics=available_metrics, 
                               filename=f"{filename}_radar.png" if filename else None)
    
    def create_radar_chart(self, results_df, metrics, top_n=4, filename=None):
        """
        Create a radar chart comparing top performing methods.

        Args:
            results_df: DataFrame with evaluation results
            metrics: List of metrics to include
            top_n: Number of top methods to include
            filename: If specified, save figure to this file
        """
        # Filter to available metrics
        available_metrics = [m for m in metrics if m in results_df.columns]

        if not available_metrics:
            self.logger.error("No metrics available for radar chart.")
            return

        # Calculate mean performance for each recommender
        mean_df = results_df.groupby('recommender')[available_metrics].mean().reset_index()

        # Select top N recommenders based on average rank across all metrics
        for metric in available_metrics:
            mean_df[f'{metric}_rank'] = mean_df[metric].rank(ascending=False)

        rank_cols = [f'{metric}_rank' for metric in available_metrics]
        mean_df['avg_rank'] = mean_df[rank_cols].mean(axis=1)
        top_recommenders = mean_df.sort_values('avg_rank').head(top_n)['recommender'].tolist()

        # Select a row from each top recommender
        selector = results_df['recommender'].isin(top_recommenders)
        filtered_df = results_df[selector].drop_duplicates('recommender')

        # Prepare data for radar chart
        categories = available_metrics
        N = len(categories)

        # Create angles for radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Add labels
        plt.xticks(angles[:-1], categories, size=12)

        # Draw lines for each recommender
        for i, recommender in enumerate(top_recommenders):
            values = filtered_df[filtered_df['recommender'] == recommender][available_metrics].values.flatten().tolist()
            values += values[:1]  # Close the loop

            # Select color based on category
            category = filtered_df[filtered_df['recommender'] == recommender]['category'].values[0]
            color = 'blue' if category == 'Traditional' else 'green'

            # Plot recommender line
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=recommender, color=color, alpha=0.8)
            ax.fill(angles, values, color=color, alpha=0.1)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.title('Comparison of Top Recommenders Across Metrics', size=15)

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
            
    def visualize_privacy_impact(self, privacy_results, filename=None):
        """
        Visualize the impact of privacy-enhancing forgetting on recommendation quality.
        
        Args:
            privacy_results: List of dictionaries with privacy impact results
            filename: If specified, save figure to this file
        """
        if not privacy_results:
            self.logger.error("No privacy results provided.")
            return
        
        # Extract data from results
        user_ids = [r['user_id'] for r in privacy_results]
        forgotten_items = [r['forgotten_items'] for r in privacy_results]
        
        # Extract metric values
        jaccard_similarities = [r['impact_metrics']['jaccard_similarity'] for r in privacy_results]
        diversity_before = [r['impact_metrics']['genre_diversity_before'] for r in privacy_results]
        diversity_after = [r['impact_metrics']['genre_diversity_after'] for r in privacy_results]
        
        # Create figure with 3 subplots
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Relationship between forgotten items and recommendation similarity
        plt.subplot(2, 2, 1)
        plt.scatter(forgotten_items, jaccard_similarities, alpha=0.7, s=100)
        
        # Add trendline
        z = np.polyfit(forgotten_items, jaccard_similarities, 1)
        p = np.poly1d(z)
        plt.plot(np.array([min(forgotten_items), max(forgotten_items)]), 
                p(np.array([min(forgotten_items), max(forgotten_items)])), 
                "r--", linewidth=2)
        
        plt.xlabel('Number of Forgotten Items')
        plt.ylabel('Jaccard Similarity')
        plt.title('Impact of Forgetting on Recommendation Similarity')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Diversity before vs after
        plt.subplot(2, 2, 2)
        plt.scatter(diversity_before, diversity_after, alpha=0.7, s=100)
        
        # Add diagonal line (no change)
        min_val = min(min(diversity_before), min(diversity_after))
        max_val = max(max(diversity_before), max(diversity_after))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.xlabel('Diversity Before Forgetting')
        plt.ylabel('Diversity After Forgetting')
        plt.title('Change in Recommendation Diversity')
        plt.grid(True, alpha=0.3)
        
        # Highlight points above diagonal (increased diversity)
        increased = [i for i in range(len(diversity_before)) if diversity_after[i] > diversity_before[i]]
        if increased:
            plt.scatter([diversity_before[i] for i in increased], 
                       [diversity_after[i] for i in increased],
                       s=150, facecolors='none', edgecolors='green', linewidth=2)
        
        # Plot 3: Distribution of similarity values
        plt.subplot(2, 2, 3)
        sns.histplot(jaccard_similarities, bins=10, kde=True)
        plt.xlabel('Jaccard Similarity')
        plt.ylabel('Count')
        plt.title('Distribution of Recommendation Similarity After Forgetting')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Bar chart showing average metrics
        plt.subplot(2, 2, 4)
        metrics = ['Jaccard Similarity', 'Diversity Before', 'Diversity After']
        values = [np.mean(jaccard_similarities), np.mean(diversity_before), np.mean(diversity_after)]
        
        plt.bar(metrics, values, color=['blue', 'green', 'orange'])
        plt.ylabel('Average Value')
        plt.title('Average Impact Metrics')
        plt.ylim(0, max(values) * 1.2)
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
            
        # Print summary statistics
        print(f"Average metrics across {len(privacy_results)} users:")
        print(f"  - Items forgotten per user: {np.mean(forgotten_items):.2f}")
        print(f"  - Jaccard similarity: {np.mean(jaccard_similarities):.2f}")
        print(f"  - Change in diversity: {np.mean(np.array(diversity_after) - np.array(diversity_before)):.4f}")
        print(f"  - Users with increased diversity: {sum(d_a > d_b for d_a, d_b in zip(diversity_after, diversity_before))}/{len(privacy_results)}")


    def visualize_decay_comparison(self, results_df, filename=None):
        """
        Visualize comparison of different memory decay approaches.
        
        Args:
            results_df: DataFrame with decay comparison results
            filename: If specified, save figure to this file
        """
        if results_df.empty:
            self.logger.error("Empty results DataFrame for decay comparison")
            return
        
        # Get decay method names (all columns except user_id)
        decay_methods = [col for col in results_df.columns if col != 'user_id']
        
        # Create figure for comparison
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        data = []
        for method in decay_methods:
            values = results_df[method].values
            data.append(values)
        
        # Create box plot
        box = plt.boxplot(data, patch_artist=True, labels=decay_methods)
        
        # Set colors for boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(decay_methods)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add individual points for each user
        for i, method in enumerate(decay_methods):
            # Scatter plot points with small horizontal jitter
            x = np.random.normal(i+1, 0.05, size=len(results_df))
            plt.scatter(x, results_df[method], alpha=0.5, s=30, c='black')
        
        plt.title('Comparison of Memory Decay Methods')
        plt.ylabel('Average Memory Strength')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()