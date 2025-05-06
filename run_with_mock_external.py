import os
import sys
sys.path.append('.')  # Add project root to path

from src.knowledge_graph import MovieLensKnowledgeGraph
from src.fixed_external_knowledge import MockExternalKnowledgeConnector

def main():
    # Create required directories
    os.makedirs('./data/external_data', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Load MovieLens dataset
    print("Loading MovieLens dataset...")
    kg = MovieLensKnowledgeGraph(data_path='./data/ml-100k')
    kg.load_data()
    
    print(f"Loaded {len(kg.ratings_df)} ratings from {kg.ratings_df['user_id'].nunique()} users on {kg.ratings_df['movie_id'].nunique()} movies")
    
    # Create mock connector
    print("Creating mock external knowledge connector...")
    connector = MockExternalKnowledgeConnector(movies_df=kg.movies_df, api_sleep=0.1)
    
    # Generate mock external links
    print("Generating mock external links...")
    links_df = connector.link_movielens_to_knowledge_graphs(
        output_file='./data/external_data/movie_links.csv',
        source='both'
    )
    
    # Generate mock enriched data
    print("Generating mock enriched data...")
    enriched_df = connector.enrich_movie_data(
        links_df,
        output_file='./data/external_data/movie_enriched.csv',
        source='both'
    )
    
    print("\nExternal knowledge integration complete!")
    print("The following files were created:")
    print("- ./data/external_data/movie_links.csv")
    print("- ./data/external_data/movie_enriched.csv")
    
    print("\nNow you can run the following commands to complete the pipeline:")
    print("1. python scripts/build_enhanced_kg.py")
    print("2. python scripts/compare_forgetting.py")

if __name__ == "__main__":
    main()
