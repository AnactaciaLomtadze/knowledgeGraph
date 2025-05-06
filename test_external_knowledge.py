import os
import sys
import pandas as pd
sys.path.append('.')  # Add project root to path

from src.knowledge_graph import MovieLensKnowledgeGraph
from src.external_knowledge import ExternalKnowledgeConnector

def main():
    # Create output directory
    os.makedirs('./data/external_data', exist_ok=True)
    
    # Load MovieLens dataset
    print("Loading MovieLens dataset...")
    kg = MovieLensKnowledgeGraph(data_path='./data/ml-100k')
    kg.load_data()
    
    # Get just 5 movies to test
    test_movies = kg.movies_df.head(5)
    
    # Create connector with longer sleep time
    connector = ExternalKnowledgeConnector(movies_df=test_movies, api_sleep=2.0)
    
    # Link movies to external knowledge bases
    links_file = './data/external_data/test_movie_links.csv'
    print("Linking test movies to external knowledge bases...")
    links_df = connector.link_movielens_to_knowledge_graphs(
        output_file=links_file,
        source='both'
    )
    
    if links_df is not None and not links_df.empty:
        # Enrich with detailed information
        enriched_file = './data/external_data/test_movie_enriched.csv'
        print("Enriching test movies with external knowledge...")
        enriched_df = connector.enrich_movie_data(
            links_df,
            output_file=enriched_file,
            source='both'
        )
        
        print("Test complete! Check the files:")
        print(f"- {links_file}")
        print(f"- {enriched_file}")
    else:
        print("Failed to link movies to external knowledge bases")

if __name__ == "__main__":
    main()
