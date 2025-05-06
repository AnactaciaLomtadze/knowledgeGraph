# scripts/build_external_knowledge.py
import os
import sys
import pandas as pd
sys.path.append('./')  # Add project root to path

from src.knowledge_graph import MovieLensKnowledgeGraph
from src.external_knowledge import ExternalKnowledgeConnector

def main():
    # Create output directory
    os.makedirs('./data/external_data', exist_ok=True)
    
    # Load MovieLens dataset
    print("Loading MovieLens dataset...")
    kg = MovieLensKnowledgeGraph(data_path='./data/ml-100k')
    kg.load_data()
    
    # Create connector
    print("Initializing external knowledge connector...")
    connector = ExternalKnowledgeConnector(movies_df=kg.movies_df, api_sleep=1.0)
    
    # Link movies to external knowledge bases
    links_file = './data/external_data/movie_links.csv'
    if not os.path.exists(links_file):
        print("Linking movies to external knowledge bases...")
        links_df = connector.link_movielens_to_knowledge_graphs(
            output_file=links_file, 
            source='both'
        )
    else:
        print(f"Loading existing links from {links_file}")
        links_df = pd.read_csv(links_file)
    
    # Enrich with detailed information
    enriched_file = './data/external_data/movie_enriched.csv'
    if not os.path.exists(enriched_file):
        print("Enriching movies with external knowledge...")
        enriched_df = connector.enrich_movie_data(
            links_df, 
            output_file=enriched_file, 
            source='both'
        )
    else:
        print(f"Using existing enriched data from {enriched_file}")
    
    print("External knowledge integration complete!")

if __name__ == "__main__":
    main()