# scripts/enhanced_knowledge_graph.py
import os
import networkx as nx
import pandas as pd
from src.knowledge_graph import MovieLensKnowledgeGraph
from src.external_knowledge import ExternalKnowledgeConnector

def build_enhanced_knowledge_graph(data_path='./ml-100k', output_dir='./enhanced_kg'):
    """Build and save an enhanced knowledge graph with external data."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize knowledge graph
    kg = MovieLensKnowledgeGraph(data_path=data_path)
    kg.load_data()
    
    # Initialize external knowledge connector
    connector = ExternalKnowledgeConnector(movies_df=kg.movies_df, api_sleep=1.0)
    
    # Link movies to external knowledge bases
    links_file = os.path.join(output_dir, 'movielens_external_links.csv')
    if not os.path.exists(links_file):
        links_df = connector.link_movielens_to_knowledge_graphs(
            output_file=links_file, 
            source='both'
        )
    else:
        links_df = pd.read_csv(links_file)
        print(f"Loaded existing links for {len(links_df)} movies from {links_file}")
    
    # Enrich with detailed information
    enriched_file = os.path.join(output_dir, 'movielens_enriched.csv')
    if not os.path.exists(enriched_file):
        enriched_df = connector.enrich_movie_data(
            links_df, 
            output_file=enriched_file, 
            source='both'
        )
    else:
        print(f"Using existing enriched data from {enriched_file}")
    
    # Load the enriched data into the knowledge graph
    kg.load_external_knowledge(enriched_file)
    
    # Build the enhanced knowledge graph
    kg.build_knowledge_graph_with_external_data()
    
    # Save the knowledge graph
    graph_file = os.path.join(output_dir, 'enhanced_knowledge_graph.gpickle')
    nx.write_gpickle(kg.G, graph_file)
    print(f"Saved enhanced knowledge graph to {graph_file}")
    
    return kg

if __name__ == "__main__":
    kg = build_enhanced_knowledge_graph()
    
    # Print some statistics
    print("\nKnowledge Graph Statistics:")
    print(f"Number of nodes: {kg.G.number_of_nodes()}")
    print(f"Number of edges: {kg.G.number_of_edges()}")
    
    # Count node types
    node_types = {}
    for node, data in kg.G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNode types:")
    for node_type, count in node_types.items():
        print(f"  - {node_type}: {count}")
    
    # Count edge types
    edge_types = {}
    for _, _, data in kg.G.edges(data=True):
        edge_type = data.get('relation_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print("\nEdge types:")
    for edge_type, count in edge_types.items():
        print(f"  - {edge_type}: {count}")