# scripts/build_enhanced_kg.py
import os
import sys
import networkx as nx
import pickle
sys.path.append('./')  # Add project root to path

from src.knowledge_graph import MovieLensKnowledgeGraph

def main():
    # Load the knowledge graph
    print("Loading MovieLens dataset...")
    kg = MovieLensKnowledgeGraph(data_path='./data/ml-100k')
    kg.load_data()
    
    # Load external knowledge
    print("Loading external knowledge...")
    kg.load_external_knowledge('./data/external_data/movie_enriched.csv')
    
    # Build enhanced knowledge graph
    print("Building enhanced knowledge graph...")
    kg.build_knowledge_graph_with_external_data()
    
    # Save the graph
    graph_file = './data/external_data/enhanced_knowledge_graph.gpickle'
    print(f"Saving enhanced knowledge graph to {graph_file}")
    
    with open(graph_file, "wb") as f:
        pickle.dump(kg.G, f)
    
    # Print statistics
    print("\nKnowledge Graph Statistics:")
    print(f"Number of nodes: {kg.G.number_of_nodes()}")
    print(f"Number of edges: {kg.G.number_of_edges()}")
    
    # Count node types
    node_types = {}
    for node, data in kg.G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNode types:")
    for node_type, count in sorted(node_types.items()):
        print(f"  - {node_type}: {count}")
    
    # Count edge types
    edge_types = {}
    for _, _, data in kg.G.edges(data=True):
        edge_type = data.get('relation_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print("\nEdge types:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"  - {edge_type}: {count}")
    
    print("\nEnhanced knowledge graph construction complete!")

if __name__ == "__main__":
    main()