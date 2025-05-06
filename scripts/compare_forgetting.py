# scripts/compare_forgetting.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('./')  # Add project root to path

from src.knowledge_graph import MovieLensKnowledgeGraph
from src.forgetting_mechanism import ForgettingMechanism
from src.context_forgetting import ContextAwareForgettingMechanism
from fix_evaluation_metrics import FixedEvaluationMetrics as EvaluationMetrics

def main():
    # Create output directory for results
    os.makedirs('./results', exist_ok=True)
    
    # Load the enhanced knowledge graph
    print("Loading enhanced knowledge graph...")
    kg = MovieLensKnowledgeGraph(data_path='./data/ml-100k')
    kg.load_data()
    kg.load_external_knowledge('./data/external_data/movie_enriched.csv')
    kg.build_knowledge_graph_with_external_data()
    
    # Initialize forgetting mechanisms
    print("Initializing forgetting mechanisms...")
    basic_fm = ForgettingMechanism(kg)
    context_fm = ContextAwareForgettingMechanism(kg)
    
    # Initialize evaluation metrics
    evaluator = EvaluationMetrics(kg)
    
    # Create test split
    print("Creating train-test split...")
    test_data = kg.create_temporal_train_test_split(test_days=30)
    
    # Select users for evaluation
    test_users = list(test_data.keys())[:50]  # First 50 users
    print(f"Selected {len(test_users)} users for evaluation")
    
    # Define forgetting strategies to compare
    forgetting_strategies = {
        # Traditional approaches
        'No Forgetting': lambda u: kg.get_hybrid_recommendations(u),
        'Basic Time Decay': lambda u: basic_fm.implement_time_based_decay(u, 0.1),
        'Basic Ebbinghaus': lambda u: basic_fm.implement_ebbinghaus_forgetting_curve(u),
        'Basic Hybrid': lambda u: basic_fm.create_hybrid_decay_function(u),
        
        # Context-aware approaches
        'Context-Aware Decay': lambda u: context_fm.implement_context_aware_decay(u),
        'Context-Aware Ebbinghaus': lambda u: context_fm.implement_context_aware_ebbinghaus(u),
        'Frequency-Significance': lambda u: context_fm.implement_frequency_significance_decay(u),
        'Context-Aware Hybrid': lambda u: context_fm.create_context_aware_hybrid_decay(u)
    }
    
    # Setup recommendation functions
    recommendation_functions = {}
    for strategy_name, forgetting_fn in forgetting_strategies.items():
        if strategy_name == 'No Forgetting':
            recommendation_functions[strategy_name] = lambda u, s=strategy_name: kg.get_hybrid_recommendations(u)
        elif strategy_name.startswith('Basic'):
            recommendation_functions[strategy_name] = lambda u, fn=forgetting_fn, s=strategy_name: (
                fn(u), kg.get_hybrid_recommendations(u))[1]
        else:
            # For context-aware strategies, use the integrated recommendation function
            recommendation_functions[strategy_name] = lambda u, fn=forgetting_fn, s=strategy_name: (
                fn(u),
                context_fm.integrate_context_aware_forgetting(
                    recommendation_algorithm='hybrid'
                )(u)
            )[1]
    
    # Evaluate all strategies
    print("Evaluating forgetting strategies...")
    results = []
    
    for user_id in test_users:
        if user_id not in test_data:
            continue
            
        user_results = {'user_id': user_id}
        
        for strategy_name, rec_fn in recommendation_functions.items():
            print(f"Evaluating {strategy_name} for user {user_id}")
            
            # Get recommendations
            recommendations = rec_fn(user_id)
            
            # Calculate metrics
            hit_rate = evaluator.calculate_hit_rate_at_k(test_data[user_id], recommendations, 10)
            precision = evaluator.calculate_precision_at_k(test_data[user_id], recommendations, 10)
            ndcg = evaluator.calculate_ndcg_at_k(test_data[user_id], recommendations, 10)
            diversity = evaluator.calculate_diversity(recommendations)
            novelty = evaluator.calculate_novelty(recommendations)
            serendipity = evaluator.calculate_serendipity(user_id, recommendations)
            
            # Store results
            user_results[f"{strategy_name}_hit_rate"] = hit_rate
            user_results[f"{strategy_name}_precision"] = precision
            user_results[f"{strategy_name}_ndcg"] = ndcg
            user_results[f"{strategy_name}_diversity"] = diversity
            user_results[f"{strategy_name}_novelty"] = novelty
            user_results[f"{strategy_name}_serendipity"] = serendipity
        
        results.append(user_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('./results/forgetting_comparison.csv', index=False)
    print(f"Results saved to ./results/forgetting_comparison.csv")
    
    # Calculate average metrics
    strategy_metrics = {}
    metrics = ['hit_rate', 'precision', 'ndcg', 'diversity', 'novelty', 'serendipity']
    
    for strategy in forgetting_strategies.keys():
        strategy_metrics[strategy] = {}
        for metric in metrics:
            col = f"{strategy}_{metric}"
            strategy_metrics[strategy][metric] = results_df[col].mean()
    
    # Convert to DataFrame for easier display
    summary_df = pd.DataFrame(strategy_metrics).T
    
    # Display summary
    print("\nAverage Metrics by Strategy:")
    print(summary_df)
    
    # Save summary
    summary_df.to_csv('./results/forgetting_summary.csv')
    print(f"Summary saved to ./results/forgetting_summary.csv")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Accuracy metrics
    plt.subplot(2, 2, 1)
    summary_df[['hit_rate', 'precision', 'ndcg']].plot(kind='bar', ax=plt.gca())
    plt.title('Accuracy Metrics')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Beyond-accuracy metrics
    plt.subplot(2, 2, 2)
    summary_df[['diversity', 'novelty', 'serendipity']].plot(kind='bar', ax=plt.gca())
    plt.title('Beyond-Accuracy Metrics')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/forgetting_comparison.png')
    print(f"Visualization saved to ./results/forgetting_comparison.png")
    
    print("Comparison complete!")

if __name__ == "__main__":
    main()