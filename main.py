import os
import argparse
import logging
import time
import json
import sys
from src.experiment_runner import ForgettingConfig, ForgettingExperimentRunner

def setup_logger():
    """Set up logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("forgetting_recommender.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run forgetting mechanism experiments')
    
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    
    parser.add_argument('--experiments', type=str, nargs='+',
                        default=['baseline_comparison', 'improved_baseline_comparison',
                                'temporal_evaluation', 'parameter_sensitivity', 
                                'privacy_impact', 'scalability_test', 'user_segmentation'],
                        help='Experiments to run')
    
    parser.add_argument('--num_users', type=int, default=50,
                        help='Number of users to evaluate')
    
    parser.add_argument('--temporal_split', action='store_true',
                        help='Use temporal split instead of random')
    
    parser.add_argument('--test_days', type=int, default=30,
                        help='Number of days for testing in temporal split')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Add the additional arguments
    parser.add_argument('--run_all', action='store_true',
                        help='Run all experiments including comprehensive tests')
    
    parser.add_argument('--test_memory_rates', action='store_true',
                        help='Test different memory decay rates')
    
    parser.add_argument('--use_external_knowledge', action='store_true',
                        help='Use external knowledge from DBpedia/Wikidata')
    
    parser.add_argument('--evaluate_user_segments', action='store_true',
                        help='Evaluate performance on different user segments')
    
    return parser.parse_args()

def verify_dataset_path(data_path):
    """Verify that the dataset exists at the specified path."""
    expected_files = ['u.data', 'u.user', 'u.item']
    for file in expected_files:
        file_path = os.path.join(data_path, file)
        if not os.path.exists(file_path):
            return False, f"Dataset file not found: {file_path}"
    return True, "Dataset verified successfully"

def run_experiments(config, experiments):
    """Run the specified experiments."""
    logger = logging.getLogger(__name__)
    
    # Initialize experiment runner
    runner = ForgettingExperimentRunner(config)
    
    # Track results
    results = {}
    
    # Run each experiment
    for experiment in experiments:
        logger.info(f"Starting experiment: {experiment}")
        start_time = time.time()
        
        try:
            experiment_results = runner.run_experiment(experiment)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"Completed experiment: {experiment} in {duration:.2f} seconds")
            
            # Store meta information
            results[experiment] = {
                'completed': True,
                'duration': duration
            }
        except Exception as e:
            logger.error(f"Error in experiment {experiment}: {e}")
            results[experiment] = {
                'completed': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    return results

def generate_report(config, experiment_results):
    """Generate a simple report about the experiments."""
    logger = logging.getLogger(__name__)
    
    report = {
        'configuration': {
            'data_path': config.data_path,
            'output_dir': config.output_dir,
            'num_users': config.num_users,
            'temporal_split': config.temporal_split,
            'test_days': config.test_days,
            'seed': config.seed
        },
        'experiments': experiment_results
    }
    
    # Save report as JSON
    report_path = os.path.join(config.output_dir, 'experiment_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"Report generated and saved to {report_path}")
    
    # Print summary to console
    print("\n===== Experiment Summary =====")
    for experiment, results in experiment_results.items():
        status = 'Completed' if results.get('completed', False) else f"Failed: {results.get('error', 'Unknown error')}"
        print(f"- {experiment}: {status} in {results.get('duration', 0):.2f} seconds")
    print("==============================\n")

def run_decay_rate_comparison(config):
    """Run a focused comparison of different memory decay approaches."""
    logger = logging.getLogger(__name__)
    logger.info("Running memory decay rate comparison...")
    
    # Import pandas here for the DataFrame creation
    import pandas as pd
    
    # Initialize experiment runner
    runner = ForgettingExperimentRunner(config)
    
    try:
        # Run the specific comparison
        results = runner.compare_decay_rates()
        
        # Generate visualization
        comparison_path = os.path.join(config.output_dir, 'decay_rate_comparison.png')
        
        # Make sure results is a DataFrame
        if not isinstance(results, pd.DataFrame):
            logger.error("Expected a DataFrame from compare_decay_rates()")
            results = pd.DataFrame()  # Create empty DataFrame to avoid further errors
        
        runner.visualizer.visualize_decay_comparison(results, filename=comparison_path)
        logger.info(f"Memory decay rate comparison completed and saved to {comparison_path}")
        return results
    except Exception as e:
        logger.error(f"Error in decay rate comparison: {e}")
        return None

def test_recommendation_evolution(config):
    """Test how recommendations evolve over time with different decay methods."""
    logger = logging.getLogger(__name__)
    logger.info("Testing recommendation evolution over time...")
    
    # Initialize experiment runner
    runner = ForgettingExperimentRunner(config)
    
    try:
        # Run the temporal evolution test
        results = runner.test_recommendation_evolution()
        logger.info("Recommendation evolution test completed")
        return results
    except Exception as e:
        logger.error(f"Error in recommendation evolution test: {e}")
        return None

def main():
    """Main function to run the application."""
    # Set up logger
    logger = setup_logger()
    logger.info("Starting Forgetting Mechanism for Recommendation Systems")
    
    # Parse arguments
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Always use the ml-100k dataset
    data_path = 'data/ml-100k'
    
    # Verify dataset exists
    dataset_valid, message = verify_dataset_path(data_path)
    if not dataset_valid:
        logger.error(f"Dataset verification failed: {message}")
        logger.error("Please make sure the MovieLens 100k dataset is downloaded and extracted in the correct location")
        logger.error("You can download it from: https://grouplens.org/datasets/movielens/100k/")
        print(f"\nERROR: {message}")
        print("Please download the MovieLens 100k dataset and extract it to data/ml-100k")
        print("Download URL: https://grouplens.org/datasets/movielens/100k/")
        sys.exit(1)
    
    # Set up configuration
    config = ForgettingConfig(
        data_path=data_path,
        output_dir=args.output_dir,
        num_users=args.num_users,
        temporal_split=args.temporal_split,
        test_days=args.test_days,
        seed=args.seed
    )
    
    logger.info(f"Configuration: {vars(config)}")
    logger.info(f"Using dataset: MovieLens 100k from path: {data_path}")
    
    # Run experiments
    experiment_results = run_experiments(config, args.experiments)
    
    # Generate report
    generate_report(config, experiment_results)
    
    # If running all tests or testing memory rates, add specific detailed evaluations
    if hasattr(args, 'test_memory_rates') and args.test_memory_rates:
        decay_results = run_decay_rate_comparison(config)
        if decay_results is not None:
            logger.info("Memory decay rate comparison completed")
    
    # If running all tests
    if hasattr(args, 'run_all') and args.run_all:
        evolution_results = test_recommendation_evolution(config)
        if evolution_results is not None:
            logger.info("Recommendation evolution test completed")
    
    logger.info("All experiments completed successfully")
    
    # Display final message with key findings
    completed_experiments = sum(1 for _, res in experiment_results.items() if res.get('completed', False))
    print(f"\n===== Final Summary =====")
    print(f"Dataset: MovieLens 100k")
    print(f"Completed: {completed_experiments}/{len(args.experiments)} experiments")
    print(f"Results saved to: {args.output_dir}")
    print("=========================\n")

if __name__ == "__main__":
    main()