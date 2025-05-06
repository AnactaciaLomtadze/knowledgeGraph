#scripts/run_experiements.py
import os
import sys
import argparse
sys.path.append('./')  # Add project root to path

def main():
    parser = argparse.ArgumentParser(description='Run experiments for enhanced recommendation system')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--external', action='store_true', help='Build external knowledge')
    parser.add_argument('--graph', action='store_true', help='Build enhanced knowledge graph')
    parser.add_argument('--compare', action='store_true', help='Compare forgetting strategies')
    
    args = parser.parse_args()
    
    # If no specific step is selected, run all
    if not (args.external or args.graph or args.compare):
        args.all = True
    
    if args.all or args.external:
        print("\n=============================================")
        print("Step 1: Building External Knowledge Integration")
        print("=============================================\n")
        os.system('python scripts/build_external_knowledge.py')
    
    if args.all or args.graph:
        print("\n=============================================")
        print("Step 2: Building Enhanced Knowledge Graph")
        print("=============================================\n")
        os.system('python scripts/build_enhanced_kg.py')
    
    if args.all or args.compare:
        print("\n=============================================")
        print("Step 3: Comparing Forgetting Strategies")
        print("=============================================\n")
        os.system('python scripts/compare_forgetting.py')
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()