import os
import sys
sys.path.append('.')  # Add project root to path

# Load the build_enhanced_kg.py file
with open('scripts/build_enhanced_kg.py', 'r') as f:
    code = f.read()

# Replace the NetworkX write_gpickle with a different approach
modified_code = code.replace(
    'nx.write_gpickle(kg.G, graph_file)',
    'import pickle\nwith open(graph_file, "wb") as f:\n    pickle.dump(kg.G, f)'
)

# Save the modified file
with open('scripts/build_enhanced_kg.py', 'w') as f:
    f.write(modified_code)

print("Fixed NetworkX issue in build_enhanced_kg.py")
