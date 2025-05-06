import os
import sys
sys.path.append('.')  # Add project root to path

# Load the external_knowledge.py file
with open('src/external_knowledge.py', 'r') as f:
    code = f.read()

# Fix the SPARQL query syntax for DBpedia
code = code.replace(
    '        }\n         FILTER(?year = {movie_year})',
    '            FILTER(?year = {movie_year})\n        }'
)

# Fix the SPARQL query syntax for Wikidata
code = code.replace(
    '        }\n         FILTER(?year = {movie_year})',
    '            FILTER(?year = {movie_year})\n        }'
)

# Save the fixed file
with open('src/external_knowledge.py', 'w') as f:
    f.write(code)

print("Fixed SPARQL queries in external_knowledge.py")

