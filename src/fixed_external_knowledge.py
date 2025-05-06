#src/fixed_external_knowledge.py
import sys
sys.path.append('.')  # Add project root to path

from src.external_knowledge import ExternalKnowledgeConnector

class MockExternalKnowledgeConnector(ExternalKnowledgeConnector):
    """
    A mock implementation of the ExternalKnowledgeConnector that creates synthetic data
    instead of querying external APIs which are currently failing.
    """
    
    def search_movie_in_dbpedia(self, movie_title, movie_year=None):
        """
        Mock searching for a movie in DBpedia by creating synthetic data.
        """
        # Extract clean title
        clean_title = movie_title
        if '(' in clean_title and ')' in clean_title:
            clean_title = clean_title.split('(')[0].strip()
        
        # Create a synthetic DBpedia URI
        movie_uri = f"http://dbpedia.org/resource/{clean_title.replace(' ', '_')}"
        
        # Create synthetic entity data
        entity_data = {
            'uri': movie_uri,
            'title': movie_title,
            'abstract': f"This is a mock abstract for the movie {clean_title}.",
            'year': movie_year
        }
        
        return entity_data
    
    def search_movie_in_wikidata(self, movie_title, movie_year=None):
        """
        Mock searching for a movie in Wikidata by creating synthetic data.
        """
        # Extract clean title
        clean_title = movie_title
        if '(' in clean_title and ')' in clean_title:
            clean_title = clean_title.split('(')[0].strip()
        
        # Create a synthetic Wikidata ID
        # Use a simple hash function to generate a number
        wikidata_id = abs(hash(clean_title)) % 10000000
        
        # Create synthetic entity data
        entity_data = {
            'uri': f"http://www.wikidata.org/entity/Q{wikidata_id}",
            'id': f"Q{wikidata_id}",
            'title': movie_title,
            'year': movie_year
        }
        
        return entity_data
    
    def get_movie_details_from_dbpedia(self, movie_uri):
        """
        Mock getting movie details from DBpedia.
        """
        # Extract movie name from URI
        movie_name = movie_uri.split('/')[-1].replace('_', ' ')
        
        # Generate synthetic directors
        directors = [
            {'uri': f"http://dbpedia.org/resource/Director1_{movie_name}", 'name': f"Director 1 of {movie_name}"},
            {'uri': f"http://dbpedia.org/resource/Director2_{movie_name}", 'name': f"Director 2 of {movie_name}"}
        ]
        
        # Generate synthetic actors
        actors = [
            {'uri': f"http://dbpedia.org/resource/Actor1_{movie_name}", 'name': f"Actor 1 in {movie_name}"},
            {'uri': f"http://dbpedia.org/resource/Actor2_{movie_name}", 'name': f"Actor 2 in {movie_name}"},
            {'uri': f"http://dbpedia.org/resource/Actor3_{movie_name}", 'name': f"Actor 3 in {movie_name}"}
        ]
        
        # Generate synthetic genres
        genres = [
            f"http://dbpedia.org/resource/Genre1_{movie_name}",
            f"http://dbpedia.org/resource/Genre2_{movie_name}"
        ]
        
        # Create mock movie details
        details = {
            'directors': directors,
            'actors': actors,
            'genres': genres,
            'runtime': 120,  # Mock runtime in minutes
            'country': "United States"
        }
        
        return details
    
    def get_movie_details_from_wikidata(self, entity_id):
        """
        Mock getting movie details from Wikidata.
        """
        # Generate mock details
        directors = [
            {'uri': f"http://www.wikidata.org/entity/Q{int(entity_id[1:]) + 1}", 'name': f"Director A of {entity_id}"},
            {'uri': f"http://www.wikidata.org/entity/Q{int(entity_id[1:]) + 2}", 'name': f"Director B of {entity_id}"}
        ]
        
        actors = [
            {'uri': f"http://www.wikidata.org/entity/Q{int(entity_id[1:]) + 3}", 'name': f"Actor A in {entity_id}"},
            {'uri': f"http://www.wikidata.org/entity/Q{int(entity_id[1:]) + 4}", 'name': f"Actor B in {entity_id}"},
            {'uri': f"http://www.wikidata.org/entity/Q{int(entity_id[1:]) + 5}", 'name': f"Actor C in {entity_id}"}
        ]
        
        genres = [
            {'uri': f"http://www.wikidata.org/entity/Q{int(entity_id[1:]) + 6}", 'name': "Action"},
            {'uri': f"http://www.wikidata.org/entity/Q{int(entity_id[1:]) + 7}", 'name': "Drama"}
        ]
        
        awards = [
            {'uri': f"http://www.wikidata.org/entity/Q{int(entity_id[1:]) + 8}", 'name': "Best Picture"},
            {'uri': f"http://www.wikidata.org/entity/Q{int(entity_id[1:]) + 9}", 'name': "Best Director"}
        ]
        
        # Create mock movie details
        details = {
            'directors': directors,
            'actors': actors,
            'genres': genres,
            'awards': awards,
            'box_office': 100000000,  # Mock box office in dollars
            'runtime': 125,  # Mock runtime in minutes
            'budget': 50000000  # Mock budget in dollars
        }
        
        return details
