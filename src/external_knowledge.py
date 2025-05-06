#src/external_knowledge.py
import json
import pandas as pd
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import logging

class ExternalKnowledgeConnector:
    """
    Class to connect MovieLens dataset with external knowledge sources 
    like DBpedia and Wikidata.
    """
    def __init__(self, movies_df=None, api_sleep=1.0):
        """
        Initialize the connector.
        
        Args:
            movies_df: DataFrame with MovieLens movies
            api_sleep: Sleep time between API calls to avoid rate limiting
        """
        self.movies_df = movies_df
        self.api_sleep = api_sleep
        self.dbpedia_endpoint = "http://dbpedia.org/sparql"
        self.wikidata_endpoint = "https://query.wikidata.org/sparql"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ExternalKnowledgeConnector')
        
        # Dictionary to cache entities to avoid duplicate lookups
        self.entity_cache = {
            'dbpedia': {},
            'wikidata': {}
        }
        
    def search_movie_in_dbpedia(self, movie_title, movie_year=None):
        """
        Search for a movie in DBpedia using SPARQL.
        
        Args:
            movie_title: The title of the movie
            movie_year: Optional release year to improve matching
            
        Returns:
            Dictionary with DBpedia entity URI and basic info if found
        """
        # Clean the movie title - remove year if it's in the title
        clean_title = movie_title
        if '(' in clean_title and ')' in clean_title:
            clean_title = clean_title.split('(')[0].strip()
        
        # Check cache first
        cache_key = f"{clean_title}_{movie_year if movie_year else ''}"
        if cache_key in self.entity_cache['dbpedia']:
            return self.entity_cache['dbpedia'][cache_key]
            
        sparql = SPARQLWrapper(self.dbpedia_endpoint)
        
        # Build query - search for movies with the given title
        # Optionally filter by year if provided
        query = """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT DISTINCT ?movie ?title ?abstract ?year WHERE {
            ?movie a dbo:Film ;
                   rdfs:label ?title ;
                   dbo:abstract ?abstract .
            OPTIONAL { ?movie dbo:releaseDate ?date . 
                      BIND(YEAR(?date) AS ?year) }
            
            FILTER(LANG(?title) = 'en')
            FILTER(LANG(?abstract) = 'en')
            FILTER(REGEX(?title, "%s", "i"))
        }
        """ % clean_title
        
        if movie_year:
            query += f" FILTER(?year = {movie_year})"
            
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        try:
            results = sparql.query().convert()
            
            if results["results"]["bindings"]:
                # Get the first result
                result = results["results"]["bindings"][0]
                movie_uri = result["movie"]["value"]
                
                entity_data = {
                    'uri': movie_uri,
                    'title': result["title"]["value"] if "title" in result else None,
                    'abstract': result["abstract"]["value"] if "abstract" in result else None,
                    'year': result["year"]["value"] if "year" in result else None
                }
                
                # Cache the result
                self.entity_cache['dbpedia'][cache_key] = entity_data
                
                return entity_data
            else:
                self.logger.info(f"No DBpedia entity found for: {movie_title}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error querying DBpedia for {movie_title}: {e}")
            return None
        finally:
            # Sleep to avoid rate limiting
            time.sleep(self.api_sleep)
    
    def search_movie_in_wikidata(self, movie_title, movie_year=None):
        """
        Search for a movie in Wikidata using SPARQL.
        
        Args:
            movie_title: The title of the movie
            movie_year: Optional release year to improve matching
            
        Returns:
            Dictionary with Wikidata entity ID and basic info if found
        """
        # Clean the movie title - remove year if it's in the title
        clean_title = movie_title
        if '(' in clean_title and ')' in clean_title:
            clean_title = clean_title.split('(')[0].strip()
        
        # Check cache first
        cache_key = f"{clean_title}_{movie_year if movie_year else ''}"
        if cache_key in self.entity_cache['wikidata']:
            return self.entity_cache['wikidata'][cache_key]
            
        sparql = SPARQLWrapper(self.wikidata_endpoint)
        sparql.setReturnFormat(JSON)
        
        # Build query - search for movies with the given title
        # Optionally filter by year if provided
        query = """
        SELECT ?movie ?movieLabel ?year WHERE {
          ?movie wdt:P31 wd:Q11424.  # Instance of film
          ?movie rdfs:label ?movieLabel.
          FILTER(LANG(?movieLabel) = "en").
          FILTER(REGEX(?movieLabel, "%s", "i")).
          OPTIONAL { ?movie wdt:P577 ?date . 
                    BIND(YEAR(?date) AS ?year) }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """ % clean_title
        
        if movie_year:
            query += f" FILTER(?year = {movie_year})"
            
        sparql.setQuery(query)
        
        try:
            results = sparql.query().convert()
            
            if results["results"]["bindings"]:
                # Get the first result
                result = results["results"]["bindings"][0]
                movie_uri = result["movie"]["value"]
                
                entity_data = {
                    'uri': movie_uri,
                    'id': movie_uri.split('/')[-1],
                    'title': result["movieLabel"]["value"] if "movieLabel" in result else None,
                    'year': result["year"]["value"] if "year" in result else None
                }
                
                # Cache the result
                self.entity_cache['wikidata'][cache_key] = entity_data
                
                return entity_data
            else:
                self.logger.info(f"No Wikidata entity found for: {movie_title}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error querying Wikidata for {movie_title}: {e}")
            return None
        finally:
            # Sleep to avoid rate limiting
            time.sleep(self.api_sleep)
    
    def get_movie_details_from_dbpedia(self, movie_uri):
        """
        Get detailed information about a movie from DBpedia.
        
        Args:
            movie_uri: DBpedia URI for the movie
            
        Returns:
            Dictionary with movie details
        """
        sparql = SPARQLWrapper(self.dbpedia_endpoint)
        sparql.setReturnFormat(JSON)
        
        query = """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbp: <http://dbpedia.org/property/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?director ?directorName ?actor ?actorName ?genre ?runtime ?country
        WHERE {
          <%s> a dbo:Film .
          OPTIONAL { <%s> dbo:director ?director . 
                    ?director rdfs:label ?directorName . 
                    FILTER(LANG(?directorName) = 'en') }
          OPTIONAL { <%s> dbo:starring ?actor . 
                    ?actor rdfs:label ?actorName . 
                    FILTER(LANG(?actorName) = 'en') }
          OPTIONAL { <%s> dbo:genre ?genre }
          OPTIONAL { <%s> dbo:runtime ?runtime }
          OPTIONAL { <%s> dbo:country ?country }
        }
        """ % (movie_uri, movie_uri, movie_uri, movie_uri, movie_uri, movie_uri)
        
        sparql.setQuery(query)
        
        try:
            results = sparql.query().convert()
            
            directors = []
            actors = []
            genres = []
            runtime = None
            country = None
            
            for result in results["results"]["bindings"]:
                if "director" in result and "directorName" in result:
                    director = {
                        'uri': result["director"]["value"],
                        'name': result["directorName"]["value"]
                    }
                    if director not in directors:
                        directors.append(director)
                
                if "actor" in result and "actorName" in result:
                    actor = {
                        'uri': result["actor"]["value"],
                        'name': result["actorName"]["value"]
                    }
                    if actor not in actors:
                        actors.append(actor)
                
                if "genre" in result:
                    genre = result["genre"]["value"]
                    if genre not in genres:
                        genres.append(genre)
                
                if "runtime" in result and not runtime:
                    runtime = result["runtime"]["value"]
                
                if "country" in result and not country:
                    country = result["country"]["value"]
            
            return {
                'directors': directors,
                'actors': actors,
                'genres': genres,
                'runtime': runtime,
                'country': country
            }
                
        except Exception as e:
            self.logger.error(f"Error getting details from DBpedia for {movie_uri}: {e}")
            return {}
        finally:
            time.sleep(self.api_sleep)
    
    def get_movie_details_from_wikidata(self, entity_id):
        """
        Get detailed information about a movie from Wikidata.
        
        Args:
            entity_id: Wikidata entity ID (e.g., Q12345)
            
        Returns:
            Dictionary with movie details
        """
        sparql = SPARQLWrapper(self.wikidata_endpoint)
        sparql.setReturnFormat(JSON)
        
        query = """
        SELECT ?director ?directorLabel ?actor ?actorLabel ?genre ?genreLabel 
               ?award ?awardLabel ?boxOffice ?runtime ?budget
        WHERE {
          wd:%s wdt:P31 wd:Q11424.
          
          OPTIONAL { wd:%s wdt:P57 ?director. }  # director
          OPTIONAL { wd:%s wdt:P161 ?actor. }    # cast member
          OPTIONAL { wd:%s wdt:P136 ?genre. }    # genre
          OPTIONAL { wd:%s wdt:P166 ?award. }    # award received
          OPTIONAL { wd:%s wdt:P2142 ?boxOffice. } # box office
          OPTIONAL { wd:%s wdt:P2047 ?runtime. }   # duration
          OPTIONAL { wd:%s wdt:P2130 ?budget. }    # budget
          
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """ % (entity_id, entity_id, entity_id, entity_id, entity_id, entity_id, entity_id, entity_id)
        
        sparql.setQuery(query)
        
        try:
            results = sparql.query().convert()
            
            directors = []
            actors = []
            genres = []
            awards = []
            box_office = None
            runtime = None
            budget = None
            
            for result in results["results"]["bindings"]:
                if "director" in result and "directorLabel" in result:
                    director = {
                        'uri': result["director"]["value"],
                        'name': result["directorLabel"]["value"]
                    }
                    if director not in directors:
                        directors.append(director)
                
                if "actor" in result and "actorLabel" in result:
                    actor = {
                        'uri': result["actor"]["value"],
                        'name': result["actorLabel"]["value"]
                    }
                    if actor not in actors:
                        actors.append(actor)
                
                if "genre" in result and "genreLabel" in result:
                    genre = {
                        'uri': result["genre"]["value"],
                        'name': result["genreLabel"]["value"]
                    }
                    if genre not in genres:
                        genres.append(genre)
                
                if "award" in result and "awardLabel" in result:
                    award = {
                        'uri': result["award"]["value"],
                        'name': result["awardLabel"]["value"]
                    }
                    if award not in awards:
                        awards.append(award)
                
                if "boxOffice" in result and not box_office:
                    box_office = result["boxOffice"]["value"]
                
                if "runtime" in result and not runtime:
                    runtime = result["runtime"]["value"]
                
                if "budget" in result and not budget:
                    budget = result["budget"]["value"]
            
            return {
                'directors': directors,
                'actors': actors,
                'genres': genres,
                'awards': awards,
                'box_office': box_office,
                'runtime': runtime,
                'budget': budget
            }
                
        except Exception as e:
            self.logger.error(f"Error getting details from Wikidata for {entity_id}: {e}")
            return {}
        finally:
            time.sleep(self.api_sleep)
    
    def link_movielens_to_knowledge_graphs(self, output_file=None, source='both'):
        """
        Link all movies in the MovieLens dataset to DBpedia and/or Wikidata.
        
        Args:
            output_file: Optional file path to save the results
            source: Which external source to use ('dbpedia', 'wikidata', or 'both')
            
        Returns:
            DataFrame with movie IDs and their corresponding external entities
        """
        if self.movies_df is None:
            self.logger.error("No movies DataFrame provided. Cannot link to external knowledge.")
            return None
        
        # Prepare results DataFrame
        results = []
        
        for _, movie in tqdm(self.movies_df.iterrows(), total=len(self.movies_df), desc="Linking movies"):
            movie_id = movie['movie_id']
            title = movie['title']
            
            # Extract year from title if present
            year = None
            if '(' in title and ')' in title:
                year_str = title.split('(')[-1].split(')')[0]
                if year_str.isdigit():
                    year = int(year_str)
            
            # Search in external knowledge bases
            dbpedia_entity = None
            wikidata_entity = None
            
            if source in ['dbpedia', 'both']:
                dbpedia_entity = self.search_movie_in_dbpedia(title, year)
                
            if source in ['wikidata', 'both']:
                wikidata_entity = self.search_movie_in_wikidata(title, year)
            
            results.append({
                'movie_id': movie_id,
                'title': title,
                'dbpedia_uri': dbpedia_entity['uri'] if dbpedia_entity else None,
                'wikidata_id': wikidata_entity['id'] if wikidata_entity else None
            })
        
        # Convert to DataFrame
        links_df = pd.DataFrame(results)
        
        # Save to file if specified
        if output_file:
            links_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved entity links to {output_file}")
        
        return links_df
    
    def enrich_movie_data(self, links_df, output_file=None, source='both'):
        """
        Enrich MovieLens movies with detailed information from external knowledge bases.
        
        Args:
            links_df: DataFrame with movie IDs and external entity IDs
            output_file: Optional file path to save the results
            source: Which external source to use ('dbpedia', 'wikidata', or 'both')
            
        Returns:
            DataFrame with enriched movie information
        """
        enriched_data = []
        
        for _, row in tqdm(links_df.iterrows(), total=len(links_df), desc="Enriching movie data"):
            movie_id = row['movie_id']
            title = row['title']
            
            # Initialize with basic info
            movie_data = {
                'movie_id': movie_id,
                'title': title,
                'dbpedia_uri': row['dbpedia_uri'],
                'wikidata_id': row['wikidata_id'],
                'directors': [],
                'actors': [],
                'genres': [],
                'awards': [],
                'runtime': None,
                'budget': None,
                'box_office': None,
                'country': None
            }
            
            # Get details from DBpedia
            if source in ['dbpedia', 'both'] and row['dbpedia_uri']:
                dbpedia_details = self.get_movie_details_from_dbpedia(row['dbpedia_uri'])
                
                if dbpedia_details:
                    movie_data['directors'].extend(dbpedia_details.get('directors', []))
                    movie_data['actors'].extend(dbpedia_details.get('actors', []))
                    movie_data['genres'].extend(dbpedia_details.get('genres', []))
                    movie_data['runtime'] = dbpedia_details.get('runtime') or movie_data['runtime']
                    movie_data['country'] = dbpedia_details.get('country')
            
            # Get details from Wikidata
            if source in ['wikidata', 'both'] and row['wikidata_id']:
                wikidata_details = self.get_movie_details_from_wikidata(row['wikidata_id'])
                
                if wikidata_details:
                    movie_data['directors'].extend(wikidata_details.get('directors', []))
                    movie_data['actors'].extend(wikidata_details.get('actors', []))
                    movie_data['genres'].extend([g['name'] for g in wikidata_details.get('genres', [])])
                    movie_data['awards'] = wikidata_details.get('awards', [])
                    movie_data['runtime'] = wikidata_details.get('runtime') or movie_data['runtime']
                    movie_data['budget'] = wikidata_details.get('budget')
                    movie_data['box_office'] = wikidata_details.get('box_office')
            
            # Remove duplicates
            movie_data['directors'] = [dict(t) for t in {tuple(d.items()) for d in movie_data['directors']}]
            movie_data['actors'] = [dict(t) for t in {tuple(d.items()) for d in movie_data['actors']}]
            movie_data['genres'] = list(set(movie_data['genres']))
            
            enriched_data.append(movie_data)
        
        # Convert to DataFrame
        enriched_df = pd.DataFrame(enriched_data)
        
        # Save to file if specified
        if output_file:
            # Convert list columns to string for easier storage
            for col in ['directors', 'actors', 'genres', 'awards']:
                enriched_df[col] = enriched_df[col].apply(json.dumps)
            
            enriched_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved enriched movie data to {output_file}")
            
            # Convert back to lists
            for col in ['directors', 'actors', 'genres', 'awards']:
                enriched_df[col] = enriched_df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        return enriched_df