from ompgov_api.query import query_results
import json
import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

def get_sessions(category=None, createdAfter=None, 
				 updatedAfter=None, livestream=None,
				 start=None, limit=None):
	"""
	Same as /sessions
	
	kwargs:
		category: array of ints
		createdAfter: int
		updatedAfter: int
		livestream: bool
		start: int
		limit: int
	returns:
		dict
	"""
	return query_results('sessions', category=category, 
					createdAfter=createdAfter, 
					updatedAfter=updatedAfter,
					livestream=livestream,
					start=start, limit=limit)

def get_session_by_id(session_id):
	"""
	Same as /sessions/{sessionId}
	
	args:
		session_id: int
	returns:
		dict
	"""
	return query_results('sessions/{}'.format(session_id))

def get_sites(state=None, start=None, limit=None):
	"""
	Same as /sites
	
	kwargs:
		state: str
		start: int
		limit: int
	returns:
		dict
	"""
	with open("{}/config.cfg".format(BASE_PATH)) as f:
		config = json.load(f)
	return query_results('sites', key=config['key'],
					 state=state, start=start, 
					 limit=limit)

def get_site_by_name(site_name):
	"""
	Same as /sites/{siteName}
	
	args:
		site_name: str
	returns:
		dict
	"""
	return query_results('sites/{}'.format(site_name))

def get_session_by_site_id(site_id, category=None, createdAfter=None,
						   updatedAfter=None, livestream=None,
						   start=None, limit=None):
	"""
	Same as /site/{siteId}/sessions
	
	args:
		site_id: int
	kwargs:
		category: array of ints
		createdAfter: int
		updatedAfter: int
		livestream: bool
		start: int
		limit: int
	returns:
		dict
	"""
	return query_results('site/{}/sessions'.format(site_id), 
					 category=category, createdAfter=createdAfter,
					 updatedAfter=updatedAfter, livestream=livestream,
					 start=start, limit=limit)

def get_captions_by_site_id(site_id, start=None, limit=None):
	"""
	Same as /site/{siteId}/captions
	
	args:
		site_id: int
	keywords
		start: int
		limit: int
	returns:
		dict
	"""
	return query_results('site/{}/captions'.format(site_id),
					 start=start, limit=limit)

def get_cuepoints_by_site_id(site_id, billId=None, start=None, limit=None):
	"""
	Same as /site/{siteId}/cuepoints
	
	args:
		site_id: int
	keywords
		start: int
		limit: int
	returns:
		dict
	"""
	return query_results('site/{}/cuepoints'.format(site_id),
					 billId=billId, start=start, limit=limit)

def get_categories_by_site_id(site_id, livestream=None,
							  start=None, limit=None):
	"""
	Same as /site/{siteId}/categories
	
	args:
		site_id: int
	keywords
		livestream: bool
		start: int
		limit: int
	returns:
		dict
	"""
	if livestream:
		livestream = str(livestream).lower()
	return query_results('site/{}/categories'.format(site_id), 
					livestream=livestream, start=start, 
					limit=limit)

def get_captions(start=None, limit=None):
	"""
	Same as /captions
	
	kwargs:
		start: int
		limit: int
	returns:
		dict
	"""
	return query_results('captions', start=start, limit=limit)

def get_cuepoints(billId=None, start=None, limit=None):
	"""
	Same as /cuepoints
	
	kwargs:
		billId: str
		start: int
		limit: int
	returns:
		dict
	"""
	return query_results('cuepoints', billId=billId, start=start, limit=limit)

def get_categories(livestream=None, start=None, limit=None):
	"""
	Same as /categories
	
	kwargs:
		livestream: bool
		start: int
		limit: int
	returns:
		dict
	"""
	if livestream:
		livestream = str(livestream).lower()
	return query_results('categories', 
					livestream=livestream, start=start, 
					limit=limit)