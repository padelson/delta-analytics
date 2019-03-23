from ompgov_api.query import run_query, check_filters, results_display
import json
import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

def get_mock_data(model):
	fpath = os.path.join(BASE_PATH, 'test_data', '{}.json'.format(model))
	with open(fpath) as f:
		data = json.load(f)
	return data

def filter_results(json_data, model, **kwargs):
	"""
	Filters stored data based on kwargs to mock api queries

	args:
		json_data: stored api json data
		model: json object model
	kwargs:
		category: array of ints, default to None
		createdAfter: int, default to None
		updatedAfter: int, default to None
		state: str, default to None
		livestream: bool, default to None
		billId: string, default to None
	"""
	check_filters(model, kwargs)
	category = kwargs.get('category', None)
	createdAfter = kwargs.get('createdAfter', None)
	updatedAfter = kwargs.get('updatedAfter', None)
	state = kwargs.get('state', None)
	livestream = kwargs.get('livestream', None)
	billId = kwargs.get('billId', None)
	start = kwargs.get('start', 0)
	if not start:
		start = 0
	limit = kwargs.get('limit', None)

	total_results = []
	
	for result_obj in json_data['results']:
		obj_passes = True
		try:
			if category:
				if not any([d['id'] in category for d in result_obj['categories']]):
					obj_passes = False
			if obj_passes and createdAfter:
				if not int(result_obj['created']) < int(createdAfter):
					obj_passes = False
			if obj_passes and updatedAfter:
				if not int(result_obj['updated']) > int(updatedAfter):
					obj_passes = False
			if obj_passes and state:
				if not (result_obj['state']==state or result_obj['state_abbrev']==state):
					obj_passes = False
			if obj_passes and livestream!=None:
				if not result_obj['livestream'] == str(int(livestream)):
					obj_passes = False
			if obj_passes and billId:
				if not any(('HB19-1011' in v) for v in result_obj.values() if v!=None):
					obj_passes = False
		except KeyError:
			obj_passes = False

		if obj_passes:
			total_results.append(result_obj)
	if limit:
		results = total_results[start:start+limit]
	else:
		results = total_results[start:]
	mock_query_res = {'start': start, 'size': len(results),
					  'results': results}
	return results_display(results, start, limit, len(total_results))

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
	name = 'sessions'
	return filter_results(get_mock_data(name), name, 
						  category=category, 
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
	name = 'sessions'
	data = get_mock_data(name)
	results = [d for d in data['results'] if d['id']==str(session_id)]
	return results_display(results)

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
	name = 'sites'
	return filter_results(get_mock_data(name), name,
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
	name = 'sites'
	data = get_mock_data(name)
	results = [d for d in data['results'] if d['om_user_settings_standalone_site']==site_name]
	return results_display(results)

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
	name = 'site/{}/sessions'.format(str(site_id))
	return filter_results(get_mock_data(name), name, 
						  category=category,
						  createdAfter=createdAfter,
						  updatedAfter=updatedAfter,
						  livestream=livestream,
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
	name = 'site/{}/captions'.format(str(site_id))
	return filter_results(get_mock_data(name), name,
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
	name = 'site/{}/cuepoints'.format(str(site_id))
	return filter_results(get_mock_data(name), name,
						  billId=billId,
						  start=start,
						  limit=limit)

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
	name = 'site/{}/categories'.format(str(site_id))
	return filter_results(get_mock_data(name), name,
						  livestream=livestream,
						  start=start,
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
	name = 'captions'
	return filter_results(get_mock_data(name), name,
						  start=start,
						  limit=limit)

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
	name = 'cuepoints'
	return filter_results(get_mock_data(name), name,
						  billId=billId,
						  start=start,
						  limit=limit)

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
	name = 'categories'
	return filter_results(get_mock_data(name), name,
						  livestream=livestream,
						  start=start,
						  limit=limit)