import requests
import json
import sys

BASE_URL = 'http://open.ompnetwork.org/api/{}'

class ParameterError(Exception):
	pass

def get_query_result_length(api_path, **kwargs):
	"""
	Returns total size of api query results
	
	args:
		api_path: string name of OMF API json path
	kwargs:
		category: array of ints
		createdAfter: int
		updatedAfter: int
		state: str
		livestream: bool
		billId: string 
	returns:
		int or None
	"""
	start = kwargs.get('start',0)
	limit = kwargs.pop('limit',0)
	q_string = get_string_query(api_path, limit=1, **kwargs)
	r = run_query(q_string)
	try:
		return int(r['totalSize'])
	except KeyError:
		return None

def get_all_query_strings(api_path, **kwargs):
	"""
	Creates all API query strings required to return all results.

	args:
		api_path: string name of OMF API json path
	kwargs:
		category: array of ints
		createdAfter: int
		updatedAfter: int
		state: str
		livestream: bool
		billId: string 
	returns:
		list of strings
	"""
	totalSize = get_query_result_length(api_path, **kwargs)
	start = kwargs.pop('start', 0)
	search_limit = kwargs.pop('limit', totalSize)

	if search_limit > totalSize:
		search_limit = totalSize

	if search_limit >= 500:
		page_limit = 500
	else:
		page_limit = search_limit

	queries = []
	if not search_limit:
		qs = get_string_query(api_path, start=start, limit=page_limit, **kwargs)
		queries.append(qs)
	elif start < search_limit:
		while start < search_limit:
			qs = get_string_query(api_path, start=start, limit=page_limit, **kwargs)
			queries.append(qs)
			start += page_limit
	else:
		qs = get_string_query(api_path, start=start, limit=page_limit, **kwargs)
		queries.append(qs)
		
	return queries

def check_filters(api_path, params):
	"""
	Verifies that query parameters match with API path model.

	args:
		api_path: string name of OMF API path
	kwargs:
		category: array of ints
		createdAfter: int
		updatedAfter: int
		state: str
		livestream: bool
		billId: string
	raises:
		ParameterError
	"""
	fltr_map = {
		'sessions': ['category','createdAfter',
					 'updatedAfter', 'livestream',
					 'start','limit'],
		'sites': ['key','state', 'start', 'limit'],
		'categories': ['livestream', 'start', 'limit'],
		'cuepoints': ['key','start', 'limit', 'billId']}
	try:	
		unacceptable_params = [k for k in params.keys() if 
							   k not in fltr_map[api_path.split('/')[-1]]]
		if unacceptable_params:
			raise ParameterError("Model '{}' cannot be searched by the "
								 "following parameters: {}".format(
								 	api_path.split('/')[-1], unacceptable_params))
		else:
			pass
	except KeyError:
		pass

def get_string_query(api_path, **kwargs):
	"""
	Write query string for OMF API
	
	args:
		api_path: string name of OMF API path
	kwargs:
		category: array of ints
		createdAfter: int
		updatedAfter: int
		state: str
		livestream: bool
		billId: string 
	returns:
		str
	"""
	start = kwargs.pop('start', 0)
	limit = kwargs.pop('limit', 500)

	params = {'start': start, 'limit': limit}
	params.update(kwargs)
	check_filters(api_path, params)
	if 'categories' in params.keys():
		params['categories'] = '&category='.join([str(i) for i in params['categories']])

	filters = '&'.join(["{}={}".format(k, v) for 
						k,v in params.items() if v])

	query_string = '?'.join([BASE_URL.format(api_path), filters])
	return query_string

def results_display(results, start=None, limit=None, totalSize=None):
	"""
	Create dict to imitate json results from OMF API

	args:
		results: list
	kwargs:
		start: int
		limit: int
		totalSize: int
	returns:
		dict
	"""
	if not start:
		start = 0
	if not limit:
		limit = len(results)
	if not totalSize:
		totalSize = len(results)
	total_result = {
		'start': start,
		'limit': limit,
		'totalSize': totalSize,
		'size': len(results),
		'results': results
	}
	return total_result

def query_results(api_path, **kwargs):
	"""
	Gather all query strings, run queries, and compile results
	into single result display

	args:
		api_path: string name of OMF API path
	kwargs:
		category: array of ints
		createdAfter: int
		updatedAfter: int
		state: str
		livestream: bool
		billId: string 
	returns:
		dict 
	"""
	kwargs = {k:v for (k,v) in kwargs.items() if v!=None}
	start = kwargs.pop('start', 0)
	queries = get_all_query_strings(api_path, start=start, **kwargs)
	req_res = [run_query(q) for q in queries]
	results = sum([r['results'] for r in req_res],[])
	totalSize = int(req_res[0]['totalSize'])
	limit = kwargs.get('limit', totalSize)
	return results_display(results, start, limit, totalSize)

def run_query(query_string):
	r = requests.get(query_string)
	if r.status_code == 200:
		json_data = json.loads(r.text)
		return process_query_result(json_data)
	else:
		raise Exception("Request failed")

def process_query_result(json_result):
	if not all([k in json_result.keys() for k in 
		['start', 'limit', 'totalSize', 'size', 'results']]):
		return results_display([json_result])
	else:
		return json_result

