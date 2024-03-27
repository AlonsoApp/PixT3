from tqdm import tqdm
from itertools import repeat
import multiprocessing.pool as mpp

def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter, total):
	args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
	#return pool.starmap(apply_args_and_kwargs, args_for_starmap)
	for _ in tqdm(pool.istarmap(apply_args_and_kwargs, args_for_starmap), total=total):
		pass

def apply_args_and_kwargs(fn, args, kwargs):
	return fn(*args, **kwargs)

def istarmap(self, func, iterable, chunksize=16):
	"""starmap-version of imap
	"""
	self._check_running()
	if chunksize < 1:
		raise ValueError(
			"Chunksize must be 1+, not {0:n}".format(
				chunksize))

	task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
	result = mpp.IMapIterator(self)
	self._taskqueue.put(
		(
			self._guarded_task_generation(result._job,
										  mpp.starmapstar,
										  task_batches),
			result._set_length
		))
	return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap