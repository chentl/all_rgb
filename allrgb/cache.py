import pickle
import hashlib
import os


_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')


class CacheLoadError(Exception):
    pass


class CacheStorage:
    def __init__(self, app_name):
        self.app_name = app_name

    def _tags_to_path(self, tags):
        cache_name = '%s_%s.cache' % (self.app_name, '_'.join(tags))
        return os.path.join(_cache_dir, cache_name)

    def write_cache(self, tags, obj):
        cache_path = self._tags_to_path(tags)
        with open(cache_path, 'wb') as f:
            pickle.dump(obj, f, protocol=4)

    def load_cache(self, tags):
        cache_path = self._tags_to_path(tags)
        try:
            with open(cache_path, 'rb') as f:
                obj = pickle.load(f)
            print('[load_cache]', os.path.basename(cache_path))
            return obj
        except (FileNotFoundError, TypeError, pickle.UnpicklingError):
            raise CacheLoadError
