import os
import pickle

from .log import auto_log

_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')

if not os.path.exists(_cache_dir):
    os.makedirs(_cache_dir)


class CacheLoadError(Exception):
    pass


class CacheStorage:
    """ Class for handling cache files. """

    def __init__(self, app_name):
        """
        Create new CacheStorage instance for a app.

        :param app_name: name of the app.
        """

        self.app_name = app_name

    def _tags_to_path(self, tags):
        """ Return path to cache file using tags """

        cache_name = '%s_%s.cache' % (self.app_name, '_'.join(tags))
        return os.path.join(_cache_dir, cache_name)

    def write_cache(self, tags, obj):
        """
        Write an object to cache on disk.

        :param tags: A tuple of tag strings used to identify the cache.
        :param obj: The object to be cached.
        :return: None
        """
        cache_path = self._tags_to_path(tags)
        with open(cache_path, 'wb') as f:
            pickle.dump(obj, f, protocol=4)

    def load_cache(self, tags):
        """
        Fina cache file by tags and load them from disk.

        :param tags: A tuple of tag strings used to identify the cache.
        :return: The object stored in cache.
        :raise CacheLoadError: The cache file does not exist or could not be loaded.
        """

        cache_path = self._tags_to_path(tags)
        try:
            with open(cache_path, 'rb') as f:
                obj = pickle.load(f)
            auto_log('Load ' + os.path.basename(cache_path), level='debug')
            return obj
        except (FileNotFoundError, TypeError, pickle.UnpicklingError):
            raise CacheLoadError
