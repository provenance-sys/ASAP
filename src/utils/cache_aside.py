'''
SQLiteDB:
A class to interact with SQLite database using sqlite3
Some code reffers from sqlitedict package
'''
import os
from typing import Any, Optional, Tuple, List
import sqlite3
from collections import OrderedDict
from collections.abc import MutableMapping
from collections import UserDict as DictClass
from pickle import dumps, loads, HIGHEST_PROTOCOL as PICKLE_PROTOCOL
from base64 import b64decode, b64encode

__all__ = ['LRUCache', 'SQLiteDB', 'CacheAside']

class LRUCache(MutableMapping):
    def __init__(self, maxlen: int, items: List[Tuple] = None, pop_n: int = 20):
        '''
        Initialize the LRU cache with a maximum length and optional items
        
        Parameters
        ----------
        maxlen : int
            The maximum length of the cache
        pop_n : int, optional
            The number of items to pop when the cache is full, by default 20
        items : List[Tuple[str, str]], optional
            A list of key-value pairs to initialize the cache with, by default None
        Examples
        --------
        >>> lru = LRUCache(3, [('a', 1), ('b', 2), ('c', 3)], 1)
        >>> lru['a']  # {'b': 2, 'c': 3, 'a': 1}
        1
        >>> lru['d'] = 4  # {'c': 3, 'a': 1, 'd': 4}
        >>> lru.set('e', 5)  # {'a': 1, 'd': 4, 'e': 5}, return 'c'
        [('c', 3)]
        '''
        if maxlen < pop_n:
            raise ValueError('pop_n must be less than maxlen')
        self.maxlen = maxlen
        self.pop_n = pop_n
        self.d = OrderedDict()
        if items is not None:
            for k, v in items:
                self[k] = v

    def __str__(self) -> str:
        return f'LRUCache({str(dict(self.d))})'
    
    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        if key not in self.d:
            raise KeyError(key)
        self.d.move_to_end(key)
        return self.d[key]

    def __setitem__(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        else:
            if len(self.d) >= self.maxlen:
                self.popitem(self.pop_n, last=False)
            self.d[key] = value

    def __delitem__(self, key):
        del self.d[key]

    def __iter__(self):
        return self.d.__iter__()

    def __len__(self):
        return len(self.d)
    
    def popitem(self, pop_n, last=False) ->Tuple[Tuple]:
        res = []
        for _ in range(pop_n):
            res.append(self.d.popitem(last=last))
        return tuple(res)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key, value) -> Optional[Tuple[Tuple]]:
        '''
        Update the cache with a new key-value pair, if the key is already in the cache 
        move it to the end, else return the earliest n items that were popped.
        '''
        if key in self.d:
            self.d.move_to_end(key)
            return None
        else:
            if len(self.d) < self.maxlen:
                self.d[key] = value
                return None
            else:
                self.d[key] = value
                return self.popitem(self.pop_n, last=False)
    
    def clear(self):
        '''
        Clear all items from the cache.
        '''
        self.d.clear()

def encode(obj):
    """Serialize an object using pickle to a binary format accepted by SQLite."""
    return sqlite3.Binary(dumps(obj, protocol=PICKLE_PROTOCOL))


def decode(obj):
    """Deserialize objects retrieved from SQLite."""
    return loads(bytes(obj))


def encode_key(key):
    """Serialize a key using pickle + base64 encoding to text accepted by SQLite."""
    return b64encode(dumps(key, protocol=PICKLE_PROTOCOL)).decode("ascii")


def decode_key(key):
    """Deserialize a key retrieved from SQLite."""
    return loads(b64decode(key.encode("ascii")))


def identity(obj):
    """Identity f(x) = x function for encoding/decoding."""
    return obj


class SQLiteDB(DictClass):
    def __init__(
            self, filename: str = 'test.db', tablename: str = 'default_table',
            flag: str = 'w', cache_size: int=4000, synchronous: str='OFF',
            journal_mode: str='MEMORY',encode=encode, decode=decode,
            encode_key=identity, decode_key=identity):
        '''
        Initialize the SQLite database based on SqliteDict.
        
        Parameters
        ----------
        file_name : str
            The name of the database file.
        table_name : str
            The name of the table.
        flag : str
            The flag to open the database file with.
            'w' default, open for r/w, but drop `tablename` contents first (start with empty table)
            'n' create a new database (erasing any existing tables, not just `tablename`!).
        cache_size : int
            The size of the cache.
        synchronous : str
            The synchronous mode of the database.
        journal_mode : str
            The journal mode of the database.
        encode : function
            The encoding function for the values.
        decode : function
            The decoding function for the values.
        encode_key : function
            The encoding function for the keys.
        decode_key : function
            The decoding function for the keys.
        '''
        self.file_name = filename
        self.table_name = tablename
        self.cache_size = cache_size
        self.synchronous = synchronous
        self.journal_mode = journal_mode
        self.encode = encode
        self.decode = decode
        self.encode_key = encode_key
        self.decode_key = decode_key

        if flag == 'n' and os.path.exists(filename):
            os.remove(filename)
        
        self.conn = sqlite3.connect(filename)
        self.cursor = self.conn.cursor()
        self.__init_pragma(
            cache_size=self.cache_size, 
            synchronous=self.synchronous, 
            journal_mode=self.journal_mode
            )

        # Use standard SQL escaping of double quote characters in identifiers, by doubling them.
        # See https://github.com/RaRe-Technologies/sqlitedict/pull/113
        self.table_name = tablename.replace('"', '""')
        MAKE_TABLE = f"CREATE TABLE IF NOT EXISTS {self.table_name} (key TEXT PRIMARY KEY, value BLOB);"
        self.conn.execute(MAKE_TABLE)
        self.conn.commit()
        if flag == 'w':
            self.delete_table()

    def __init_pragma(self, **kv_pairs):
        for pragma, value in kv_pairs.items():
            self.set_pragma(pragma, value)

    def set_pragma(self, pragma: str, value: str):
        self.cursor.execute(f"PRAGMA {pragma} = {value};")
        self.conn.commit()
    
    def __str__(self):
        return f"SqliteDict({self.file_name})"

    def __repr__(self):
        return str(self)  # no need of something complex

    def __contains__(self, key):
        HAS_ITEM = f'SELECT 1 FROM "{self.table_name}" WHERE key = ?'
        return self.conn.execute(HAS_ITEM, (self.encode_key(key),)).fetchone() is not None

    def __getitem__(self, key):
        GET_ITEM = f'SELECT value FROM "{self.table_name}" WHERE key = ?'
        item = self.conn.execute(GET_ITEM, (self.encode_key(key),)).fetchone()
        if item is None:
            raise KeyError(key)
        return self.decode(item[0])

    def __setitem__(self, key, value):
        ADD_ITEM = f'REPLACE INTO "{self.table_name}" (key, value) VALUES (?,?)'
        self.conn.execute(ADD_ITEM, (self.encode_key(key), self.encode(value)))
        self.conn.commit()

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)
        DEL_ITEM = f'DELETE FROM "{self.table_name}" WHERE key = ?'
        self.conn.execute(DEL_ITEM, (self.encode_key(key),))
        self.conn.commit()

    def __len__(self):
        # `select count (*)` is super slow in sqlite (does a linear scan!!)
        # As a result, len() is very slow too once the table size grows beyond trivial.
        # We could keep the total count of rows ourselves, by means of triggers,
        # but that seems too complicated and would slow down normal operation
        # (insert/delete etc).
        GET_LEN = f'SELECT COUNT(*) FROM "{self.table_name}"'
        rows = self.conn.execute(GET_LEN).fetchone()[0]
        return rows if rows is not None else 0

    def __bool__(self):
        # No elements is False, otherwise True
        GET_MAX = f'SELECT MAX(ROWID) FROM "{self.table_name}"'
        m = self.conn.execute(GET_MAX).fetchone()[0]
        return True if m is not None else False

    def iterkeys(self):
        GET_KEYS = f'SELECT key FROM "{self.table_name}" ORDER BY rowid'
        for key in self.conn.execute(GET_KEYS):
            yield self.decode_key(key[0])

    def itervalues(self):
        GET_VALUES = f'SELECT value FROM "{self.table_name}" ORDER BY rowid'
        for value in self.conn.execute(GET_VALUES):
            yield self.decode(value[0])

    def iteritems(self):
        GET_ITEMS = f'SELECT key, value FROM "{self.table_name}" ORDER BY rowid'
        for key, value in self.conn.execute(GET_ITEMS):
            yield self.decode_key(key[0]), self.decode(value[0])

    def keys(self):
        return self.iterkeys()

    def values(self):
        return self.itervalues()

    def items(self):
        return self.iteritems()

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __iter__(self):
        return self.iterkeys()
    
    def set(self, items=()):
        '''
        Update the database with a list of key-value pairs.
        '''        
        INSERT_MANY = f'INSERT OR IGNORE INTO "{self.table_name}" (key, value) VALUES (?,?)'
        self.conn.executemany(INSERT_MANY, ((self.encode_key(k), self.encode(v)) for k, v in items))
        self.conn.commit()

    @staticmethod
    def get_tablenames(filename):
        """get the names of the tables in an sqlite db as a list"""
        if not os.path.isfile(filename):
            raise IOError('file %s does not exist' % (filename))
        GET_TABLENAMES = 'SELECT name FROM sqlite_master WHERE type="table"'
        with sqlite3.connect(filename) as conn:
            cursor = conn.execute(GET_TABLENAMES)
            res = cursor.fetchall()

        return [name[0] for name in res]
    
    def delete_table(self):
        DELETE_TABLE = f"DELETE FROM {self.table_name}"
        self.cursor.execute(DELETE_TABLE)
        self.conn.commit()

    def close(self):
        print('SQLite connection closed.')
        self.conn.close()


class CacheAside:
    def __init__(self, cache: LRUCache, db: SQLiteDB):
        '''
        Initialize the CacheAside class with a cache and a database.
        Usage is similar to a dictionary.
        '''
        self.cache = cache
        self.db = db
    
    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        else:
            res = self.db[key]
            self.set(key, res)  # update the cache, and pop the earliest n items if the cache is full
            return res
    
    def __setitem__(self, key, value):
        '''
        same as set method
        '''
        poped_items = self.cache.set(key, value)
        if poped_items:
            self.db.set(poped_items)

    def __delitem__(self, key):
        if key in self.cache:
            del self.cache[key]
        if key in self.db:
            del self.db[key]
    
    def get(self, key, default=None) -> Optional[Any]:
        '''
        Get the value of a key from the cache if it exists, else get it from the database.
        if the key is not in the database, return default.
        '''
        res = self.cache.get(key)
        if res is None:
            res = self.db.get(key)
            if res is None:
                return default
            else:
                self.set(key, res)
        return res

    def set(self, key, value):
        '''
        Update the cache with a new key-value pair.
        '''
        poped_items = self.cache.set(key, value)
        if poped_items:
            self.db.set(poped_items)
    
    def pop_cache_to_db(self):
        '''
        Pop all k-v from the cache to the database.
        '''
        try:
            poped_items = self.cache.popitem(len(self.cache), last=False)
            if poped_items:
                self.db.set(poped_items)
        except KeyError:
            raise KeyError('KEY NOT FOUND IN CACHE')
