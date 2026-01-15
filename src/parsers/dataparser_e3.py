from typing import Dict, Tuple, Optional
import ujson
import re
import os
import sys
import hashlib

current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from utils.opreventmodel import OPREventModel
from utils.cache_aside import CacheAside, LRUCache, SQLiteDB
from utils import config

__all__ = [
    'DarpaE3CadetsParser',
    'DarpaE3TheiaParser',
    'DarpaE3TraceParser',
    'DarpaE3FivedirectionsParser',
    'DarpaE3ClearscopeParser',
    ]

def find_path(text):
    # Modified Unix-style path regex
    # start with /, and end with space
    path_regex = r"(^/[^ ]*)"
    match = re.search(path_regex, text)
    if match:
        return match.group()
    else:
        return None

class BasicParser:
    def __init__(
            self, mode: str = 'ca',  # 'im', 'db', 'ca'
            # SQLiteDB
            db_name: str = 'test.db',
            tablename: str = 'default_table',
            flag: str = 'w',
            cache_size: int = 4000,
            synchronous: str = 'OFF',
            journal_mode: str = 'MEMORY',
            # LRU
            cache_length: int = 100,
            pop_n: int = 10,
            ):
        self._mode = mode
        if self._mode == 'ca':
            self.kvmap = CacheAside(
                LRUCache(maxlen=cache_length, pop_n=pop_n),
                SQLiteDB(filename=db_name, tablename=tablename, flag=flag, cache_size=cache_size, synchronous=synchronous, journal_mode=journal_mode)
            )
        elif self._mode == 'im':
            pass
        elif self._mode == 'db':
            pass
        else:
            raise ValueError('Invalid mode. Valid modes are "im", "db", and "ca".')

    def convert_to_OPREM(self, *attrs) -> OPREventModel:
        '''Update the event model from the parser item.'''
        edge_type, sub_id, sub_type, sub_content, sub_malicious_flag, obj_id, obj_type, obj_content, obj_malicious_flag, timestamp = attrs
        timestamp = int(timestamp)
        sub_malicious_flag = int(sub_malicious_flag)
        obj_malicious_flag = int(obj_malicious_flag)
        # if edge_type in config.edge_reversed, reverse the source and target node.
        if edge_type in config.edge_reversed:
            sub_id, obj_id = obj_id, sub_id
            sub_type, obj_type = obj_type, sub_type
            sub_content, obj_content = obj_content, sub_content
            sub_malicious_flag, obj_malicious_flag = obj_malicious_flag, sub_malicious_flag
        oprem = OPREventModel(
            {'nid': sub_id,
            'type': sub_type, 'content': sub_content, 'flag': sub_malicious_flag},
            {'nid': obj_id, 
            'type': obj_type, 'content': obj_content, 'flag': obj_malicious_flag},
            {'type': edge_type, 'ts': timestamp, 'te': timestamp}
        )
        return oprem

    def parse_single_entry(self, entry: str) -> Optional[Tuple[str, ...]]:
        return NotImplementedError

    def parse_event(self, entry: str):
        raise NotImplementedError
    
    def parse_subject(self, entry: str):
        raise NotImplementedError
    
    def parse_fileobject(self, entry: str):
        raise NotImplementedError
    
    def parse_netflowobject(self, entry: str):
        raise NotImplementedError

    def get(self, key: str) -> Optional[Dict]:
        return self.kvmap.get(key)
    
    def set(self, key: str, value: Dict):
        self.kvmap[key] = value

    def pop_cache_to_db(self):
        self.kvmap.pop_cache_to_db()


class DarpaE3CadetsParser(BasicParser):
    '''
    E3 Cadets dataset parser.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_type = set()
        self.subject_type = set()
        # self.file1 = open('/data2/sj/DARPA_Engagement3/parsed_new/cadets/unknown_entry.json', 'a')

    def parse_single_entry(self, entry: str) -> Optional[Tuple[str, ...]]:
        if config.DarpaE3_reserved_type_2_strings_map['Event'] in entry:
            return self.parse_event(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['Subject'] in entry:
            return self.parse_subject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['FileObject'] in entry:
            return self.parse_fileobject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['NetFlowObject'] in entry:
            return self.parse_netflowobject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['UnnamedPipeObject'] in entry:
            return self.parse_unnamedPipeObject(entry)
        else:
            # # //////////////////////////////////////////////////////////////
            # if 'cdm18.SrcSinkObject' not in entry and 'cdm18.Principal' not in entry:
            #     self.file1.write(entry + '\n')
            # # //////////////////////////////////////////////////////////////
            return None
    
    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.Event": {
            "uuid": "28B5879A-6B54-5928-9B3B-B3ADA4D696D0",
            "sequence": {
                "long": 1262
            },
            "type": "EVENT_WRITE",
            "threadId": {
                "int": 100195
            },
            "hostId": "83C8ED1F-5045-DBCD-B39F-918F0DF4F851",
            "subject": {
                "com.bbn.tc.schema.avro.cdm18.UUID": "423C17F2-36B9-11E8-BF66-D9AA8AFF4A69"
            },
            "predicateObject": {
                "com.bbn.tc.schema.avro.cdm18.UUID": "ED281817-20F6-7E52-B620-7E7B827E05AF"
            },
            "predicateObjectPath": {
                "string": "/dev/tty"
            },
            "predicateObject2": null,
            "predicateObject2Path": null,
            "timestampNanos": 1522706902583352904,
            "name": {
                "string": "aue_write"
            },
            "parameters": {
                "array": []
            },
            "location": null,
            "size": {
                "long": 1
            },
            "programPoint": null,
            "properties": {
                "map": {
                "host": "83c8ed1f-5045-dbcd-b39f-918f0df4f851",
                "partial_path": "/dev/tty",
                "fd": "2",
                "exec": "sh",
                "ppid": "2336"
                }
            }
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_FREEBSD_DTRACE_CADETS"
    }
    '''
    def parse_event(self, entry: str) -> Tuple:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['Event']]
        timestamp = entry_attrs['timestampNanos']
        relation_type = entry_attrs['type']
        if relation_type not in config.edge_type['e3cadets']:
            return None

        _subject = self.get_subject(entry_attrs)
        if _subject is None:
            return None
        sbj_id, sbj_type, sbj_content, sbj_malicious_flag = _subject
        
        _object = self.get_object(entry_attrs, 'predicateObject', 'predicateObjectPath')
        if _object is None:
            return None
        obj_id, obj_type, obj_content, obj_malicious_flag = _object

        oprem1 = self.convert_to_OPREM(relation_type, sbj_id, sbj_type, sbj_content, sbj_malicious_flag, obj_id, obj_type, obj_content, obj_malicious_flag, timestamp)

        _object2 = self.get_object(entry_attrs, 'predicateObject2', 'predicateObject2Path')
        if _object2 is None:
            return [oprem1]
        obj_id2, obj_type2, obj_content2, obj_malicious_flag2 = _object2

        oprem2 = self.convert_to_OPREM(relation_type, sbj_id, sbj_type, sbj_content, sbj_malicious_flag, obj_id2, obj_type2, obj_content2, obj_malicious_flag2, timestamp)
        return [oprem1, oprem2]
        
    def get_subject(self, entry_attrs):
        if entry_attrs['subject'] is None:  # skip unknown subject
            return None
        subjectid = entry_attrs['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        sbj_dict = self.get(subjectid)
        if sbj_dict is None:
            return None
        
        sbj_type = sbj_dict['type']
        try:
            sbj_content = entry_attrs['properties']['map']['exec']
        except KeyError:
            return None  # if 'exec' not in properties, skip the entry
        sbj_id = subjectid
        sbj_malicious_flag = -1
        return sbj_id, sbj_type, sbj_content, sbj_malicious_flag
    
    def get_object(self, entry_attrs, predobject='predicateObject', predobjectpath='predicateObjectPath'):  # 'predicateObject' or 'predicateObject2'
        if entry_attrs[predobject] is None:  # skip unknown object
            return None
        objectid = entry_attrs[predobject]['com.bbn.tc.schema.avro.cdm18.UUID']
        obj_dict = self.get(objectid)
        if obj_dict is None:  # skip if object is not in the cache
            return None
        
        obj_type = obj_dict['type']
        if obj_type in self.file_type:
            if entry_attrs[predobjectpath] is not None:
                obj_content = entry_attrs[predobjectpath]['string']
            elif entry_attrs.get('properties', {}).get('map', {}).get('partial_path') is not None:
                obj_content = entry_attrs.get('properties', {}).get('map', {}).get('partial_path')
            elif entry_attrs.get('properties', {}).get('map', {}).get('address') is not None:
                obj_content = entry_attrs.get('properties', {}).get('map', {}).get('address')
            else:
                obj_content = 'unknown'
        elif obj_type == 'NetFlowObject':
            obj_content = obj_dict['remoteip']
        elif obj_type == 'UnnamedPipeObject':
            obj_content = 'UnnamedPipeObject'
        elif obj_type in self.subject_type:
            if entry_attrs['subject']['com.bbn.tc.schema.avro.cdm18.UUID'] == objectid:  # self-loop
                obj_content = entry_attrs['properties']['map']['exec']
            else:
                obj_content = 'unknown'
        else:
            raise ValueError(f'Invalid object type.{obj_type}')
        obj_id = objectid
        obj_malicious_flag = -1
        return obj_id, obj_type, obj_content, obj_malicious_flag
    
    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.Subject": {
                "uuid": "72FB0406-3678-11E8-BF66-D9AA8AFF4A69",
                "type": "SUBJECT_PROCESS",
                "cid": 787,
                "parentSubject": null,
                "hostId": "83C8ED1F-5045-DBCD-B39F-918F0DF4F851",
                "localPrincipal": "7DCA248E-1BBA-59F5-9227-B25D5F253594",
                "startTimestampNanos": 0,
                "unitId": null,
                "iteration": null,
                "count": null,
                "cmdLine": null,
                "privilegeLevel": null,
                "importedLibraries": null,
                "exportedLibraries": null,
                "properties": {
                    "map": {
                    "host": "83c8ed1f-5045-dbcd-b39f-918f0df4f851"
                    }
                }
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_FREEBSD_DTRACE_CADETS"
    }
    '''
    def parse_subject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['Subject']]
        entry_type = entry_attrs['type']
        self.subject_type.add(entry_type)
        nodeid = entry_attrs['uuid']
        res_dict = {
            'type': entry_type,
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
    "datum": {
        "com.bbn.tc.schema.avro.cdm18.FileObject": {
            "uuid": "0C0D74DB-3680-11E8-BF66-D9AA8AFF4A69",
            "baseObject": {
                "hostId": "83C8ED1F-5045-DBCD-B39F-918F0DF4F851",
                "permission": null,
                "epoch": null,
                "properties": {
                "map": {}
                }
            },
            "type": "FILE_OBJECT_UNIX_SOCKET",
            "fileDescriptor": null,
            "localPrincipal": null,
            "size": null,
            "peInfo": null,
            "hashes": null
        }
    },
    "CDMVersion": "18",
    "source": "SOURCE_FREEBSD_DTRACE_CADETS"
    }
    '''
    def parse_fileobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['FileObject']]
        entry_type = entry_attrs['type']
        if entry_type not in config.node_type['e3cadets']:
            return None
        self.file_type.add(entry_type)
        nodeid = entry_attrs['uuid']
        res_dict = {
            'type': entry_type,
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
    "datum": {
        "com.bbn.tc.schema.avro.cdm18.NetFlowObject": {
            "uuid": "71A3EC7D-3678-11E8-BF66-D9AA8AFF4A69",
            "baseObject": {
                "hostId": "83C8ED1F-5045-DBCD-B39F-918F0DF4F851",
                "permission": null,
                "epoch": null,
                "properties": {
                "map": {}
                }
            },
            "localAddress": "10.0.6.23",
            "localPort": 123,
            "remoteAddress": "10.0.4.1",
            "remotePort": 123,
            "ipProtocol": null,
            "fileDescriptor": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_FREEBSD_DTRACE_CADETS"
    }
    '''
    def parse_netflowobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['NetFlowObject']]
        entry_type = 'NetFlowObject'
        nodeid = entry_attrs['uuid']
        remoteaddr = entry_attrs['remoteAddress'].strip()  # only remote, ref.shadewatcher
        remoteport = str(entry_attrs['remotePort'])
        res_dict = {
            'type': entry_type,
            'remoteip': remoteaddr + ':' + remoteport
        }
        self.set(nodeid, res_dict)
        return None
    
    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject": {
            "uuid": "86A72F7F-78C2-5205-A7E7-481B173514C2",
            "baseObject": {
                "hostId": "83C8ED1F-5045-DBCD-B39F-918F0DF4F851",
                "permission": null,
                "epoch": null,
                "properties": {
                "map": {}
                }
            },
            "sourceFileDescriptor": null,
            "sinkFileDescriptor": null,
            "sourceUUID": {
                "com.bbn.tc.schema.avro.cdm18.UUID": "FB793908-36C3-11E8-BF66-D9AA8AFF4A69"
            },
            "sinkUUID": {
                "com.bbn.tc.schema.avro.cdm18.UUID": "FB793933-36C3-11E8-BF66-D9AA8AFF4A69"
            }
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_FREEBSD_DTRACE_CADETS"
    }
    '''
    def parse_unnamedPipeObject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['UnnamedPipeObject']]
        entry_type = 'UnnamedPipeObject'
        nodeid = entry_attrs['uuid']
        res_dict = {
            'type': entry_type,
        }
        self.set(nodeid, res_dict)
        return None


class DarpaE3TheiaParser(BasicParser):
    '''
    E3 Theia dataset parser.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_type = set()
        self.subject_type = set()
        # self.file1 = open('/data2/sj/DARPA_Engagement3/parsed_new/theia/unknown_entry.json', 'a')

    def parse_single_entry(self, entry: str) -> Optional[OPREventModel]:
        if config.DarpaE3_reserved_type_2_strings_map['Event'] in entry:
            return self.parse_event(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['Subject'] in entry:
            return self.parse_subject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['FileObject'] in entry:
            return self.parse_fileobject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['NetFlowObject'] in entry:
            return self.parse_netflowobject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['MemoryObject'] in entry:
            return self.parse_memoryobject(entry)
        else:
            # # //////////////////////////////////////////////////////////////
            # if 'cdm18.SrcSinkObject' not in entry and 'cdm18.Principal' not in entry:
            #     self.file1.write(entry + '\n')
            # # //////////////////////////////////////////////////////////////
            return None
    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.Event": {
                "uuid": "EBF0E78C-E9F1-2115-ED01-000000000010",
                "sequence": {
                    "long": 563
                },
                "type": "EVENT_READ",
                "threadId": {
                    "int": 2440
                },
                "hostId": "0A00063C-5254-00F0-0D60-000000000070",
                "subject": {
                    "com.bbn.tc.schema.avro.cdm18.UUID": "88094E00-0000-0000-0000-000000000020"
                },
                "predicateObject": {
                    "com.bbn.tc.schema.avro.cdm18.UUID": "0100D00F-D76F-1C00-0000-0000C1BD9A00"
                },
                "predicateObjectPath": null,
                "predicateObject2": {
                    "com.bbn.tc.schema.avro.cdm18.UUID": "00000000-0000-0000-0000-000000000000"
                },
                "predicateObject2Path": null,
                "timestampNanos": 1522764134421623019,
                "name": {
                    "string": "read"
                },
                "parameters": null,
                "location": null,
                "size": {
                    "long": 0
                },
                "programPoint": null,
                "properties": {
                    "map": {}
                }
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_LINUX_THEIA"
    }
    '''
    def parse_event(self, entry: str) -> Optional[OPREventModel]:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['Event']]
        timestamp = entry_attrs['timestampNanos']
        relation_type = entry_attrs['type']
        if relation_type not in config.edge_type['e3theia']:
            return None

        _subject = self.get_subject(entry_attrs)
        if _subject is None:
            return None
        sbj_id, sbj_type, sbj_content, sbj_malicious_flag = _subject

        _object = self.get_object(entry_attrs, 'predicateObject')
        if _object is None:
            return None
        obj_id, obj_type, obj_content, obj_malicious_flag = _object

        oprem1 = self.convert_to_OPREM(relation_type, sbj_id, sbj_type, sbj_content, sbj_malicious_flag, obj_id, obj_type, obj_content, obj_malicious_flag, timestamp)

        _object2 = self.get_object(entry_attrs, 'predicateObject2')
        if _object2 is None:
            return [oprem1]
        obj_id2, obj_type2, obj_content2, obj_malicious_flag2 = _object2

        oprem2 = self.convert_to_OPREM(relation_type, sbj_id, sbj_type, sbj_content, sbj_malicious_flag, obj_id2, obj_type2, obj_content2, obj_malicious_flag2, timestamp)
        return [oprem1, oprem2]

    def get_subject(self, entry_attrs):
        if entry_attrs['subject'] is None:  # skip unknown subject
            return None
        subjectid = entry_attrs['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        sbj_dict = self.get(subjectid)
        if sbj_dict is None:
            return None
        
        sbj_type = sbj_dict['type']
        sbj_content = sbj_dict['path']

        sbj_id = subjectid
        sbj_malicious_flag = -1
        return sbj_id, sbj_type, sbj_content, sbj_malicious_flag
    
    def get_object(self, entry_attrs, predobject='predicateObject'):  # 'predicateObject' or 'predicateObject2'
        if entry_attrs[predobject] is None:  # skip unknown object
            return None
        objectid = entry_attrs[predobject]['com.bbn.tc.schema.avro.cdm18.UUID']
        obj_dict = self.get(objectid)
        if obj_dict is None:  # skip if object is not in the cache
            return None
        
        obj_type = obj_dict['type']
        if obj_type in self.file_type:
            obj_content = obj_dict['path']
        elif obj_type == 'NetFlowObject':
            obj_content = obj_dict['remoteip']
        elif obj_type == 'MemoryObject':
            obj_content = 'MemoryObject'
        elif obj_type in self.subject_type:
            obj_content = obj_dict['path']
        else:
            raise ValueError(f'Invalid object type.{obj_type}')
        
        obj_id = objectid
        obj_malicious_flag = -1
        return obj_id, obj_type, obj_content, obj_malicious_flag

    '''
    {
    "datum": {
        "com.bbn.tc.schema.avro.cdm18.Subject": {
            "uuid": "02000000-0000-0000-0000-000000000020",
            "type": "SUBJECT_PROCESS",
            "cid": 2,
            "parentSubject": {
                "com.bbn.tc.schema.avro.cdm18.UUID": "0000FFFF-FFFF-0788-FFFF-000000000020"
            },
            "hostId": "0A00063C-5254-00F0-0D60-000000000070",
            "localPrincipal": "00000000-0000-0000-0000-000000000060",
            "startTimestampNanos": 1522764134345124727,
            "unitId": null,
            "iteration": null,
            "count": null,
            "cmdLine": {
                "string": "N/A"
            },
            "privilegeLevel": null,
            "importedLibraries": null,
            "exportedLibraries": null,
            "properties": {
                "map": {
                "tgid": "2",
                "path": "kthreadd",
                "ppid": "0"
                }
            }
        }
    },
    "CDMVersion": "18",
    "source": "SOURCE_LINUX_THEIA"
    }
    '''
    def parse_subject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['Subject']]
        entry_type = entry_attrs['type']
        self.subject_type.add(entry_type)
        nodeid = entry_attrs['uuid']
        if nodeid == '00000000-0000-0000-0000-000000000000':  # ref. MAGIC
            return None

        try:  # some subjects don't have path
            image_path = entry_attrs['properties']['map']['path']
        except:
            cmdline = entry_attrs['cmdLine']['string'].strip()
            image_path = find_path(cmdline)
        if not image_path:  # if subpath is None
            image_path = 'unknown'

        res_dict = {
            'type': entry_type,
            'path': image_path,
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
    "datum": {
        "com.bbn.tc.schema.avro.cdm18.FileObject": {
            "uuid": "0100D00F-BF1D-1C00-0000-0000C0DD3B08",
            "baseObject": {
                "hostId": "0A00063C-5254-00F0-0D60-000000000070",
                "permission": null,
                "epoch": null,
                "properties": {
                    "map": {
                        "dev": "265289729",
                        "inode": "1842623",
                        "filename": "/usr/lib/python2.7/glob.py"
                    }
                }
            },
            "type": "FILE_OBJECT_BLOCK",
            "fileDescriptor": null,
            "localPrincipal": {
                "com.bbn.tc.schema.avro.cdm18.UUID": "00000000-0000-0000-0000-000000000060"
            },
            "size": null,
            "peInfo": null,
            "hashes": null
            }
        },
    "CDMVersion": "18",
    "source": "SOURCE_LINUX_THEIA"
    }
    '''
    def parse_fileobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['FileObject']]
        entry_type = entry_attrs['type']
        if entry_type not in config.node_type['e3theia']:
            return None
        self.file_type.add(entry_type)
        nodeid = entry_attrs['uuid']

        try:
            filepath = entry_attrs['baseObject']['properties']['map']['filename'].strip()
        except:
            filepath = 'unknown'

        
        res_dict = {
            'type': entry_type,
            'path': filepath
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.NetFlowObject": {
                "uuid": "FEFFFFFF-0000-FFFF-FFFF-000000000040",
                "baseObject": {
                    "hostId": "0A00063C-5254-00F0-0D60-000000000070",
                    "permission": null,
                    "epoch": null,
                    "properties": null
                },
                "localAddress": "LOCAL",
                "localPort": 0,
                "remoteAddress": "NA",
                "remotePort": 0,
                "ipProtocol": null,
                "fileDescriptor": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_LINUX_THEIA"
    }
    '''
    def parse_netflowobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['NetFlowObject']]
        entry_type = 'NetFlowObject'
        nodeid = entry_attrs['uuid']

        remoteaddr = entry_attrs['remoteAddress'].strip()  # only remote, ref.shadewatcher
        remoteport = str(entry_attrs['remotePort'])

        res_dict = {
            'type': entry_type,
            'remoteip': remoteaddr + ':' + remoteport
        }
        self.set(nodeid, res_dict)
        return None
    
    '''
    {
    "datum": {
            "com.bbn.tc.schema.avro.cdm18.MemoryObject": {
                "uuid": "0F0D0050-6200-0000-0000-000000000050",
                "baseObject": {
                    "hostId": "0A00063C-5254-00F0-0D60-000000000070",
                    "permission": null,
                    "epoch": null,
                    "properties": {
                        "map": {}
                    }
                },
                "memoryAddress": 6443008,
                "pageNumber": null,
                "pageOffset": null,
                "size": {
                    "long": 4096
                }
            }
        },
    "CDMVersion": "18",
    "source": "SOURCE_LINUX_THEIA"
    }
    '''
    def parse_memoryobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['MemoryObject']]
        entry_type = 'MemoryObject'
        nodeid = entry_attrs['uuid']
        res_dict = {
            'type': entry_type,
        }
        self.set(nodeid, res_dict)
        return None


class DarpaE3TraceParser(BasicParser):
    '''
    E3 Trace dataset parser.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_type = set()
        self.subject_type = set()
        # self.file1 = open('/data2/sj/DARPA_Engagement3/parsed_new/trace/unknown_entry.json', 'a')

    def parse_single_entry(self, entry: str) -> Optional[Tuple[str, ...]]:
        if config.DarpaE3_reserved_type_2_strings_map['Event'] in entry:
            return self.parse_event(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['Subject'] in entry:
            return self.parse_subject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['FileObject'] in entry:
            return self.parse_fileobject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['NetFlowObject'] in entry:
            return self.parse_netflowobject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['UnnamedPipeObject'] in entry:
            return self.parse_unnamedPipeObject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['MemoryObject'] in entry:
            return self.parse_memoryobject(entry)
        else:
            # # //////////////////////////////////////////////////////////////
            # if 'cdm18.SrcSinkObject' not in entry and 'cdm18.Principal' not in entry and 'cdm18.UnitDependency' not in entry:
            #     self.file1.write(entry + '\n')
            # # //////////////////////////////////////////////////////////////
            return None

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.Event": {
                "uuid": "BB77C342-8C7B-DA7F-5DF6-96D53B15BC4A",
                "sequence": {
                    "long": 2283026
                },
                "type": "EVENT_CLOSE",
                "threadId": {
                    "int": 17294
                },
                "hostId": "3120B2A9-057E-E4EB-4BB9-154983D2C063",
                "subject": {
                    "com.bbn.tc.schema.avro.cdm18.UUID": "7E10E90B-0E77-93E9-587A-48980BEE29A3"
                },
                "predicateObject": {
                    "com.bbn.tc.schema.avro.cdm18.UUID": "17972A5B-F14D-1101-36F0-413C2D11C62E"
                },
                "predicateObjectPath": null,
                "predicateObject2": null,
                "predicateObject2Path": null,
                "timestampNanos": 1522707421507000000,
                "name": null,
                "parameters": null,
                "location": null,
                "size": null,
                "programPoint": null,
                "properties": {
                    "map": {
                        "opm": "Used"
                    }
                }
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_LINUX_SYSCALL_TRACE"
    }
    '''
    def parse_event(self, entry: str) -> Tuple:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['Event']]
        timestamp = entry_attrs['timestampNanos']
        relation_type = entry_attrs['type']
        if relation_type not in config.edge_type['e3trace']:
            return None
        
        _subject = self.get_subject(entry_attrs)
        if _subject is None:
            return None
        sbj_id, sbj_type, sbj_content, sbj_malicious_flag = _subject

        _object = self.get_object(entry_attrs, 'predicateObject')
        if _object is None:
            return None
        obj_id, obj_type, obj_content, obj_malicious_flag = _object

        oprem1 = self.convert_to_OPREM(relation_type, sbj_id, sbj_type, sbj_content, sbj_malicious_flag, obj_id, obj_type, obj_content, obj_malicious_flag, timestamp)

        _object2 = self.get_object(entry_attrs, 'predicateObject2')
        if _object2 is None:
            return [oprem1]
        obj_id2, obj_type2, obj_content2, obj_malicious_flag2 = _object2

        oprem2 = self.convert_to_OPREM(relation_type, sbj_id, sbj_type, sbj_content, sbj_malicious_flag, obj_id2, obj_type2, obj_content2, obj_malicious_flag2, timestamp)
        return [oprem1, oprem2]

    def get_subject(self, entry_attrs):
        if entry_attrs['subject'] is None:  # skip unknown subject
            return None
        subjectid = entry_attrs['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        sbj_dict = self.get(subjectid)
        if sbj_dict is None:
            return None
        
        sbj_type = sbj_dict['type']
        sbj_content = sbj_dict['path']

        sbj_id = subjectid
        sbj_malicious_flag = -1
        return sbj_id, sbj_type, sbj_content, sbj_malicious_flag
    
    def get_object(self, entry_attrs, predobject='predicateObject'):  # 'predicateObject' or 'predicateObject2'
        if entry_attrs[predobject] is None:  # skip unknown object
            return None
        objectid = entry_attrs[predobject]['com.bbn.tc.schema.avro.cdm18.UUID']
        obj_dict = self.get(objectid)
        if obj_dict is None:  # skip if object is not in the cache
            return None
        
        obj_type = obj_dict['type']
        if obj_type in self.file_type:
            obj_content = obj_dict['path']
        elif obj_type == 'NetFlowObject':
            obj_content = obj_dict['remoteip']
        elif obj_type == 'MemoryObject':
            obj_content = 'MemoryObject'
        elif obj_type == 'UnnamedPipeObject':
            obj_content = 'UnnamedPipeObject'
        elif obj_type in self.subject_type:
            obj_content = obj_dict['path']
        else:
            raise ValueError(f'Invalid object type.{obj_type}')
        
        obj_id = objectid
        obj_malicious_flag = -1
        return obj_id, obj_type, obj_content, obj_malicious_flag
    

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.Subject": {
                "uuid": "6E9241E8-9D0E-452E-617F-1548D39D5D2F",
                "type": "SUBJECT_PROCESS",
                "cid": 15466,
                "parentSubject": {
                    "com.bbn.tc.schema.avro.cdm18.UUID": "99464F86-590F-8FA7-2544-D359B8F931BD"
                },
                "hostId": "3120B2A9-057E-E4EB-4BB9-154983D2C063",
                "localPrincipal": "29895546-B124-1BEC-E91C-C9107B81C616",
                "startTimestampNanos": 1522703821498000000,
                "unitId": {
                    "int": 0
                },
                "iteration": null,
                "count": null,
                "cmdLine": {
                    "string": "/bin/sh -c 2020206364202F2026262072756E2D7061727473202D2D7265706F7274202F6574632F63726F6E2E686F75726C79"
                },
                "privilegeLevel": null,
                "importedLibraries": null,
                "exportedLibraries": null,
                "properties": {
                    "map": {
                        "name": "sh",
                        "cwd": "/root",
                        "ppid": "15465"
                    }
                }
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_LINUX_SYSCALL_TRACE"
    }
    '''
    def parse_subject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['Subject']]
        entry_type = entry_attrs['type']
        self.subject_type.add(entry_type)
        nodeid = entry_attrs['uuid']
        if nodeid == '00000000-0000-0000-0000-000000000000':  # ref. MAGIC
            return None
        
        try:  # some subjects don't have path
            image_path = entry_attrs['properties']['map']['name']
        except:
            cmdline = entry_attrs['cmdLine']['string'].strip()
            image_path = find_path(cmdline)
        if not image_path:  # if subpath is None
            image_path = 'unknown'

        res_dict = {
            'type': entry_type,
            'path': image_path,
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.FileObject": {
                "uuid": "49463062-60DC-4F2A-39DD-1020749C0642",
                "baseObject": {
                    "hostId": "3120B2A9-057E-E4EB-4BB9-154983D2C063",
                    "permission": {
                        "com.bbn.tc.schema.avro.cdm18.SHORT": "01ED"
                    },
                    "epoch": {
                        "int": 0
                    },
                    "properties": {
                        "map": {
                            "path": "/lib64/ld-linux-x86-64.so.2"
                        }
                    }
                },
                "type": "FILE_OBJECT_FILE",
                "fileDescriptor": null,
                "localPrincipal": null,
                "size": null,
                "peInfo": null,
                "hashes": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_LINUX_SYSCALL_TRACE"
    }
    '''
    def parse_fileobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['FileObject']]
        entry_type = entry_attrs['type']
        if entry_type not in config.node_type['e3trace']:
            return None
        self.file_type.add(entry_type) 
        nodeid = entry_attrs['uuid']

        try:
            filepath = entry_attrs['baseObject']['properties']['map']['path'].strip()
        except:
            filepath = 'unknown'
        
        res_dict = {
            'type': entry_type,
            'path': filepath
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.NetFlowObject": {
                "uuid": "87612FC3-9C41-7BDC-6FF0-75AC0E6C17AF",
                "baseObject": {
                    "hostId": "3120B2A9-057E-E4EB-4BB9-154983D2C063",
                    "permission": null,
                    "epoch": {
                        "int": 168
                    },
                    "properties": {
                        "map": {}
                    }
                },
                "localAddress": "0.0.0.0",
                "localPort": 5353,
                "remoteAddress": "128.55.12.171",
                "remotePort": 5353,
                "ipProtocol": {
                "int": 17
                },
                "fileDescriptor": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_LINUX_SYSCALL_TRACE"
    }
    '''
    def parse_netflowobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['NetFlowObject']]
        entry_type = 'NetFlowObject'
        nodeid = entry_attrs['uuid']

        remoteaddr = entry_attrs['remoteAddress'].strip()  # only remote, ref.shadewatcher
        remoteport = str(entry_attrs['remotePort'])

        res_dict = {
            'type': entry_type,
            'remoteip': remoteaddr + ':' + remoteport
        }
        self.set(nodeid, res_dict)
        return None
    
    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject": {
                "uuid": "17972A5B-F14D-1101-36F0-413C2D11C62E",
                "baseObject": {
                    "hostId": "3120B2A9-057E-E4EB-4BB9-154983D2C063",
                    "permission": null,
                    "epoch": {
                        "int": 0
                    },
                    "properties": {
                        "map": {
                        "pid": "17294"
                        }
                    }
                },
                "sourceFileDescriptor": {
                    "int": 3
                },
                "sinkFileDescriptor": {
                    "int": 4
                },
                "sourceUUID": null,
                "sinkUUID": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_LINUX_SYSCALL_TRACE"
    }
    '''
    def parse_unnamedPipeObject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['UnnamedPipeObject']]
        entry_type = 'UnnamedPipeObject'
        nodeid = entry_attrs['uuid']
        res_dict = {
            'type': entry_type,
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.MemoryObject": {
                "uuid": "D7D64766-68C2-4E05-B96B-60001F3B2767",
                "baseObject": {
                    "hostId": "3120B2A9-057E-E4EB-4BB9-154983D2C063",
                    "permission": null,
                    "epoch": null,
                    "properties": {
                        "map": {
                        "tgid": "15559"
                        }
                    }
                },
                "memoryAddress": 139645949095936,
                "pageNumber": null,
                "pageOffset": null,
                "size": {
                    "long": 95866
                }
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_LINUX_SYSCALL_TRACE"
    }
    '''
    def parse_memoryobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['MemoryObject']]
        entry_type = 'MemoryObject'
        nodeid = entry_attrs['uuid']
        res_dict = {
            'type': entry_type,
        }
        self.set(nodeid, res_dict)
        return None

class DarpaE3FivedirectionsParser(BasicParser):
    '''
    E3 Fivedirections dataset parser.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_type = set()
        self.subject_type = set()
        # self.file1 = open('/data2/sj/DARPA_Engagement3/parsed_new/fivedirections/unknown_entry.json', 'a')

    def parse_single_entry(self, entry: str) -> Optional[OPREventModel]:
        if config.DarpaE3_reserved_type_2_strings_map['Event'] in entry:
            return self.parse_event(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['Subject'] in entry:
            return self.parse_subject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['FileObject'] in entry:
            return self.parse_fileobject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['NetFlowObject'] in entry:
            return self.parse_netflowobject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['MemoryObject'] in entry:
            return self.parse_memoryobject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['RegistryKeyObject'] in entry:
            return self.parse_registrykeyobject(entry)
        else:
            # # //////////////////////////////////////////////////////////////
            # if 'cdm18.SrcSinkObject' not in entry and 'cdm18.Principal' not in entry and 'cdm18.StartMarker' not in entry and 'cdm18.TimeMarker' not in entry:
            #     self.file1.write(entry + '\n')
            # # //////////////////////////////////////////////////////////////
            return None
        
    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.Event": {
                "uuid": "671E83F0-A4DE-4C63-9279-DB122CF9EDF1",
                "sequence": {
                    "long": 0
                },
                "type": "EVENT_READ",
                "threadId": {
                    "int": 0
                },
                "hostId": "D2842312-8456-4C8E-ADD6-E2295D0939D3",
                "subject": {
                    "com.bbn.tc.schema.avro.cdm18.UUID": "0E0936EB-026E-4AF3-832E-F75C2D85F99F"
                },
                "predicateObject": {
                    "com.bbn.tc.schema.avro.cdm18.UUID": "F2ACAB25-82AC-4036-B6AA-18B01BD929C4"
                },
                "predicateObjectPath": null,
                "predicateObject2": null,
                "predicateObject2Path": null,
                "timestampNanos": 1522850025376000000,
                "name": {
                    "string": "EVENT_READ"
                },
                "parameters": null,
                "location": {
                    "long": 0
                },
                "size": {
                    "long": 0
                },
                "programPoint": null,
                "properties": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_WINDOWS_FIVEDIRECTIONS"
    }
    '''
    def parse_event(self, entry: str) -> Optional[OPREventModel]:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['Event']]
        timestamp = entry_attrs['timestampNanos']
        relation_type = entry_attrs['type']
        if relation_type not in config.edge_type['e3fivedirections']:
            return None

        _subject = self.get_subject(entry_attrs)
        if _subject is None:
            return None
        sbj_id, sbj_type, sbj_content, sbj_malicious_flag = _subject

        _object = self.get_object(entry_attrs, 'predicateObject', 'predicateObjectPath')
        if _object is None:
            return None
        obj_id, obj_type, obj_content, obj_malicious_flag = _object

        oprem1 = self.convert_to_OPREM(relation_type, sbj_id, sbj_type, sbj_content, sbj_malicious_flag, obj_id, obj_type, obj_content, obj_malicious_flag, timestamp)

        _object2 = self.get_object(entry_attrs, 'predicateObject2', 'predicateObject2Path')
        if _object2 is None:
            return [oprem1]
        obj_id2, obj_type2, obj_content2, obj_malicious_flag2 = _object2

        oprem2 = self.convert_to_OPREM(relation_type, sbj_id, sbj_type, sbj_content, sbj_malicious_flag, obj_id2, obj_type2, obj_content2, obj_malicious_flag2, timestamp)
        return [oprem1, oprem2]

    def get_subject(self, entry_attrs):
        if entry_attrs['subject'] is None:  # skip unknown subject
            return None
        subjectid = entry_attrs['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        sbj_dict = self.get(subjectid)
        if sbj_dict is None:
            return None
        
        sbj_type = sbj_dict['type']
        sbj_content = sbj_dict['path']

        sbj_id = subjectid
        sbj_malicious_flag = -1
        return sbj_id, sbj_type, sbj_content, sbj_malicious_flag
    
    def get_object(self, entry_attrs, predobject='predicateObject', predobjectpath='predicateObjectPath'):  # 'predicateObject' or 'predicateObject2'
        if entry_attrs[predobject] is None:  # skip unknown object
            return None
        objectid = entry_attrs[predobject]['com.bbn.tc.schema.avro.cdm18.UUID']
        obj_dict = self.get(objectid)
        if obj_dict is None:  # skip if object is not in the cache
            return None
        
        obj_type = obj_dict['type']
        if obj_type in self.file_type:
            try:
                obj_content = entry_attrs[predobjectpath]['string'].replace('\\', '/').strip('"').lower()
                parts = obj_content.split("/", 3)
                obj_content = parts[3] if len(parts) > 3 else obj_content
            except:
                obj_content = 'unknown'
        elif obj_type == 'NetFlowObject':
            obj_content = obj_dict['remoteip']
        elif obj_type == 'MemoryObject':
            obj_content = 'MemoryObject'
        elif obj_type == 'RegistryKeyObject':
            obj_content = obj_dict['key']
        elif obj_type in self.subject_type:
            obj_content = obj_dict['path']
        else:
            raise ValueError(f'Invalid object type.{obj_type}')
        
        obj_id = objectid
        obj_malicious_flag = -1
        return obj_id, obj_type, obj_content, obj_malicious_flag
    
    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.Subject": {
                "uuid": "29C5B1F2-BE34-4D23-A118-19BDB02413E6",
                "type": "SUBJECT_PROCESS",
                "cid": 4804,
                "parentSubject": null,
                "hostId": "D2842312-8456-4C8E-ADD6-E2295D0939D3",
                "localPrincipal": "00000000-0000-0000-0000-000000000000",
                "startTimestampNanos": 1522850023313000000,
                "unitId": null,
                "iteration": null,
                "count": null,
                "cmdLine": {
                    "string": "explorer.exe"
                },
                "privilegeLevel": null,
                "importedLibraries": null,
                "exportedLibraries": null,
                "properties": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_WINDOWS_FIVEDIRECTIONS"
    }
    ''' 
    def parse_subject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['Subject']]
        entry_type = entry_attrs['type']
        self.subject_type.add(entry_type)
        nodeid = entry_attrs['uuid']

        try:
            cmdline = entry_attrs['cmdLine']['string']
            if '.exe' in cmdline:
                cmdline = cmdline.split('.exe')[0] + '.exe'
            image_path = cmdline.replace('\\', '/').strip('"').lower()
            image_path = image_path.rstrip("/").split("/")[-1]
        except:
            try:
                parentSubjectId = entry_attrs['parentSubject']['com.bbn.tc.schema.avro.cdm18.UUID']
                parent_subject_path = self.get(parentSubjectId)['path']
                if parent_subject_path is not None:
                    image_path = parent_subject_path
                else:
                    image_path = 'unknown'
            except:
                image_path = 'unknown'
                
        res_dict = {
            'type': entry_type,
            'path': image_path,
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.FileObject": {
                "uuid": "2773F8B1-60E4-4FC2-8402-91B7BDDBA502",
                "baseObject": {
                    "hostId": "D2842312-8456-4C8E-ADD6-E2295D0939D3",
                    "permission": null,
                    "epoch": null,
                    "properties": null
                },
                "type": "FILE_OBJECT_PEFILE",
                "fileDescriptor": null,
                "localPrincipal": null,
                "size": null,
                "peInfo": null,
                "hashes": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_WINDOWS_FIVEDIRECTIONS"
    }
    '''
    def parse_fileobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['FileObject']]
        entry_type = entry_attrs['type']
        if entry_type not in config.node_type['e3fivedirections']:
            return None
        self.file_type.add(entry_type)
        nodeid = entry_attrs['uuid']

        res_dict = {
            'type': entry_type,
        }
        self.set(nodeid, res_dict)
        return None
    
    '''
    {
        "datum": {
        "com.bbn.tc.schema.avro.cdm18.NetFlowObject": {
            "uuid": "40AE5205-F7CD-4334-9554-AF77439608CD",
            "baseObject": {
                "hostId": "D2842312-8456-4C8E-ADD6-E2295D0939D3",
                "permission": null,
                "epoch": null,
                "properties": null
            },
            "localAddress": "::",
            "localPort": 50237,
            "remoteAddress": "",
            "remotePort": 0,
            "ipProtocol": {
                "int": 17
            },
            "fileDescriptor": null
        }
        },
        "CDMVersion": "18",
        "source": "SOURCE_WINDOWS_FIVEDIRECTIONS"
    }
    '''
    def parse_netflowobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['NetFlowObject']]
        entry_type = 'NetFlowObject'
        nodeid = entry_attrs['uuid']

        remoteaddr = entry_attrs['remoteAddress'].strip()  # only remote, ref.shadewatcher
        remoteport = str(entry_attrs['remotePort'])

        res_dict = {
            'type': entry_type,
            'remoteip': remoteaddr + ':' + remoteport
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.RegistryKeyObject": {
                "uuid": "9FA8BD5E-BC2F-4EAE-BA14-737BC90BDF7B",
                "baseObject": {
                    "hostId": "D2842312-8456-4C8E-ADD6-E2295D0939D3",
                    "permission": null,
                    "epoch": null,
                    "properties": null
                },
                "key": "\\REGISTRY\\MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\LanguagePack\\DataStore_V1.0",
                "value": {
                    "com.bbn.tc.schema.avro.cdm18.Value": {
                        "size": -1,
                        "type": "VALUE_TYPE_SRC",
                        "valueDataType": "VALUE_DATA_TYPE_BYTE",
                        "isNull": true,
                        "name": null,
                        "runtimeDataType": null,
                        "valueBytes": null,
                        "provenance": null,
                        "tag": null,
                        "components": null
                    }
                },
                "size": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_WINDOWS_FIVEDIRECTIONS"
    }
    '''
    def parse_registrykeyobject(self, entry:str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['RegistryKeyObject']]
        entry_type = 'RegistryKeyObject'
        nodeid = entry_attrs['uuid']
        
        key = entry_attrs['key'].replace('\\', '/').strip('"').lower()
        res_dict = {
            'type': entry_type,
            'key': key
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.MemoryObject": {
                "uuid": "B3C6582B-CDF7-40EE-9E33-7E519B9A5531",
                "baseObject": {
                "hostId": "D2842312-8456-4C8E-ADD6-E2295D0939D3",
                "permission": null,
                "epoch": null,
                "properties": {
                    "map": {
                        "flags": "4096"
                    }
                }
                },
                "memoryAddress": 2090514571264,
                "pageNumber": null,
                "pageOffset": null,
                "size": {
                "long": 36864
                }
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_WINDOWS_FIVEDIRECTIONS"
    }
    '''
    def parse_memoryobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['MemoryObject']]
        entry_type = 'MemoryObject'
        nodeid = entry_attrs['uuid']
        res_dict = {
            'type': entry_type,
        }
        self.set(nodeid, res_dict)
        return None


class DarpaE3ClearscopeParser(BasicParser):
    '''
    E3 Clearscope dataset parser.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_type = set()
        self.subject_type = set()
        # self.file1 = open('/data2/sj/DARPA_Engagement3/parsed_new/clearscope/unknown_entry.json', 'a')

    def parse_single_entry(self, entry: str) -> Optional[OPREventModel]:
        if config.DarpaE3_reserved_type_2_strings_map['Event'] in entry:
            return self.parse_event(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['Subject'] in entry:
            return self.parse_subject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['FileObject'] in entry:
            return self.parse_fileobject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['NetFlowObject'] in entry:
            return self.parse_netflowobject(entry)
        elif config.DarpaE3_reserved_type_2_strings_map['UnnamedPipeObject'] in entry:
            return self.parse_unnamedPipeObject(entry)
        else:
            # # //////////////////////////////////////////////////////////////
            # if 'cdm18.SrcSinkObject' not in entry and 'cdm18.Principal' not in entry and 'cdm18.ProvenanceTagNode' not in entry:
            #     self.file1.write(entry + '\n')
            # # //////////////////////////////////////////////////////////////
            return None

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.Event": {
                "uuid": "BA08DD6A-89B7-6AB2-91B8-A0942E498BD4",
                "sequence": {
                    "long": 50
                },
                "type": "EVENT_OPEN",
                "threadId": {
                    "int": 1192
                },
                "hostId": "5957F7A8-2EAB-D99C-459A-408A1F427D29",
                "subject": {
                    "com.bbn.tc.schema.avro.cdm18.UUID": "00000000-0000-0000-0000-000000000002"
                },
                "predicateObject": {
                    "com.bbn.tc.schema.avro.cdm18.UUID": "FA97321C-EA08-3210-9A6E-BAC60DDC7806"
                },
                "predicateObjectPath": {
                    "string": "/system/app/webview/webview.apk"
                },
                "predicateObject2": null,
                "predicateObject2Path": null,
                "timestampNanos": 2241942260000000,
                "name": {
                    "string": "java.io.FileDescriptor libcore.io.Posix.open(java.lang.String path, int flags, int mode) [line: 204]"
                },
                "parameters": {
                "array": [
                    {
                    "size": -1,
                    "type": "VALUE_TYPE_CONTROL",
                    "valueDataType": "VALUE_DATA_TYPE_CHAR",
                    "isNull": false,
                    "name": {
                        "string": "path"
                    },
                    "runtimeDataType": null,
                    "valueBytes": {
                        "bytes": "0000002F00000073000000790000007300000074000000650000006D0000002F0000006100000070000000700000002F000000770000006500000062000000760000006900000065000000770000002F000000770000006500000062000000760000006900000065000000770000002E00000061000000700000006B"
                    },
                    "provenance": null,
                    "tag": {
                        "array": [
                        {
                            "numValueElements": 12,
                            "tagId": "00000000-0000-0000-0000-000000000000"
                        },
                        {
                            "numValueElements": 7,
                            "tagId": "00000000-0000-0000-0000-00000000078F"
                        },
                        {
                            "numValueElements": 1,
                            "tagId": "00000000-0000-0000-0000-000000000000"
                        },
                        {
                            "numValueElements": 11,
                            "tagId": "00000000-0000-0000-0000-0000000008AE"
                        }
                        ]
                    },
                    "components": null
                    },
                    {
                    "size": -1,
                    "type": "VALUE_TYPE_CONTROL",
                    "valueDataType": "VALUE_DATA_TYPE_INT",
                    "isNull": false,
                    "name": {
                        "string": "flags"
                    },
                    "runtimeDataType": null,
                    "valueBytes": {
                        "bytes": "00000000"
                    },
                    "provenance": null,
                    "tag": {
                        "array": [
                        {
                            "numValueElements": -1,
                            "tagId": "00000000-0000-0000-0000-000000000000"
                        }
                        ]
                    },
                    "components": null
                    },
                    {
                    "size": -1,
                    "type": "VALUE_TYPE_CONTROL",
                    "valueDataType": "VALUE_DATA_TYPE_INT",
                    "isNull": false,
                    "name": {
                        "string": "mode"
                    },
                    "runtimeDataType": null,
                    "valueBytes": {
                        "bytes": "00000000"
                    },
                    "provenance": null,
                    "tag": {
                        "array": [
                        {
                            "numValueElements": -1,
                            "tagId": "00000000-0000-0000-0000-000000000000"
                        }
                        ]
                    },
                    "components": null
                    },
                    {
                    "size": -1,
                    "type": "VALUE_TYPE_SRC",
                    "valueDataType": "VALUE_DATA_TYPE_COMPLEX",
                    "isNull": false,
                    "name": {
                        "string": "r"
                    },
                    "runtimeDataType": {
                        "string": "java.io.FileDescriptor"
                    },
                    "valueBytes": null,
                    "provenance": null,
                    "tag": null,
                    "components": {
                        "array": [
                        {
                            "size": -1,
                            "type": "VALUE_TYPE_SRC",
                            "valueDataType": "VALUE_DATA_TYPE_INT",
                            "isNull": false,
                            "name": {
                            "string": "descriptor"
                            },
                            "runtimeDataType": null,
                            "valueBytes": {
                            "bytes": "0000009C"
                            },
                            "provenance": null,
                            "tag": {
                            "array": [
                                {
                                "numValueElements": -1,
                                "tagId": "00000000-0000-0000-0000-000000000000"
                                }
                            ]
                            },
                            "components": null
                        }
                        ]
                    }
                    }
                ]
                },
                "location": null,
                "size": null,
                "programPoint": {
                "string": "boolean com.android.providers.settings.SettingsProvider$SettingsRegistry.isSystemSettingsKey(int) [line: 1777]"
                },
                "properties": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_ANDROID_JAVA_CLEARSCOPE"
    }
    '''
    def parse_event(self, entry: str) -> Optional[OPREventModel]:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['Event']]
        timestamp = entry_attrs['timestampNanos']
        relation_type = entry_attrs['type']
        if relation_type not in config.edge_type['e3clearscope']:
            return None

        _subject = self.get_subject(entry_attrs)
        if _subject is None:
            return None
        sbj_id, sbj_type, sbj_content, sbj_malicious_flag = _subject
        
        _object = self.get_object(entry_attrs, 'predicateObject', 'predicateObjectPath')
        if _object is None:
            return None
        obj_id, obj_type, obj_content, obj_malicious_flag = _object

        oprem1 = self.convert_to_OPREM(relation_type, sbj_id, sbj_type, sbj_content, sbj_malicious_flag, obj_id, obj_type, obj_content, obj_malicious_flag, timestamp)

        _object2 = self.get_object(entry_attrs, 'predicateObject2', 'predicateObject2Path')
        if _object2 is None:
            return [oprem1]
        obj_id2, obj_type2, obj_content2, obj_malicious_flag2 = _object2

        oprem2 = self.convert_to_OPREM(relation_type, sbj_id, sbj_type, sbj_content, sbj_malicious_flag, obj_id2, obj_type2, obj_content2, obj_malicious_flag2, timestamp)
        return [oprem1, oprem2]
    
    def get_subject(self, entry_attrs):
        if entry_attrs['subject'] is None:  # skip unknown subject
            return None
        subjectid = entry_attrs['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        sbj_dict = self.get(subjectid)
        if sbj_dict is None:
            return None
        
        sbj_type = sbj_dict['type']
        sbj_content = sbj_dict['path']

        sbj_id = subjectid
        sbj_malicious_flag = -1
        return sbj_id, sbj_type, sbj_content, sbj_malicious_flag
    
    def get_object(self, entry_attrs, predobject='predicateObject', predobjectpath='predicateObjectPath'):  # 'predicateObject' or 'predicateObject2'
        if entry_attrs[predobject] is None:  # skip unknown object
            return None
        objectid = entry_attrs[predobject]['com.bbn.tc.schema.avro.cdm18.UUID']
        obj_dict = self.get(objectid)
        if obj_dict is None:  # skip if object is not in the cache
            return None
        
        obj_type = obj_dict['type']
        if obj_type in self.file_type:
            obj_content = obj_dict['path']
        elif obj_type == 'NetFlowObject':
            obj_content = obj_dict['remoteip']
        elif obj_type == 'UnnamedPipeObject':
            obj_content = 'UnnamedPipeObject'
        elif obj_type in self.subject_type:
            obj_content = obj_dict['path']
        else:
            raise ValueError(f'Invalid object type.{obj_type}')
        
        if entry_attrs[predobjectpath] is not None:
            obj_content = entry_attrs[predobjectpath]['string']

        obj_id = objectid
        obj_malicious_flag = -1
        return obj_id, obj_type, obj_content, obj_malicious_flag
    
    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.Subject": {
                "uuid": "00000000-0000-0000-0000-000000000BBA",
                "type": "SUBJECT_PROCESS",
                "cid": 1915,
                "parentSubject": {
                    "com.bbn.tc.schema.avro.cdm18.UUID": "00000000-0000-0000-0000-000000000000"
                },
                "hostId": "5957F7A8-2EAB-D99C-459A-408A1F427D29",
                "localPrincipal": "71AF708F-8A9F-2889-1565-66F0BF1D4685",
                "startTimestampNanos": 2241931605000000,
                "unitId": null,
                "iteration": null,
                "count": null,
                "cmdLine": {
                    "string": "com.android.providers.calendar"
                },
                "privilegeLevel": null,
                "importedLibraries": null,
                "exportedLibraries": null,
                "properties": {
                    "map": {}
                }
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_ANDROID_JAVA_CLEARSCOPE"
    }
    '''
    def parse_subject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['Subject']]
        entry_type = entry_attrs['type']
        self.subject_type.add(entry_type) 
        nodeid = entry_attrs['uuid']

        # all nodes' properties are none,so use the cmdline 
        try:  # some subjects don't have path
            image_path = entry_attrs['cmdLine']['string']
        except:
            image_path = 'unknown'

        res_dict = {
            'type': entry_type,
            'path': image_path,
        }
        self.set(nodeid, res_dict)
        return None
        
    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.FileObject": {
                "uuid": "CF3272CB-436F-53EB-CB1B-463FD0414CD5",
                "baseObject": {
                    "hostId": "5957F7A8-2EAB-D99C-459A-408A1F427D29",
                    "permission": {
                        "com.bbn.tc.schema.avro.cdm18.SHORT": "0124"
                    },
                    "epoch": null,
                    "properties": {
                        "map": {
                            "path": "/proc/1915/cmdline"
                        }
                    }
                },
                "type": "FILE_OBJECT_FILE",
                "fileDescriptor": null,
                "localPrincipal": null,
                "size": {
                    "long": 0
                },
                "peInfo": null,
                "hashes": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_ANDROID_JAVA_CLEARSCOPE"
    }
    '''
    def parse_fileobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['FileObject']]
        entry_type = entry_attrs['type']
        if entry_type not in config.node_type['e3clearscope']:
            return None
        self.file_type.add(entry_type)
        nodeid = entry_attrs['uuid']

        try:
            filepath = entry_attrs['baseObject']['properties']['map']['path'].strip()
        except:
            filepath = 'unknown'

        res_dict = {
            'type': entry_type,
            'path': filepath
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.NetFlowObject": {
                "uuid": "00000000-0000-0000-0000-000000000E89",
                "baseObject": {
                    "hostId": "5957F7A8-2EAB-D99C-459A-408A1F427D29",
                    "permission": null,
                    "epoch": null,
                    "properties": null
                },
                "localAddress": "<UNNAMED fd:157>",
                "localPort": -1,
                "remoteAddress": "<UNNAMED fd:157>",
                "remotePort": -1,
                "ipProtocol": {
                    "int": -1
                },
                "fileDescriptor": null
            }
            },
            "CDMVersion": "18",
            "source": "SOURCE_ANDROID_JAVA_CLEARSCOPE"
        }
    '''
    def parse_netflowobject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['NetFlowObject']]
        entry_type = 'NetFlowObject'
        nodeid = entry_attrs['uuid']

        remoteaddr = entry_attrs['remoteAddress'].strip()  # only remote, ref.shadewatcher
        remoteport = str(entry_attrs['remotePort'])

        res_dict = {
            'type': entry_type,
            'remoteip': remoteaddr + ':' + remoteport
        }
        self.set(nodeid, res_dict)
        return None

    '''
    {
        "datum": {
            "com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject": {
                "uuid": "1D587E51-CC5F-70C3-7484-79426A8A1B5A",
                "baseObject": {
                    "hostId": "5957F7A8-2EAB-D99C-459A-408A1F427D29",
                    "permission": null,
                    "epoch": null,
                    "properties": null
                },
                "sourceFileDescriptor": null,
                "sinkFileDescriptor": null,
                "sourceUUID": null,
                "sinkUUID": null
            }
        },
        "CDMVersion": "18",
        "source": "SOURCE_ANDROID_JAVA_CLEARSCOPE"
    }
    '''
    def parse_unnamedPipeObject(self, entry: str) -> None:
        entry = ujson.loads(entry)
        entry_attrs = entry['datum'][config.DarpaE3_reserved_type_2_strings_map['UnnamedPipeObject']]
        entry_type = 'UnnamedPipeObject'
        nodeid = entry_attrs['uuid']
        res_dict = {
            'type': entry_type,
        }
        self.set(nodeid, res_dict)
        return None
