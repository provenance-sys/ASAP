import os

e3_targz_dir = "DARPA_E3_raw_data_path"  # cadets theia trace fivedirections clearscope
artifact_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/")
CENTRAL_NODE_TYPE: list = ['SUBJECT_PROCESS', 'SUBJECT_THREAD']

# ============data parse config=========== #
DarpaE3_reserved_type_2_strings_map = {
    'Event': 'com.bbn.tc.schema.avro.cdm18.Event',
    'NetFlowObject': 'com.bbn.tc.schema.avro.cdm18.NetFlowObject',
    'FileObject': 'com.bbn.tc.schema.avro.cdm18.FileObject',
    'Subject': 'com.bbn.tc.schema.avro.cdm18.Subject',
    'RegistryKeyObject':'com.bbn.tc.schema.avro.cdm18.RegistryKeyObject',
    'UnnamedPipeObject': 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject',
    'MemoryObject': 'com.bbn.tc.schema.avro.cdm18.MemoryObject',
}

edge_reversed = [
    'EVENT_READ', 
    'EVENT_MMAP',
    'EVENT_ACCEPT', 
    'EVENT_EXECUTE', 
    'EVENT_RECVMSG', 
    'EVENT_RECVFROM', 
    'EVENT_READ_SOCKET_PARAMS', 
    'EVENT_LOADLIBRARY', 
    'EVENT_CHECK_FILE_ATTRIBUTES', 
]

node_type = {
    'e3cadets': ['SUBJECT_PROCESS', 'FILE_OBJECT_FILE', 'FILE_OBJECT_UNIX_SOCKET', 'FILE_OBJECT_DIR', 'UnnamedPipeObject', 'NetFlowObject'],
    'e3theia' : ['SUBJECT_PROCESS', 'MemoryObject', 'FILE_OBJECT_BLOCK', 'NetFlowObject'],
    'e3trace': ['FILE_OBJECT_UNIX_SOCKET', 'SUBJECT_PROCESS', 'NetFlowObject', 'FILE_OBJECT_FILE', 'MemoryObject', 'FILE_OBJECT_CHAR', 'FILE_OBJECT_DIR', 'UnnamedPipeObject'],
    'e3clearscope': ['SUBJECT_PROCESS', 'FILE_OBJECT_FILE', 'NetFlowObject', 'FILE_OBJECT_UNIX_SOCKET', 'FILE_OBJECT_DIR', 'UnnamedPipeObject'],
}

edge_type = {
    'e3cadets': ['EVENT_READ', 'EVENT_WRITE', 'EVENT_MMAP', 'EVENT_FORK', 'EVENT_EXECUTE', 'EVENT_SENDTO', 'EVENT_RECVFROM', 'EVENT_SENDMSG', 'EVENT_RECVMSG', 'EVENT_CONNECT', 'EVENT_ACCEPT', 'EVENT_OPEN'], 
    'e3theia': ['EVENT_MMAP', 'EVENT_READ', 'EVENT_MPROTECT', 'EVENT_SENDTO', 'EVENT_RECVMSG', 'EVENT_READ_SOCKET_PARAMS', 'EVENT_SENDMSG', 'EVENT_EXECUTE', 'EVENT_RECVFROM', 'EVENT_WRITE', 'EVENT_WRITE_SOCKET_PARAMS', 'EVENT_CLONE'],
    'e3trace': ['EVENT_RECVMSG', 'EVENT_SENDMSG', 'EVENT_MMAP', 'EVENT_READ', 'EVENT_WRITE', 'EVENT_MPROTECT', 'EVENT_FORK', 'EVENT_LOADLIBRARY', 'EVENT_EXECUTE', 'EVENT_CLONE', 'EVENT_UPDATE', 'EVENT_UNIT'],
    'e3clearscope': ['EVENT_READ', 'EVENT_CONNECT', 'EVENT_WRITE', 'EVENT_LOADLIBRARY', 'EVENT_RECVFROM', 'EVENT_SENDTO', 'EVENT_MMAP', 'EVENT_SENDMSG', 'EVENT_RECVMSG'],
}

