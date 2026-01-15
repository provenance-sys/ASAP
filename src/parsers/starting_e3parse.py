import os
import sys
import tarfile
import argparse
import re
import datetime

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.utils import config
from src.parsers import dataparser_e3
from src.utils.datasaver import OpremSaveStrategy
from src.utils.opreventmodel import OPREventModel
from src.utils.cache_aside import CacheAside, LRUCache, SQLiteDB


parser_map = {
    'e3cadets': dataparser_e3.DarpaE3CadetsParser,
    'e3theia': dataparser_e3.DarpaE3TheiaParser,
    'e3trace': dataparser_e3.DarpaE3TraceParser,
    'e3fivedirections': dataparser_e3.DarpaE3FivedirectionsParser,
    'e3clearscope': dataparser_e3.DarpaE3ClearscopeParser,
}

# =========================unzip=========================

def start_unzip_tar_gz(src_path: str, dst_path: str):
    print(f'Start extracting: {src_path}')
    for file_name in os.listdir(src_path):
        if file_name.endswith('.json.tar.gz'):
            folder_name = file_name.replace('.json.tar.gz', '')
            folder_path = os.path.join(dst_path, folder_name)
            
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(src_path, file_name)

            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=folder_path)
            
            print(f'{file_name} has been extracted to {folder_path}')
    print(f'Finished extracting: {src_path}\n')

# =========================parser=========================
def get_json_filepaths(dataset_file_path: str) -> list:
    def extract_path_number(filename: str) -> int:
        match = re.search(r'\d+$', filename)
        if match:
            return int(match.group())
        else:
            return 0
    log_filepaths = []
    for file in os.listdir(dataset_file_path):
        filesplit = file.split('.')
        if filesplit[-1] == 'json' or filesplit[-1].isdigit():
            log_filepaths.append(os.path.join(dataset_file_path, file))
    log_filepaths = sorted(log_filepaths, key=extract_path_number)
    return log_filepaths

def parse_subfloader(dataset_name: str, json_dir: str, triple_dir: str, subfloader: str):
    json_path = os.path.join(json_dir, subfloader)
    triple_path = os.path.join(triple_dir, subfloader)
    os.makedirs(triple_path, exist_ok=True)

    db_path = os.path.join(triple_path, subfloader + '_entities_index.db')
    open(db_path, 'a').close()  # create file if not exist

    DParser = parser_map[dataset_name]
    dparser = DParser(db_name=db_path)

    total_num = 0
    json_file_list = get_json_filepaths(json_path)
    print("json_file_list:", json_file_list)
    for json_file_path in json_file_list:
        if dataset_name == 'e3trace':
            _file_name = json_file_path.split('/')[-1]
            if _file_name.startswith("ta1-trace-e3-official.json."):
                num = int(_file_name.rsplit(".", 1)[-1])
                if 155 <= num <= 203:
                    break
        if dataset_name == 'e3fivedirections':
            if subfloader == 'ta1-fivedirections-e3-official':
                break
        triple_file_path = os.path.join(triple_path, json_file_path.split('/')[-1] + '_parser_items.txt')
        gs = OpremSaveStrategy(oprem_path=triple_file_path)
        file_num = 0

        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                file_num += 1
                oprems = dparser.parse_single_entry(line.strip())
                if oprems is not None:
                    for _oprem in oprems:
                        gs.save_OPREM(_oprem)
        gs.close()
        print(f'{json_file_path} has been parsed to {triple_file_path}, Num of entries: {file_num}')
        total_num += file_num
    dparser.pop_cache_to_db()

def start_parse_json(dataset_name: str, json_dir: str, triple_dir: str):
    print(f'Start parsering: {json_dir}')
    for dataset_subfloader in os.listdir(json_dir):
        if os.path.isdir(os.path.join(json_dir, dataset_subfloader)):
            parse_subfloader(dataset_name, json_dir, triple_dir, dataset_subfloader)
    print(f'Finished parsering: {json_dir}\n')

# =========================trans2day=========================

def get_parser_floaderpath(dataset_file_path: str) -> list:
    def extract_floader_number(name: str) -> str:
        match = re.search(r'official(-[\w\d]+)?$', name)
        if match:
            return match.group(1) or ''
        else:
            return '-0'
    log_filepaths = []
    for file in os.listdir(dataset_file_path):
        log_filepaths.append(os.path.join(dataset_file_path, file))
    log_filepaths = sorted(log_filepaths, key=extract_floader_number)
    return log_filepaths

def get_parser_filepath(dataset_file_path: str) -> list:
    def extract_sort_key(filename):
        match = re.search(r'\.json(?:\.(\d+))?_parser_items\.txt$', filename)
        if match:
            number = match.group(1)
            return int(number) if number is not None else -1 
        else:
            return float('inf')
    log_filepaths = []
    for file in os.listdir(dataset_file_path):
        filesplit = file.split('.')
        if filesplit[-1] == 'txt':
            log_filepaths.append(os.path.join(dataset_file_path, file))
    return sorted(log_filepaths, key=extract_sort_key)

def start_trans_day(triple_dir: str, daily_dir: str):
    db_path = os.path.join(daily_dir, 'uid2vid.db')
    open(db_path, 'a').close()  # create file if not exist
    kvmap = CacheAside(
        LRUCache(maxlen=1000, pop_n=100),
        SQLiteDB(filename=db_path, tablename='default_table', flag='w', cache_size=40000, synchronous='OFF', journal_mode='MEMORY')
    )
    new_day = None
    old_day = None
    _vid = 1
    parser_floader_list = get_parser_floaderpath(triple_dir)
    for folder in parser_floader_list:
        file_list = get_parser_filepath(folder)
        for file_path in file_list:
            
            with open(file_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.rstrip('\n')
                    oprem = OPREventModel()
                    oprem.update_from_loprem(line.strip().split('\t'))
                    standard_date = datetime.datetime.fromtimestamp(oprem['e']['ts'] / 1e9).strftime("%Y-%m-%d")

                    u_vid = kvmap.get(oprem['u']['nid'])
                    if u_vid is None:
                        u_vid = _vid
                        _vid += 1
                        kvmap[oprem['u']['nid']] = u_vid
                    oprem['u']['vid'] = u_vid
                    
                    v_vid = kvmap.get(oprem['v']['nid'])
                    if v_vid is None:
                        v_vid = _vid
                        _vid += 1
                        kvmap[oprem['v']['nid']] = v_vid
                    oprem['v']['vid'] = v_vid

                    new_day = standard_date
                    if old_day is None:
                        old_day = new_day
                        gs = OpremSaveStrategy(reset=False, oprem_path=os.path.join(daily_dir, new_day + '.txt'))
                    
                    if new_day > old_day:
                        old_day = new_day
                        gs.close()
                        gs = OpremSaveStrategy(reset=False, oprem_path=os.path.join(daily_dir, new_day + '.txt'))
                    
                    gs.save_OPREM(oprem)
            print(f'{file_path} has already been processed.')
        kvmap.cache.clear()
        kvmap.db.delete_table()
    gs.close()

def start_trans_day_trace(triple_dir: str, daily_dir: str):
    '''
    merge SUBJECT_UNIT to SUBJECT_PROCESS
    '''
    db_path = os.path.join(daily_dir, 'uid2vid.db')
    open(db_path, 'a').close()  # create file if not exist
    kvmap = CacheAside(
        LRUCache(maxlen=1000, pop_n=100),
        SQLiteDB(filename=db_path, tablename='default_table', flag='w', cache_size=40000, synchronous='OFF', journal_mode='MEMORY')
    )

    unit_db_path = os.path.join(daily_dir, 'unit2process.db')
    open(unit_db_path, 'a').close()  # create file if not exist
    unit_kvmap = CacheAside(
        LRUCache(maxlen=1000, pop_n=100),
        SQLiteDB(filename=unit_db_path, tablename='default_table', flag='w', cache_size=40000, synchronous='OFF', journal_mode='MEMORY')
    )

    new_day = None
    old_day = None
    _vid = 1
    parser_floader_list = get_parser_floaderpath(triple_dir)
    for folder in parser_floader_list:
        file_list = get_parser_filepath(folder)
        for file_path in file_list:
            
            with open(file_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.rstrip('\n')
                    oprem = OPREventModel()
                    oprem.update_from_loprem(line.strip().split('\t'))
                    standard_date = datetime.datetime.fromtimestamp(oprem['e']['ts'] / 1e9).strftime("%Y-%m-%d")

                    if oprem['e']['type'] == 'EVENT_UNIT':
                        unit_kvmap[oprem['v']['nid']] = oprem['u']['nid']
                        continue
                    if oprem['u']['type'] == 'SUBJECT_UNIT':
                        process_nid = unit_kvmap.get(oprem['u']['nid'])
                        if process_nid is None:
                            continue
                        oprem['u']['type'] = 'SUBJECT_PROCESS'
                        oprem['u']['nid'] = process_nid
                    
                    if oprem['v']['type'] == 'SUBJECT_UNIT':
                        process_nid = unit_kvmap.get(oprem['v']['nid'])
                        if process_nid is None:
                            continue
                        oprem['v']['type'] = 'SUBJECT_PROCESS'
                        oprem['v']['nid'] = process_nid

                    u_vid = kvmap.get(oprem['u']['nid'])
                    if u_vid is None:
                        u_vid = _vid
                        _vid += 1
                        kvmap[oprem['u']['nid']] = u_vid
                    oprem['u']['vid'] = u_vid
                    
                    v_vid = kvmap.get(oprem['v']['nid'])
                    if v_vid is None:
                        v_vid = _vid
                        _vid += 1
                        kvmap[oprem['v']['nid']] = v_vid
                    oprem['v']['vid'] = v_vid

                    new_day = standard_date
                    if old_day is None:
                        old_day = new_day
                        gs = OpremSaveStrategy(reset=False, oprem_path=os.path.join(daily_dir, new_day + '.txt'))
                    
                    if new_day > old_day:
                        old_day = new_day
                        gs.close()
                        gs = OpremSaveStrategy(reset=False, oprem_path=os.path.join(daily_dir, new_day + '.txt'))
                    
                    gs.save_OPREM(oprem)
            print(f'{file_path} has already been processed.')
        kvmap.cache.clear()
        kvmap.db.delete_table()
    gs.close()

def parsering_darpae3(dataset_name):
    # The data has been downloaded from https://drive.google.com/drive/folders/1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK
    e3_targz_dataset_dir = os.path.join(config.e3_targz_dir, dataset_name[2:])
    if not os.path.isdir(e3_targz_dataset_dir) or not os.listdir(e3_targz_dataset_dir):
        print(f"Folder does not exist or is empty: {e3_targz_dataset_dir}")
        sys.exit(1)

    # unzip raw data to JSON
    e3_unziped_json_dir = os.path.join(config.artifact_dir, 'json', dataset_name)
    os.makedirs(e3_unziped_json_dir, exist_ok=True)
    if not os.listdir(e3_unziped_json_dir):
        start_unzip_tar_gz(e3_targz_dataset_dir, e3_unziped_json_dir)

    # Parse JSON data into (subject, object, event) triples
    e3_parsed_triple_dir = os.path.join(config.artifact_dir, 'parse', dataset_name)
    os.makedirs(e3_parsed_triple_dir, exist_ok=True)
    if not os.listdir(e3_parsed_triple_dir):
        start_parse_json(dataset_name, e3_unziped_json_dir, e3_parsed_triple_dir)
    
    e3_daily_split_dir = os.path.join(config.artifact_dir, 'daily', dataset_name)
    os.makedirs(e3_daily_split_dir, exist_ok=True)
    if not os.listdir(e3_daily_split_dir):
        if 'trace' not in dataset_name:            
            start_trans_day(e3_parsed_triple_dir, e3_daily_split_dir)
        else:
            start_trans_day_trace(e3_parsed_triple_dir, e3_daily_split_dir)

if __name__ == '__main__':
    parsering_darpae3("e3theia")
    # start_trans_day("/data2/sj/ASAP_DATA/parse/e3cadets", "/data2/sj/ASAP_DATA/daily/e3cadets")

