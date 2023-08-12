# -*- encoding: utf-8 -*-

import zipfile
import os
import logging
from datetime import datetime


def zip_dir(dirname, zipfilename, clear_dir=False):
    logging.debug(f"zipping dir: {dirname}")
    filelist = []
    for root, dirs, files in os.walk(dirname):
        for dir in dirs:
            filelist.append(os.path.join(root,dir))
        for name in files:
            filelist.append(os.path.join(root, name))
    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:
        arcname = tar[len(dirname):]
        logging.debug(f"zipping {arcname} to {zipfilename}")
        zf.write(tar, arcname)
    zf.close()
    logging.debug(f"done zipping dir.")


def select_files_by_timestamp_range(folder_path: str, ts_start: int, ts_end: int):
    selected_files = []
    if ts_start >= ts_end:
        return []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Get the timestamp from the filename
            timestamp = int(os.path.splitext(filename)[0])
            
            # Check if the timestamp falls within the specified range
            if ts_start <= timestamp <= ts_end:
                selected_files.append(file_path)
        except ValueError:
            # Skip files with invalid timestamp format
            continue
    
    return selected_files

