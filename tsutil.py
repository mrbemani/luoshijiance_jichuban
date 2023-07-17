# -*- encoding: utf-8 -*-

import zipfile
import os
import logging


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


