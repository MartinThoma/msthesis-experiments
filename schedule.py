#!/usr/bin/env python

"""Run job scheduler."""

import logging
import sys
import json
import subprocess
import time
import shlex
# Make it work for Python 2+3 and with Unicode
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def write_json(fname, data):
    # Write JSON file
    with io.open(fname, 'w', encoding='utf8') as outfile:
        for i in range(len(data)):
            if 'output' in data[i]:
                data[i]['output'] = data[i]['output'].decode('utf8')
        str_ = json.dumps(data,
                          indent=4, sort_keys=True,
                          separators=(',', ': '),
                          ensure_ascii=False)
        outfile.write(to_unicode(str_))


def main(jobs_fpath):
    # Read JSON file
    with open(jobs_fpath) as data_file:
        jobs = json.load(data_file)
    for job in jobs:
        if 'time' in job and job['time'] > 0:
            continue  # was already exectued
        t0 = time.time()
        output = subprocess.check_output(shlex.split(job['cmd']))
        t1 = time.time()
        job['output'] = output
        job['time'] = t1 - t0
        write_json(jobs_fpath, jobs)


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="jobs file",
                        metavar="FILE.json",
                        required=True)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.filename)
