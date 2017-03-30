#!/usr/bin/env python

"""Create a jobs.json file for schedule.py."""

import glob
import natsort
import json
# Make it work for Python 2+3 and with Unicode
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str


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

experiment_files = glob.glob("experiments/*.yaml")
experiment_files = natsort.natsorted(experiment_files)

data = []
for f in experiment_files:
    data.append({'cmd': './run_training.py -f {}'.format(f)})

write_json("jobs-auto.json", data)
