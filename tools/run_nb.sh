#!/bin/sh
# Regenera y ejecuta un notebook desde su fuente .py
set -e
cd "$(dirname "$0")/.."
python3 tools/py2nb.py "tools/$1" "notebooks/$2"
cd notebooks && jupyter nbconvert --to notebook --execute --inplace "$2" 2>&1 | grep -v -E "^\[NbConvertApp\]" | tail -15
python3 -c "
import json,sys
nb=json.load(open(sys.argv[1]))
for c in nb['cells']:
  if c['cell_type']=='code':
    for o in c.get('outputs',[]):
      if o.get('output_type')=='stream': print(''.join(o['text']))
      elif 'data' in o and 'text/plain' in o['data'] and 'image/png' not in o['data']: print(''.join(o['data']['text/plain']))
" "$2"
