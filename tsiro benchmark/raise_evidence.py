import hashlib, json, urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent/'results-2026-10-04-raise2000/evidence'
records=[]
for page in ['index.php','guide.html']:
    url='https://loki.disi.unitn.it/RAISE/'+page
    data=urllib.request.urlopen(url,timeout=30).read()
    (p/('publisher-'+page+'.html')).write_bytes(data)
    records.append(dict(url=url,sha256=hashlib.sha256(data).hexdigest()))
records.append(dict(file='RAISE_all.csv',sha256=hashlib.sha256((p/'RAISE_all.csv').read_bytes()).hexdigest()))
(p/'publisher-pages.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records))
