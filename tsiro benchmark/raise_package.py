import hashlib
import json
import shutil
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'results-2026-10-04-raise2000'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

verification = read(RUN/'verification.json')
qualification = read(RUN/'qualification.json')
checks = read(RUN/'csv-verification.json')
assert verification['passed'] and set(verification['unique'].values()) == {2016}
assert len(checks) == 2 and all(row['passed'] and row['distinct_input_rasters'] == 2016 for row in checks)
timings = [json.loads(line) for line in (RUN/'timings.jsonl').read_text().splitlines()]
assert len(timings) == 24 and all(z['exact'] for row in timings for z in row['codecs'].values())

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(value for key, value in attrs if key == 'href')

parser = Links()
parser.feed((RUN/'report.html').read_text(encoding='utf-8'))
local_links = [link for link in parser.links if not link.startswith(('https://', 'http://', '#'))]
for link in local_links:
    assert (RUN/link).exists(), link

write(RUN/'execution.json', dict(
    completed_utc=datetime.now(timezone.utc).isoformat(),
    bulk=read(RUN/'bulk-complete.json'),
    qualified_sources=verification['sources'],
    tested_sources=qualification['all_tested_sources'],
    duplicate_replacements=len(qualification['replacements']),
    verified_cases=verification['cases'],
    verified_streams=verification['streams'],
    timing_cases=len(timings),
))
write(RUN/'progress.json', dict(phase='complete', sources=verification['sources'], cases=verification['cases'], streams=verification['streams'], errors=0))
for name in ('jpeg-recompression-only.csv', 'tsiro-only-compression.csv'):
    shutil.copyfile(RUN/name, ROOT/name)
files = [RUN/name for name in (
    'jpeg-recompression-only.csv', 'tsiro-only-compression.csv', 'report.html', 'summary.json',
    'verification.json', 'csv-verification.json', 'protocol.json', 'environment.json',
    'qualified-plan.json', 'qualification.json', 'timings.jsonl', 'execution.json',
)]
files += sorted(ROOT.glob('*.py'))
files += [ROOT/name for name in ('commands.txt', 'requirements.txt', '.csv-export-tools/raise-results.mjs', 'vendor/Tsiro/tsiro.py', 'vendor/Tsiro/REVISION')]
write(RUN/'manifest.json', dict(files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))
print(json.dumps(dict(sources=verification['sources'], streams=verification['streams'], csv_checks=checks, report_links=len(local_links))))
