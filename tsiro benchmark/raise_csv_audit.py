import collections, csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'results-2026-10-04-raise2000'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
sources={r['source_id']:r for r in (read(p) for p in (RUN/'source-results').glob('*.json'))}
verification=read(RUN/'verification.json'); assert verification['passed']
summary=read(RUN/'summary.json'); exports=read(RUN/'csv-export.json')
checks=[]
for exported in exports:
    path=RUN/exported['file']; is_jpeg=path.name.startswith('jpeg-')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==exported['sha256']
    with path.open(encoding='utf-8-sig',newline='') as file: rows=list(csv.DictReader(file))
    assert len(rows)==16128
    seen=set(); totals=collections.Counter(); profiles=collections.Counter(); by_source=collections.Counter()
    for row in rows:
        pair=(row['case_id'],row['codec']); assert pair not in seen; seen.add(pair)
        source=sources[row['source_id']]; s=source['source']
        case=next(c for c in source['cases'] if c['id']==row['case_id'])
        z=case['codecs'][row['codec']]
        assert bool(case.get('jpeg'))==is_jpeg
        assert row['pixel_exact']==row['fresh_disk_audit_passed']=='True'
        assert int(row['encoded_bytes'])==z['bytes']
        assert float(row['bits_per_pixel'])==8*z['bytes']/case['width']/case['height']
        assert int(row['width'])==case['width'] and int(row['height'])==case['height']
        assert int(row['channels'])==3 and int(row['bit_depth'])==8
        assert int(row['raw_rgb_bytes'])==case['raw_bytes']
        assert row['input_pixel_sha256']==row['decoded_pixel_sha256']==case['input_sha256']==z['decoded_sha256']
        assert row['stream_sha256']==z['stream_sha256']
        assert (RUN/row['stream_path']).stat().st_size==z['bytes']
        assert row['source_raw_file_sha256']==s['raw_file_sha256']
        assert row['sensor_sha256']==s['sensor_sha256']
        assert row['prepared_pixel_sha256']==s['prepared_sha256']
        assert row['source_url']==s['source_url']
        assert row['camera']==source['camera']
        assert int(row['source_rank'])==source['rank']
        assert int(row['sensor_compression_tag']) in (1,34713)
        assert int(row['jpeg_generation_count'])==int(is_jpeg)
        assert float(row['encode_s_bulk'])==z['encode_s_bulk']
        assert float(row['decode_s_bulk'])==z['decode_s_bulk']
        assert row['tsiro_mode']==(z['mode'] if row['codec']=='Tsiro' else '')
        if is_jpeg:
            j=case['jpeg']
            assert int(row['jpeg_quality'])==j['quality'] and int(row['jpeg_subsampling_code'])==j['subsampling']
            assert row['jpeg_chroma']==j['chroma']
            assert row['jpeg_file_sha256']==j['sha256'] and int(row['jpeg_file_bytes'])==j['bytes']
            assert json.loads(row['jpeg_quantization_tables_row_major_json'])==j['qtables']
            assert json.loads(row['jpeg_components_id_h_v_qtable_json'])==j['components']
            assert row['jpeg_progressive']==row['jpeg_optimize']=='False'
            assert row['jpeg_encoder']==j['encoder'] and row['jpeg_decoder']==j['decoder']
        else:
            assert all(value=='' for key,value in row.items() if key.startswith('jpeg_') and key!='jpeg_generation_count')
        totals[row['codec']]+=int(row['encoded_bytes']); profiles[row['profile']]+=1; by_source[row['source_id']]+=1
    assert len(by_source)==2016 and set(by_source.values())=={8}
    unique_inputs=len({row['input_pixel_sha256'] for row in rows})
    assert unique_inputs==2016
    if is_jpeg: assert len(profiles)==12 and set(profiles.values())=={1344}
    else: assert profiles=={'original':16128}
    assert dict(totals)==summary['groups']['jpeg_all' if is_jpeg else 'original']['codec_bytes']
    checks.append(dict(file=path.name,rows=len(rows),sources=len(by_source),distinct_input_rasters=unique_inputs,profiles=dict(profiles),passed=True))
(RUN/'csv-verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
