import concurrent.futures as cf
import csv, json, shutil
import raise_verify as r

def main():
    r.verify_revision()
    initial=r.readjson(r.RUN/'plan.json')
    assert len(initial)==2016
    audit_records={row['File']:r.readjson(r.RUN/'audit-records'/(row['File']+'.json')) for row in initial}
    assert not any(x['errors'] for x in audit_records.values())
    if (r.RUN/'verification.json').exists() and not (r.RUN/'initial-sample-verification.json').exists():
        shutil.copyfile(r.RUN/'verification.json',r.RUN/'initial-sample-verification.json')
    qualified=[]; excluded=[]; hashes={key:{} for key in ['raw_file_sha256','sensor_sha256','prepared_sha256']}
    def duplicates(result):
        return {key:hashes[key][result['source'][key]] for key in hashes if result['source'][key] in hashes[key]}
    def accept(row,result):
        qualified.append(row)
        for key in hashes: hashes[key][result['source'][key]]=row['File']
    for row in initial:
        result=r.readjson(r.RUN/'source-results'/(row['File']+'.json'))
        assert not result.get('error')
        duplicate_of=duplicates(result)
        if duplicate_of: excluded.append(dict(row=row,duplicate_of=duplicate_of))
        else: accept(row,result)
    catalog=sorted(list(csv.DictReader((r.RUN/'evidence/RAISE_all.csv').open(encoding='utf-8-sig'))),key=lambda row:r.sha(('tsiro-raise-2016-v1/'+row['File']).encode()))
    cursor=2016; replacements=[]; additional_duplicates=[]
    with cf.ProcessPoolExecutor(4,initializer=r.init_worker) as encoder, cf.ProcessPoolExecutor(2,initializer=r.init_worker) as auditor:
        pending=list(excluded)
        while pending:
            candidates=[]
            for missing in pending:
                row=dict(catalog[cursor]); row.update(rank=cursor,jpeg_quality=missing['row']['jpeg_quality'],jpeg_subsampling=missing['row']['jpeg_subsampling']); cursor+=1
                candidates.append(row)
            encoded=list(encoder.map(r.run_source,candidates))
            audited=list(auditor.map(r.audit_source,candidates))
            retry=[]
            for missing,row,result,audit in zip(pending,candidates,encoded,audited):
                assert not result.get('error'),result
                assert all(z.get('exact') for c in result['cases'] for z in c['codecs'].values()),result
                r.writejson(r.RUN/'audit-records'/(row['File']+'.json'),audit)
                assert not audit['errors'],audit
                duplicate_of=duplicates(result)
                if duplicate_of:
                    additional_duplicates.append(dict(row=row,duplicate_of=duplicate_of)); retry.append(missing); continue
                audit_records[row['File']]=audit; accept(row,result)
                replacements.append(dict(replaces=missing['row']['File'],source_id=row['File'],rank=row['rank'],jpeg_quality=row['jpeg_quality'],jpeg_subsampling=row['jpeg_subsampling']))
                print(json.dumps(replacements[-1]),flush=True)
            pending=retry
    qualified.sort(key=lambda row:row['rank'])
    assert len(qualified)==2016 and all(len(values)==2016 for values in hashes.values())
    from collections import Counter
    counts=Counter((row['jpeg_quality'],row['jpeg_subsampling']) for row in qualified)
    assert len(counts)==12 and set(counts.values())=={168}
    r.writejson(r.RUN/'qualified-plan.json',qualified)
    r.writejson(r.RUN/'qualification.json',dict(
        rule='Keep lowest hash-ranked source for duplicate RAW-file, sensor or prepared-pixel hashes. Replace excluded IDs using next unused catalog hash ranks, keeping the excluded JPEG profile. Selection depends only on source identity, never codec outcomes.',
        initial_catalog_ids=2016,distinct_qualified_sources=2016,initial_duplicates=excluded,additional_duplicates=additional_duplicates,replacements=replacements,
        all_tested_sources=len(initial)+len(replacements)+len(additional_duplicates)))
    r.finish_audit([audit_records[row['File']] for row in qualified])
    protocol=r.readjson(r.RUN/'protocol.json')
    if not (r.RUN/'evidence/initial-protocol.json').exists(): r.writejson(r.RUN/'evidence/initial-protocol.json',protocol)
    protocol['selection']=r.readjson(r.RUN/'evidence/initial-protocol.json')['selection']+' Duplicate IDs are excluded from clean exports and replaced with next unused catalog hash ranks while preserving the excluded JPEG profile. See qualification.json and qualified-plan.json.'
    protocol['execution']=r.readjson(r.RUN/'bulk-complete.json')
    protocol['audit_execution']='Sources redeveloped and saved streams decoded in separate audit processes. Duplicate replacements use four encoding workers and two audit workers.'
    r.writejson(r.RUN/'protocol.json',protocol)
    print(json.dumps(dict(qualified=2016,excluded_duplicates=len(excluded),replacements=len(replacements))),flush=True)

if __name__=='__main__': main()
