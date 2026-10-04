import concurrent.futures as cf
import json, time, urllib.request, hashlib
import raise_verify as r

def main():
    r.verify_revision()
    rows=r.readjson(r.RUN/'plan.json')[1400:]
    todo=iter(row for row in rows if not (r.RUN/'source-results'/(row['File']+'.json')).exists())
    completed=0; started=time.perf_counter()
    with cf.ProcessPoolExecutor(12,initializer=r.init_worker) as pool:
        pending={}
        def submit():
            if r.readjson(r.RUN/'progress.json')['completed_sources']>=1250: return False
            try: row=next(todo)
            except StopIteration: return False
            pending[pool.submit(r.run_source,row)]=row
            return True
        for _ in range(12): submit()
        while pending:
            done,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            for future in done:
                row=pending.pop(future); result=future.result(); completed+=1
                assert not result.get('error'),result
                assert all(z.get('exact') for c in result['cases'] for z in c['codecs'].values()),result
                if completed%16==0: print(json.dumps(dict(tail_completed=completed,elapsed_s=round(time.perf_counter()-started,1))),flush=True)
                submit()
    r.writejson(r.RUN/'tail-complete.json',dict(completed=completed,elapsed_s=time.perf_counter()-started,workers=12,rank_start=1400))

if __name__=='__main__': main()
