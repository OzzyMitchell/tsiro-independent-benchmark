import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
import argparse, concurrent.futures as cf, csv, hashlib, io, json, platform, shutil, statistics, subprocess, sys, time, traceback, urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
import rawpy
import tifffile
from PIL import Image, features
import benchmark as b

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results-2026-10-04-raise2000'
CODECS = ['Tsiro', 'PNG', 'WebP', 'JXL3', 'JXL7', 'JXL9', 'QOI', 'JPEGLS']
PROFILES = [(q, ss) for q in [60, 75, 85, 90, 95, 98] for ss in [0, 2]]
REV = '36287d4d7d753f8a74beb8c5047fe6e5974fcfa6'
SOURCE_SHA = '3357f18c502276e41e6e081c3119b454dfae4a538fd7eb3cdca322cd0f3302d1'

def sha(data): return hashlib.sha256(data).hexdigest()
def writejson(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt==19: raise
            time.sleep(.05*(attempt+1))
def readjson(path): return json.loads(path.read_text(encoding='utf-8'))
def selection(): return readjson(RUN/'source-selection.json')
def verify_revision():
    assert sha((ROOT/'vendor/Tsiro/tsiro.py').read_bytes()) == SOURCE_SHA
    assert (ROOT/'vendor/Tsiro/REVISION').read_text().strip() == REV
    if (RUN/'environment.json').exists():
        expected=readjson(RUN/'environment.json')
        assert Image.__version__==expected['pillow']
        assert features.version('libjpeg_turbo')==expected['turbo']
        assert rawpy.__version__==expected['rawpy'] and list(rawpy.libraw_version)==expected['libraw']
        assert b.ic.version()==expected['imagecodecs']

def plan():
    verify_revision()
    (RUN/'evidence').mkdir(parents=True,exist_ok=True)
    catalog=ROOT/'inputs/RAISE_all.csv'
    assert sha(catalog.read_bytes())=='2d78a40847564a4d2e810377b8b208c7207c25d11e1b7b2b10c6de5395c760d3'
    for name in ['RAISE_all.csv','metadata-source.json']:
        shutil.copyfile(ROOT/'inputs'/name,RUN/'evidence'/name)
    ranked=sorted(list(csv.DictReader(catalog.open(encoding='utf-8-sig'))),key=lambda r:sha(('tsiro-raise-2016-v1/'+r['File']).encode()))
    rows=ranked[:2016]
    assert len(rows)==2016 and len({r['File'] for r in rows})==2016
    writejson(RUN/'source-selection.json',rows)
    writejson(RUN/'hardware.json',dict(platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),logical_cpus=os.cpu_count()))
    for i,r in enumerate(rows):
        r['rank']=i; r['jpeg_quality'],r['jpeg_subsampling']=PROFILES[i%12]
    writejson(RUN/'plan.json',rows)
    writejson(RUN/'protocol.json',dict(
        tsiro_revision=REV,tsiro_source_sha256=SOURCE_SHA,
        target_distinct_sources=2016,planned_cases=4032,
        source='RAISE original NEF captures from https://loki.disi.unitn.it/RAISE/NEF/. Public catalog pinned in evidence/metadata-source.json.',
        citation='Dang-Nguyen, Pasquini, Conotter, Boato. RAISE - A Raw Images Dataset for Digital Image Forensics. ACM MMSys 2015.',
        selection='First 2016 source IDs sorted by SHA256(tsiro-raise-2016-v1/ + source ID), fixed before compression. Exact RAW, mosaic, and RGB duplicates audited separately. Different exposures/scenes may be similar; distinct files are not claimed to be independent scenes.',
        provenance='Require camera-native Nikon CFA sensor IFD (PhotometricInterpretation 32803), non-JPEG sensor compression (1 or Nikon 34713), uint16 2D Bayer mosaic. Embedded JPEG previews are inspected as separate IFDs and never used. Each NEF is hashed and re-read in the fresh audit.',
        development=dict(rawpy='0.27.1',libraw=list(rawpy.libraw_version),demosaic='AHD',use_camera_wb=True,no_auto_bright=True,output_color='sRGB',output_bps=8,gamma=[2.222,4.5],half_size=False,user_flip=0),
        preparation='Develop full visible sensor raster via LibRaw; Pillow LANCZOS thumbnail bounded by 1024x1024, preserving aspect ratio; centered crop removes at most 15 pixels on either axis to dimensions divisible by 16. Save exact RGB8 PNG. One raster per source. This qualifies prepared rasters, not full-resolution images or original RAW-file byte reconstruction.',
        profiles='Each source: original RGB and one JPEG-derived RGB case. Quality 60/75/85/90/95/98 crossed with 4:4:4/4:2:0, assigned cyclically in hash order: 168 distinct sources per combination. No selection by codec outcome.',
        jpeg='Pillow 12.3.0/libjpeg-turbo 3.1.4.1; optimize=False; progressive=False; decode with same Pillow. Store original generated JPEG, exact DQT tables in natural row-major order, component sampling/table IDs, byte SHA256 and RGB SHA256. Tsiro receives only decoded pixels.',
        codecs=CODECS,settings={'PNG':'Pillow optimize=True compress_level=9','WebP':'Pillow lossless=True quality=100 method=6 exact=True','JXL3/7/9':'imagecodecs lossless=True effort=3/7/9 numthreads=1','QOI':'imagecodecs','JPEGLS':'imagecodecs level=0','Tsiro':'unmodified compress_pixels automatic mode selection'},
        correctness='Check dtype, shape, every RGB sample and decoded SHA256 for all codecs; retain every output. Second fresh process rechecks every NEF, redevelops every source, re-reads every stored JPEG/PNG and decodes every saved stream. Keep all failures in results.',
        timing='Bulk runs under parallel contention; use bulk timings only descriptively. Separate serial warm timing on first 12 hash-ranked sources, both profiles, 3 repetitions with rotated codec order.',
        resource_scope='1024-edge prepared RGB pixel study. Existing original-resolution Kodak and odd-dimension/input-format diagnostics are separate evidence.',
    ))
    writejson(RUN/'environment.json',dict(python=sys.version,platform=platform.platform(),cpu=os.environ.get('PROCESSOR_IDENTIFIER'),cpu_count=os.cpu_count(),rawpy=rawpy.__version__,libraw=list(rawpy.libraw_version),pillow=Image.__version__,jpeg=features.version('jpg'),turbo=features.version('libjpeg_turbo'),imagecodecs=b.ic.version(),pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)))
    print(json.dumps({'sources':len(rows),'cameras':dict(Counter(r['Device'] for r in rows)),'profiles':12,'per_profile':168}),flush=True)

def ifd_inventory(path):
    entries=[]
    def visit(pages):
        for page in pages:
            entries.append(dict(offset=page.offset,photometric=int(page.photometric),compression=int(page.compression),shape=list(page.shape),bits_per_sample=page.bitspersample))
            if page.subifds: visit(page.pages)
    with tifffile.TiffFile(path) as t: visit(t.pages)
    sensor=[p for p in entries if p['photometric']==32803]
    assert len(sensor)==1 and sensor[0]['compression'] in (1,34713), entries
    assert sensor[0]['bits_per_sample'] in (12,14,16)
    return entries

def develop(path):
    with rawpy.imread(str(path)) as raw:
        mosaic=raw.raw_image_visible
        assert mosaic.ndim==2 and mosaic.dtype==np.uint16
        assert raw.raw_pattern is not None and raw.raw_pattern.shape==(2,2)
        meta=dict(sensor_shape=list(mosaic.shape),sensor_dtype=str(mosaic.dtype),sensor_sha256=sha(mosaic.tobytes()),white_level=raw.white_level,bayer_pattern=raw.raw_pattern.tolist(),color_desc=raw.color_desc.decode('ascii'))
        rgb=raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,use_camera_wb=True,no_auto_bright=True,output_color=rawpy.ColorSpace.sRGB,output_bps=8,gamma=(2.222,4.5),half_size=False,user_flip=0)
    meta['developed_shape']=list(rgb.shape)
    im=Image.fromarray(rgb); im.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    w,h=im.size; cw,ch=w-w%16,h-h%16; left,top=(w-cw)//2,(h-ch)//2
    meta.update(resized_size=[w,h],crop_box=[left,top,left+cw,top+ch])
    arr=np.array(im.crop(tuple(meta['crop_box'])))
    assert arr.dtype==np.uint8 and arr.ndim==3 and arr.shape[2]==3
    meta.update(prepared_shape=list(arr.shape),prepared_sha256=sha(arr.tobytes()))
    return np.ascontiguousarray(arr),meta

def download(row):
    folder=RUN/'raw'; folder.mkdir(exist_ok=True)
    path=folder/(row['File']+'.NEF')
    url='https://loki.disi.unitn.it/RAISE/NEF/'+row['File']+'.NEF'
    if path.exists(): return path,url
    for attempt in range(4):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'Tsiro-independent-research/1.0'})
            with urllib.request.urlopen(request,timeout=60) as response:
                data=response.read(); expected=response.headers.get('Content-Length')
            if expected: assert len(data)==int(expected)
            assert len(data)>1_000_000 and data[:4] in (b'MM\x00*',b'II*\x00')
            tmp=path.with_suffix('.part'); tmp.write_bytes(data); tmp.replace(path)
            return path,url
        except Exception:
            if attempt==3: raise
            time.sleep(2*(attempt+1))

def init_worker():
    b.RUN=RUN
    b.init_worker()

def test_pixels(row,profile,arr,jpeg=None):
    cid='RAISE__'+row['File']+'__'+profile
    key=sha(cid.encode())[:20]; d=RUN/'streams'/key; d.mkdir(parents=True,exist_ok=True)
    record=dict(id=cid,key=key,source_id=row['File'],dataset='RAISE',rank=row['rank'],camera=row['Device'],profile=profile,width=arr.shape[1],height=arr.shape[0],dtype=str(arr.dtype),shape=list(arr.shape),raw_bytes=arr.nbytes,input_sha256=sha(arr.tobytes()),codecs={})
    if jpeg: record['jpeg']=jpeg
    for codec in CODECS:
        z={}
        try:
            start=time.perf_counter(); stream=b.encode(codec,arr); z['encode_s_bulk']=time.perf_counter()-start
            (d/(codec+'.bin')).write_bytes(stream)
            start=time.perf_counter(); dec=b.decode(codec,stream); z['decode_s_bulk']=time.perf_counter()-start
            exact=arr.dtype==dec.dtype and arr.shape==dec.shape and np.array_equal(arr,dec)
            z.update(bytes=len(stream),stream_sha256=sha(stream),decoded_sha256=sha(dec.tobytes()),decoded_shape=list(dec.shape),decoded_dtype=str(dec.dtype),exact=bool(exact))
            if codec=='Tsiro': z['mode']=stream[:4].decode('ascii',errors='replace')
            if not exact and arr.shape==dec.shape: z['different_samples']=int(np.count_nonzero(arr!=dec))
        except Exception as e: z.update(error=repr(e),traceback=traceback.format_exc())
        record['codecs'][codec]=z
    return record

def run_source(row):
    dest=RUN/'source-results'/(row['File']+'.json')
    if dest.exists():
        previous=readjson(dest)
        if not previous.get('error') and len(previous.get('cases',[]))==2 and all(z.get('exact') for c in previous['cases'] for z in c['codecs'].values()): return previous
    start=time.perf_counter()
    result=dict(source_id=row['File'],rank=row['rank'],camera=row['Device'],cases=[])
    try:
        path,url=download(row)
        inventory=ifd_inventory(path)
        arr,meta=develop(path)
        meta.update(source_id=row['File'],camera=row['Device'],source_url=url,raw_file_sha256=sha(path.read_bytes()),raw_file_bytes=path.stat().st_size,ifds=inventory,raw_quality=row['Image Quality'],keywords=row['Keywords'])
        png=RUN/'prepared'/(row['File']+'.png'); png.parent.mkdir(exist_ok=True)
        Image.fromarray(arr).save(png,compress_level=1)
        meta['png_file_sha256']=sha(png.read_bytes()); result['source']=meta
        result['cases'].append(test_pixels(row,'original',arr))
        q,ss=row['jpeg_quality'],row['jpeg_subsampling']
        buf=io.BytesIO(); Image.fromarray(arr).save(buf,'JPEG',quality=q,subsampling=ss,optimize=False,progressive=False)
        blob=buf.getvalue(); jp=RUN/'derived'/(row['File']+'.jpg'); jp.parent.mkdir(exist_ok=True); jp.write_bytes(blob)
        with Image.open(io.BytesIO(blob)) as image:
            image.load(); jpeg=dict(file=jp.relative_to(RUN).as_posix(),bytes=len(blob),sha256=sha(blob),quality=q,subsampling=ss,chroma='4:4:4' if ss==0 else '4:2:0',qtables=image.quantization,components=image.layer,progressive=bool(image.info.get('progressive') or image.info.get('progression')),optimize=False,generations=1,encoder='Pillow 12.3.0/libjpeg-turbo 3.1.4.1',decoder='Pillow 12.3.0/libjpeg-turbo 3.1.4.1')
            pixels=np.array(image.convert('RGB'))
        result['cases'].append(test_pixels(row,f'jpeg_q{q}_{444 if ss==0 else 420}',pixels,jpeg))
    except Exception as e: result.update(error=repr(e),traceback=traceback.format_exc())
    result['elapsed_s']=time.perf_counter()-start
    writejson(dest,result)
    return result

def run(workers,limit=None):
    verify_revision(); rows=readjson(RUN/'plan.json')
    if limit: rows=rows[:limit]
    todo=[]
    for row in rows:
        path=RUN/'source-results'/(row['File']+'.json')
        previous=readjson(path) if path.exists() else {}
        if previous.get('error') or len(previous.get('cases',[]))!=2 or not all(z.get('exact') for c in previous['cases'] for z in c['codecs'].values()): todo.append(row)
    existing=len(rows)-len(todo); started=time.perf_counter(); errors=0
    print(json.dumps(dict(action='run',remaining=len(todo),existing=existing,workers=workers)),flush=True)
    with cf.ProcessPoolExecutor(workers,initializer=init_worker) as pool:
        futures={pool.submit(run_source,r):r for r in todo}
        for n,f in enumerate(cf.as_completed(futures),1):
            r=f.result(); errors+=bool(r.get('error'))+sum(not z.get('exact',False) for c in r.get('cases',[]) for z in c['codecs'].values())
            progress=dict(completed_sources=existing+n,total_sources=len(rows),errors=errors,elapsed_s=round(time.perf_counter()-started,1),latest=r['source_id'])
            writejson(RUN/'progress.json',progress)
            if n%8==0 or errors or n==len(todo): print(json.dumps(progress),flush=True)
    writejson(RUN/'bulk-complete.json',dict(sources=len(rows),elapsed_s=time.perf_counter()-started,workers=workers,errors=errors))
    if errors: raise RuntimeError(f'{errors} benchmark failures')

def audit_source(row):
    result=readjson(RUN/'source-results'/(row['File']+'.json'))
    errors=[]; streams=0
    if result.get('error'): return dict(source_id=row['File'],errors=[result['error']],streams=0)
    source=result['source']; path=RUN/'raw'/(row['File']+'.NEF')
    with tifffile.TiffFile(path) as tf:
        actual_camera=tf.pages[0].tags['Model'].value.strip()
        if actual_camera.casefold()!=row['Device'].casefold(): errors.append('Camera model differs from catalog')
    if sha(path.read_bytes())!=source['raw_file_sha256']: errors.append('NEF hash differs')
    if ifd_inventory(path)!=source['ifds']: errors.append('IFD inventory differs')
    arr,meta=develop(path)
    if meta['sensor_sha256']!=source['sensor_sha256']: errors.append('Sensor mosaic differs')
    if meta['prepared_sha256']!=source['prepared_sha256']: errors.append('Fresh RAW development differs')
    png=RUN/'prepared'/(row['File']+'.png')
    if sha(png.read_bytes())!=source['png_file_sha256']: errors.append('PNG file differs')
    with Image.open(png) as image:
        if not np.array_equal(arr,np.array(image)): errors.append('Saved PNG differs from developed pixels')
    if len(result['cases'])!=2 or result['cases'][0]['profile']!='original': errors.append('Case pair differs from plan')
    for c in result['cases']:
        expected=arr
        if 'jpeg' in c:
            j=c['jpeg']; jp=RUN/j['file']; blob=jp.read_bytes()
            if (j['quality'],j['subsampling'])!=(row['jpeg_quality'],row['jpeg_subsampling']): errors.append(c['id']+': JPEG recipe differs from plan')
            regenerated=io.BytesIO()
            Image.fromarray(arr).save(regenerated,'JPEG',quality=row['jpeg_quality'],subsampling=row['jpeg_subsampling'],optimize=False,progressive=False)
            if regenerated.getvalue()!=blob: errors.append(c['id']+': regenerated JPEG bytes differ')
            if sha(blob)!=j['sha256'] or len(blob)!=j['bytes']: errors.append(c['id']+': JPEG bytes differ')
            with Image.open(io.BytesIO(blob)) as im:
                im.load()
                if {str(k):v for k,v in im.quantization.items()}!=j['qtables'] or [list(x) for x in im.layer]!=j['components']: errors.append(c['id']+': JPEG tables/sampling differ')
                expected=np.array(im.convert('RGB'))
        if sha(expected.tobytes())!=c['input_sha256']: errors.append(c['id']+': reference pixels differ')
        for codec in CODECS:
            streams+=1; z=c['codecs'].get(codec,{})
            try:
                stream=(RUN/'streams'/c['key']/(codec+'.bin')).read_bytes()
                assert sha(stream)==z['stream_sha256'] and len(stream)==z['bytes']
                dec=b.decode(codec,stream)
                assert z['exact'] and dec.dtype==expected.dtype and dec.shape==expected.shape and np.array_equal(expected,dec)
                assert sha(dec.tobytes())==z['decoded_sha256']==c['input_sha256']
            except Exception as e: errors.append(c['id']+': '+codec+' '+repr(e))
    return dict(source_id=row['File'],auditor_pid=os.getpid(),actual_camera=actual_camera,streams=streams,cases=len(result['cases']),errors=errors,sensor_sha256=source['sensor_sha256'],prepared_sha256=source['prepared_sha256'],raw_file_sha256=source['raw_file_sha256'])

def finish_audit(results):
    unique={key:len({r.get(key) for r in results}) for key in ['raw_file_sha256','sensor_sha256','prepared_sha256']}
    result=dict(sources=len(results),cases=sum(r.get('cases',0) for r in results),streams=sum(r['streams'] for r in results),unique=unique,errors=[r for r in results if r['errors']],records=results)
    result['passed']=not result['errors'] and result['sources']==2016 and result['cases']==4032 and result['streams']==32256 and min(unique.values())>=2000
    writejson(RUN/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    assert result['passed']

def audit(workers,watch=False):
    verify_revision(); rows=readjson(RUN/('qualified-plan.json' if (RUN/'qualified-plan.json').exists() else 'plan.json')); results=[]
    if not watch:
        with cf.ProcessPoolExecutor(workers,initializer=init_worker) as pool:
            for n,r in enumerate(pool.map(audit_source,rows),1):
                results.append(r)
                if n%64==0: print(json.dumps(dict(audited=n,total=len(rows),errors=sum(len(x['errors']) for x in results))),flush=True)
    else:
        submitted=set()
        for row in rows:
            record=RUN/'audit-records'/(row['File']+'.json')
            if record.exists():
                results.append(readjson(record)); submitted.add(row['File'])
        with cf.ProcessPoolExecutor(workers,initializer=init_worker) as pool:
            pending={}
            while len(results)<len(rows):
                for row in rows:
                    if row['File'] not in submitted and (RUN/'source-results'/(row['File']+'.json')).exists():
                        pending[pool.submit(audit_source,row)]=row; submitted.add(row['File'])
                if not pending:
                    time.sleep(1); continue
                finished,_=cf.wait(pending,timeout=1,return_when=cf.FIRST_COMPLETED)
                for future in finished:
                    row=pending.pop(future)
                    try: result=future.result()
                    except Exception as e: result=dict(source_id=row['File'],streams=0,errors=[repr(e)],traceback=traceback.format_exc())
                    writejson(RUN/'audit-records'/(row['File']+'.json'),result); results.append(result)
                    if len(results)%64==0 or result['errors']:
                        print(json.dumps(dict(audited=len(results),total=len(rows),errors=sum(len(x['errors']) for x in results))),flush=True)
    finish_audit(results)

def timings():
    verify_revision(); init_worker(); rows=readjson(RUN/('qualified-plan.json' if (RUN/'qualified-plan.json').exists() else 'plan.json'))[:12]
    dest=RUN/'timings.jsonl'; done=set()
    if dest.exists(): done={json.loads(line)['id'] for line in dest.read_text().splitlines()}
    with dest.open('a',encoding='utf-8') as out:
        for row in rows:
            for c in readjson(RUN/'source-results'/(row['File']+'.json'))['cases']:
                if c['id'] in done: continue
                p=RUN/c['jpeg']['file'] if 'jpeg' in c else RUN/'prepared'/(row['File']+'.png')
                with Image.open(p) as im: arr=np.array(im.convert('RGB'))
                rec=dict(id=c['id'],pixels=arr.shape[0]*arr.shape[1],profile=c['profile'],codecs={})
                for codec in CODECS:
                    stream=b.encode(codec,arr); dec=b.decode(codec,stream)
                    assert np.array_equal(arr,dec) and sha(stream)==c['codecs'][codec]['stream_sha256']
                    rec['codecs'][codec]=dict(encode_s=[],decode_s=[],bytes=len(stream))
                for repetition in range(3):
                    for codec in CODECS[repetition:]+CODECS[:repetition]:
                        z=rec['codecs'][codec]
                        t=time.perf_counter(); stream=b.encode(codec,arr); z['encode_s'].append(time.perf_counter()-t)
                        t=time.perf_counter(); dec=b.decode(codec,stream); z['decode_s'].append(time.perf_counter()-t)
                        assert np.array_equal(arr,dec) and arr.dtype==dec.dtype and sha(stream)==c['codecs'][codec]['stream_sha256']
                for z in rec['codecs'].values():
                    z.update(encode_median_s=statistics.median(z['encode_s']),decode_median_s=statistics.median(z['decode_s']),exact=True)
                out.write(json.dumps(rec)+'\n'); out.flush(); print('TIMED '+c['id'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['plan','run','audit','audit-watch','timings']); parser.add_argument('--workers',type=int,default=12); parser.add_argument('--limit',type=int); args=parser.parse_args()
    if args.action=='plan': plan()
    elif args.action=='run': run(args.workers,args.limit)
    elif args.action in ('audit','audit-watch'): audit(args.workers,args.action=='audit-watch')
    else: timings()
