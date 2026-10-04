import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
import io, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'results-2026-10-04-raise2000'
sys.path.insert(0,str(ROOT/'vendor/Tsiro'))
import numpy as np
import imagecodecs as ic
import tsiro
from PIL import Image
CODECS=['Tsiro','PNG','WebP','JXL3','JXL7','QOI','JPEGLS']


def writejson(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2),encoding='utf-8')

def encode(codec,a):
    if codec=='Tsiro': return tsiro.compress_pixels(a)
    if codec=='PNG':
        b=io.BytesIO();Image.fromarray(a).save(b,'PNG',optimize=True,compress_level=9);return b.getvalue()
    if codec=='WebP':
        b=io.BytesIO();Image.fromarray(a).save(b,'WEBP',lossless=True,quality=100,method=6,exact=True);return b.getvalue()
    if codec.startswith('JXL'): return ic.jpegxl_encode(a,lossless=True,effort=int(codec[3:]),numthreads=1)
    if codec=='QOI': return ic.qoi_encode(a)
    if codec=='JPEGLS': return ic.jpegls_encode(a,level=0)
    raise ValueError(codec)

def decode(codec,b):
    if codec=='Tsiro': return tsiro.decompress_any(b)
    if codec in ('PNG','WebP'):
        with Image.open(io.BytesIO(b)) as im:return np.array(im.convert('RGB'))
    if codec.startswith('JXL'):return ic.jpegxl_decode(b,numthreads=1)
    if codec=='QOI':return ic.qoi_decode(b)
    if codec=='JPEGLS':return ic.jpegls_decode(b)
    raise ValueError(codec)

def init_worker():
    scratch=RUN/'scratch'/str(os.getpid());scratch.mkdir(parents=True,exist_ok=True);os.chdir(scratch)
    a=np.array(Image.open(ROOT/'inputs/kodak/kodim01.png').convert('RGB'))[:64,:64].copy()
    t=time.perf_counter()
    for c in CODECS: b=encode(c,a);assert np.array_equal(a,decode(c,b))

    ycc,exact=tsiro._invert_ycc_nb(a,tsiro._PCR,tsiro._PCB,tsiro._PSTART,tsiro._D1LO)
    k=np.zeros((3,64,64),np.int16);qt=np.full((3,8,8),5,np.int64)
    tsiro._refine_all(k,qt,np.zeros_like(k,dtype=np.int8),a,8,8)
    ky=np.zeros((64,64),np.int16);kc=np.zeros((16,64),np.int16)
    ys,cbs,crs=tsiro._sim_planes420(ky,kc,kc,qt[0],qt[1],64,64)
    tsiro._refine420(ky,kc.copy(),kc.copy(),qt[0],qt[1],np.zeros_like(ky,dtype=np.int8),np.zeros_like(kc,dtype=np.int8),np.zeros_like(kc,dtype=np.int8),a,ys,cbs,crs)
    writejson(scratch/'warmup.json',dict(seconds=time.perf_counter()-t,pid=os.getpid()))
