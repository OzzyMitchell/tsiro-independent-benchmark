import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

const root=path.resolve(import.meta.dirname,'..');
const dir=path.join(root,'results-2026-10-04-raise2000');
const read=async name=>JSON.parse(await fs.readFile(path.join(dir,name),'utf8'));
const verification=await read('verification.json');
assert.equal(verification.passed,true);
const audit=new Map(verification.records.map(r=>[r.source_id,r]));
const plan=await read('qualified-plan.json');
const protocol=await read('protocol.json');
const sources=await Promise.all(plan.map(r=>read(`source-results/${r.File}.json`)));
assert.equal(sources.length,2016);
assert.equal(new Set(sources.map(r=>r.source_id)).size,2016);
for(const r of sources){
  assert.equal(r.cases.length,2);
  assert.equal(audit.get(r.source_id).errors.length,0);
}
const codecs=['Tsiro','PNG','WebP','JXL3','JXL7','JXL9','QOI','JPEGLS'];
const headers=['source_id','case_id','profile','codec','encoded_bytes','bits_per_pixel','width','height','channels','bit_depth','raw_rgb_bytes','pixel_exact','fresh_disk_audit_passed','tsiro_mode','encode_s_bulk','decode_s_bulk','timing_scope','input_kind','jpeg_generation_count','input_pixel_sha256','decoded_pixel_sha256','stream_sha256','stream_path','dataset','source_rank','camera','source_url','source_raw_file_sha256','source_raw_bytes','sensor_sha256','sensor_compression_tag','sensor_bit_depth','raw_quality_label','prepared_pixel_sha256','prepared_png','source_record','source_conversion_chain','jpeg_file','jpeg_file_sha256','jpeg_file_bytes','jpeg_quality','jpeg_chroma','jpeg_subsampling_code','jpeg_encoder','jpeg_decoder','jpeg_quantization_tables_row_major_json','jpeg_components_id_h_v_qtable_json','jpeg_progressive','jpeg_optimize','tsiro_revision','codec_settings'];
const settings={...protocol.settings};
for(const c of ['JXL3','JXL7','JXL9']) settings[c]=`imagecodecs JPEG XL lossless=True effort=${c.slice(3)} numthreads=1`;
const quote=v=>{
  const s=v==null?'':typeof v==='boolean'?(v?'True':'False'):String(v);
  return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;
};
const manifests=[];
for(const output of [
  {name:'JPEG recompression',file:'jpeg-recompression-only.csv',jpeg:true},
  {name:'Tsiro only',file:'tsiro-only-compression.csv',jpeg:false},
]){
  const rows=[];
  for(const result of sources){
    const s=result.source;
    const c=result.cases.find(c=>(c.profile!=='original')===output.jpeg);
    assert(c);
    const j=c.jpeg;
    assert.equal(Boolean(j),output.jpeg);
    const sensor=s.ifds.find(p=>p.photometric===32803);
    for(const codec of codecs){
      const z=c.codecs[codec];
      assert.equal(z.exact,true);
      assert.equal(z.decoded_sha256,c.input_sha256);
      const record={
        source_id:result.source_id,case_id:c.id,profile:c.profile,codec,
        encoded_bytes:z.bytes,bits_per_pixel:8*z.bytes/c.width/c.height,
        width:c.width,height:c.height,channels:3,bit_depth:8,raw_rgb_bytes:c.raw_bytes,
        pixel_exact:z.exact,fresh_disk_audit_passed:true,tsiro_mode:codec==='Tsiro'?z.mode:null,
        encode_s_bulk:z.encode_s_bulk,decode_s_bulk:z.decode_s_bulk,timing_scope:'parallel bulk; not a speed ranking',
        input_kind:j?'jpeg_decoded_rgb':'raw_developed_rgb',jpeg_generation_count:j?1:0,
        input_pixel_sha256:c.input_sha256,decoded_pixel_sha256:z.decoded_sha256,stream_sha256:z.stream_sha256,
        stream_path:`streams/${c.key}/${codec}.bin`,dataset:'RAISE',source_rank:result.rank,camera:result.camera,
        source_url:s.source_url,source_raw_file_sha256:s.raw_file_sha256,source_raw_bytes:s.raw_file_bytes,
        sensor_sha256:s.sensor_sha256,sensor_compression_tag:sensor.compression,sensor_bit_depth:sensor.bits_per_sample,
        raw_quality_label:s.raw_quality,prepared_pixel_sha256:s.prepared_sha256,prepared_png:`prepared/${result.source_id}.png`,
        source_record:`source-results/${result.source_id}.json`,
        source_conversion_chain:'Nikon CFA sensor RAW > LibRaw AHD sRGB8 > LANCZOS 1024-edge resize > centered 16-alignment crop'+(j?' > controlled JPEG > Pillow RGB8 decode':''),
        jpeg_file:j?.file,jpeg_file_sha256:j?.sha256,jpeg_file_bytes:j?.bytes,jpeg_quality:j?.quality,
        jpeg_chroma:j?.chroma,jpeg_subsampling_code:j?.subsampling,jpeg_encoder:j?.encoder,jpeg_decoder:j?.decoder,
        jpeg_quantization_tables_row_major_json:j?JSON.stringify(j.qtables):null,
        jpeg_components_id_h_v_qtable_json:j?JSON.stringify(j.components):null,
        jpeg_progressive:j?.progressive,jpeg_optimize:j?.optimize,
        tsiro_revision:protocol.tsiro_revision,codec_settings:settings[codec],
      };
      rows.push(headers.map(h=>record[h]??null));
    }
  }
  assert.equal(rows.length,16128);
  const values=[headers,...rows];
  const csv='\uFEFF'+values.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n';
  await fs.writeFile(path.join(dir,output.file),csv,'utf8');
  assert.equal(await fs.readFile(path.join(dir,output.file),'utf8'),csv);
  manifests.push({file:output.file,source_images:2016,cases:2016,codec_rows:rows.length,columns:headers.length,sha256:createHash('sha256').update(csv).digest('hex')});
}
await fs.writeFile(path.join(dir,'csv-export.json'),JSON.stringify(manifests,null,2),'utf8');
console.log(JSON.stringify(manifests,null,2));
