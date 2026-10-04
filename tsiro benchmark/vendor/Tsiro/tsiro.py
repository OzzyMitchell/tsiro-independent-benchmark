import sys, os, time, zlib
import numpy as np
from numba import njit
from PIL import Image
import jpeglib

TILE = 512
MIN_BLK = 8
FLAT_VAR = 12.0
SCALE_BITS = 14
RANS_L = np.uint64(1 << 23)
MAGIC = b"TSO1"
MAGICJ = b"TSOJ"
MAGICR = b"TSOR"
RAW_TB = np.array([1, 3, 7, 15, 31, 63], np.int64)
RAW_Q = 6
RAW_SPAN = 2 * RAW_Q + 1
RAW_NCTX = (RAW_Q + 1) * RAW_SPAN * RAW_SPAN
KMAX = 20
DC_DEFAULT = (128, 0, 0)
NCTX = 8
CTX_BOUNDS = np.array([1, 3, 6, 12, 24, 48, 96], np.int64)
MED_PEN = 3
MEAN_PEN = 21
WED_PEN = 55
COST2 = np.round(2 * (2 * np.log2(1 + np.arange(1024)) + 1)).astype(np.int64)

def _wedge_masks():
    m = np.zeros((32, 8, 8), np.uint8)
    i = 0
    for ai in range(16):
        nx = np.cos(np.pi * 2 * ai / 16); ny = np.sin(np.pi * 2 * ai / 16)
        for off in (0.5, 2.5):
            for y in range(8):
                for x in range(8):
                    m[i, y, x] = 1 if (x - 3.5) * nx + (y - 3.5) * ny > off else 0
            i += 1
    return m

WMASKS = _wedge_masks()

@njit(cache=True)
def _rans_encode(symbols, freq, cum, scale_bits):
    n = symbols.shape[0]
    out = np.empty(n * 2 + 1024, np.uint8)
    pos = out.shape[0]
    x = RANS_L
    for i in range(n - 1, -1, -1):
        s = symbols[i]
        f = np.uint64(freq[s]); c = np.uint64(cum[s])
        x_max = ((RANS_L >> np.uint64(scale_bits)) << np.uint64(8)) * f
        while x >= x_max:
            pos -= 1; out[pos] = np.uint8(x & np.uint64(0xFF)); x >>= np.uint64(8)
        x = ((x // f) << np.uint64(scale_bits)) + (x % f) + c
    for _ in range(4):
        pos -= 1; out[pos] = np.uint8(x & np.uint64(0xFF)); x >>= np.uint64(8)
    return out[pos:].copy()

@njit(cache=True)
def _rans_decode(data, n, freq, cum, sym_of_slot, scale_bits):
    mask = np.uint64((1 << scale_bits) - 1)
    pos = 0; x = np.uint64(0)
    for _ in range(4):
        x = (x << np.uint64(8)) | np.uint64(data[pos]); pos += 1
    out = np.empty(n, np.int32)
    for i in range(n):
        slot = x & mask
        s = sym_of_slot[slot]
        out[i] = s
        f = np.uint64(freq[s]); c = np.uint64(cum[s])
        x = f * (x >> np.uint64(scale_bits)) + slot - c
        while x < RANS_L:
            x = (x << np.uint64(8)) | np.uint64(data[pos]); pos += 1
    return out

def _normalize_freqs(counts, scale_bits):
    M = 1 << scale_bits
    counts = counts.astype(np.float64)
    total = counts.sum()
    freq = np.zeros(counts.shape[0], np.uint32)
    if total == 0:
        return freq
    scaled = counts / total * M
    f = np.floor(scaled).astype(np.int64)
    used = counts > 0
    f[used & (f < 1)] = 1
    diff = M - int(f.sum())
    if diff > 0:
        order = np.argsort(-(scaled - f))
        idx = 0
        while diff > 0:
            j = order[idx % order.shape[0]]
            if counts[j] > 0:
                f[j] += 1; diff -= 1
            idx += 1
    elif diff < 0:
        order = np.argsort(-f.astype(np.float64))
        idx = 0
        while diff < 0:
            j = order[idx % order.shape[0]]
            if f[j] > 1:
                f[j] -= 1; diff += 1
            idx += 1
    return f.astype(np.uint32)

def _build_tables(freq, scale_bits):
    cum = np.zeros(freq.shape[0], np.uint32)
    np.cumsum(freq[:-1], out=cum[1:])
    sym_of_slot = np.zeros(1 << scale_bits, np.int32)
    for s in range(freq.shape[0]):
        if freq[s] > 0:
            sym_of_slot[cum[s]:cum[s] + freq[s]] = s
    return cum, sym_of_slot

def rans_compress(symbols, nsym):
    counts = np.bincount(symbols, minlength=nsym).astype(np.uint32)
    freq = _normalize_freqs(counts, SCALE_BITS)
    cum, _ = _build_tables(freq, SCALE_BITS)
    payload = _rans_encode(symbols.astype(np.int32), freq.astype(np.uint64), cum.astype(np.uint64), SCALE_BITS)
    return payload, freq

def rans_decompress(payload, n, freq):
    cum, sym_of_slot = _build_tables(freq, SCALE_BITS)
    return _rans_decode(payload, n, freq.astype(np.uint64), cum.astype(np.uint64), sym_of_slot, SCALE_BITS)

def _u32(v):
    return int(v).to_bytes(4, "little")

def _ru32(buf, off):
    return int.from_bytes(buf[off:off + 4], "little"), off + 4

def _write_freq(freq, nsym):
    nz = np.nonzero(freq[:nsym])[0]
    buf = bytearray(len(nz).to_bytes(4, "little"))
    for s in nz:
        buf += int(s).to_bytes(2, "little") + int(freq[s]).to_bytes(2, "little")
    return bytes(buf)

def _read_freq(data, off, nsym):
    cnt, off = _ru32(data, off)
    freq = np.zeros(nsym, np.uint32)
    for _ in range(cnt):
        s = int.from_bytes(data[off:off + 2], "little"); off += 2
        f = int.from_bytes(data[off:off + 2], "little"); off += 2
        freq[s] = f
    return freq, off

def _put_arr(out, arr):
    arr = np.asarray(arr).astype(np.int64)
    out += _u32(arr.size)
    if arr.size == 0:
        return
    lo = int(arr.min())
    syms = (arr - lo).astype(np.int32)
    nsym = int(syms.max()) + 1
    out += lo.to_bytes(4, "little", signed=True) + _u32(nsym)
    payload, freq = rans_compress(syms, nsym)
    out += _write_freq(freq, nsym) + _u32(len(payload)) + payload.tobytes()

def _get_arr(data, off):
    n, off = _ru32(data, off)
    if n == 0:
        return np.zeros(0, np.int32), off
    lo = int.from_bytes(data[off:off + 4], "little", signed=True); off += 4
    nsym, off = _ru32(data, off)
    freq, off = _read_freq(data, off, nsym)
    plen, off = _ru32(data, off)
    payload = np.frombuffer(data, np.uint8, plen, off); off += plen
    return rans_decompress(payload, n, freq) + lo, off

@njit(cache=True)
def _med(a, b, c):
    if c >= a and c >= b:
        return b if a > b else a
    elif c <= a and c <= b:
        return a if a > b else b
    else:
        return a + b - c

@njit(cache=True)
def _med_cost_map(tile, dcd, cost2):
    H, W = tile.shape
    cm = np.zeros((H + 1, W + 1), np.int64)
    for r in range(H):
        rowsum = 0
        for c in range(W):
            a = tile[r, c - 1] if c > 0 else (tile[r - 1, c] if r > 0 else dcd)
            b = tile[r - 1, c] if r > 0 else (tile[r, c - 1] if c > 0 else dcd)
            cc = tile[r - 1, c - 1] if (r > 0 and c > 0) else b
            d = tile[r, c] - _med(a, b, cc)
            rowsum += cost2[d if d >= 0 else -d]
            cm[r + 1, c + 1] = cm[r, c + 1] + rowsum
    return cm

@njit(cache=True)
def _leaf_stats(tile, leaves, cm, masks, cost2):
    n = leaves.shape[0]
    dcost = np.empty(n, np.int64)
    mcost = np.empty(n, np.int64); mval = np.empty(n, np.int32)
    wcost = np.full(n, np.int64(1) << 60, np.int64)
    widx = np.zeros(n, np.int32); w0 = np.zeros(n, np.int32); w1 = np.zeros(n, np.int32)
    for i in range(n):
        r0, c0, h, w = leaves[i, 0], leaves[i, 1], leaves[i, 2], leaves[i, 3]
        dcost[i] = cm[r0 + h, c0 + w] - cm[r0, c0 + w] - cm[r0 + h, c0] + cm[r0, c0]
        s = 0
        for y in range(h):
            for x in range(w):
                s += tile[r0 + y, c0 + x]
        mu = int(np.floor(s / (h * w) + 0.5))
        cst = 0
        for y in range(h):
            for x in range(w):
                d = tile[r0 + y, c0 + x] - mu
                cst += cost2[d if d >= 0 else -d]
        mcost[i] = cst; mval[i] = mu
        if h == 8 and w == 8:
            best = np.int64(1) << 60; bw = 0; b0 = 0; b1 = 0
            for wi in range(masks.shape[0]):
                s0 = 0; s1 = 0; n1 = 0
                for y in range(8):
                    for x in range(8):
                        if masks[wi, y, x] == 1:
                            s1 += tile[r0 + y, c0 + x]; n1 += 1
                        else:
                            s0 += tile[r0 + y, c0 + x]
                n0 = 64 - n1
                if n0 == 0 or n1 == 0:
                    continue
                mu0 = int(np.floor(s0 / n0 + 0.5)); mu1 = int(np.floor(s1 / n1 + 0.5))
                cst = 0
                for y in range(8):
                    for x in range(8):
                        p = mu1 if masks[wi, y, x] == 1 else mu0
                        d = tile[r0 + y, c0 + x] - p
                        cst += cost2[d if d >= 0 else -d]
                if cst < best:
                    best = cst; bw = wi; b0 = mu0; b1 = mu1
            wcost[i] = best; widx[i] = bw; w0[i] = b0; w1[i] = b1
    return dcost, mcost, mval, wcost, widx, w0, w1

@njit(cache=True)
def _fill_maps(leaves, lmode, mval, widx, w0, w1, masks, modes, means):
    for i in range(leaves.shape[0]):
        r0, c0, h, w = leaves[i, 0], leaves[i, 1], leaves[i, 2], leaves[i, 3]
        if lmode[i] == 1:
            for y in range(h):
                for x in range(w):
                    modes[r0 + y, c0 + x] = 1
        elif lmode[i] == 0:
            for y in range(h):
                for x in range(w):
                    modes[r0 + y, c0 + x] = 0
                    means[r0 + y, c0 + x] = mval[i]
        else:
            for y in range(8):
                for x in range(8):
                    modes[r0 + y, c0 + x] = 0
                    means[r0 + y, c0 + x] = w1[i] if masks[widx[i], y, x] == 1 else w0[i]

@njit(cache=True)
def _expand_streams(lmode, lmeans, lwidx, mstart, wstart):
    n = lmode.shape[0]
    mval = np.zeros(n, np.int32); widx = np.zeros(n, np.int32)
    w0 = np.zeros(n, np.int32); w1 = np.zeros(n, np.int32)
    mi = mstart; wi = wstart
    for i in range(n):
        if lmode[i] == 0:
            mval[i] = lmeans[mi]; mi += 1
        elif lmode[i] == 2:
            widx[i] = lwidx[wi]; wi += 1
            w0[i] = lmeans[mi]; w1[i] = lmeans[mi + 1]; mi += 2
    return mval, widx, w0, w1, mi, wi

@njit(cache=True)
def _enc_res_ctx(tile, modes, means, dcd, bounds):
    H, W = tile.shape
    res = np.empty(H * W, np.int32); ctx = np.empty(H * W, np.uint8)
    i = 0
    for r in range(H):
        for c in range(W):
            a = tile[r, c - 1] if c > 0 else (tile[r - 1, c] if r > 0 else dcd)
            b = tile[r - 1, c] if r > 0 else (tile[r, c - 1] if c > 0 else dcd)
            cc = tile[r - 1, c - 1] if (r > 0 and c > 0) else b
            act = abs(a - cc) + abs(b - cc) + abs(a - b)
            k = 0
            while k < 7 and act >= bounds[k]:
                k += 1
            pred = means[r, c] if modes[r, c] == 0 else _med(a, b, cc)
            res[i] = tile[r, c] - pred; ctx[i] = k; i += 1
    return res, ctx

@njit(cache=True)
def _dec_res_ctx(modes, means, dcd, bounds, cdata, cbase, ccur):
    H, W = modes.shape
    out = np.empty((H, W), np.int32)
    for r in range(H):
        for c in range(W):
            a = out[r, c - 1] if c > 0 else (out[r - 1, c] if r > 0 else dcd)
            b = out[r - 1, c] if r > 0 else (out[r, c - 1] if c > 0 else dcd)
            cc = out[r - 1, c - 1] if (r > 0 and c > 0) else b
            act = abs(a - cc) + abs(b - cc) + abs(a - b)
            k = 0
            while k < 7 and act >= bounds[k]:
                k += 1
            pred = means[r, c] if modes[r, c] == 0 else _med(a, b, cc)
            out[r, c] = pred + cdata[cbase[k] + ccur[k]]
            ccur[k] += 1
    return out

def _leaves_enc(tile, r0, c0, h, w, splits, leaves):
    forced = h <= MIN_BLK and w <= MIN_BLK
    if not forced:
        flat = tile[r0:r0 + h, c0:c0 + w].var() <= FLAT_VAR
        splits.append(0 if flat else 1)
        if not flat:
            hh = (h + 1) // 2; ww = (w + 1) // 2
            for dr, dh in ((0, hh), (hh, h - hh)):
                for dc, dw in ((0, ww), (ww, w - ww)):
                    if dh > 0 and dw > 0:
                        _leaves_enc(tile, r0 + dr, c0 + dc, dh, dw, splits, leaves)
            return
    leaves.append((r0, c0, h, w))

def _leaves_dec(r0, c0, h, w, splits, si, leaves):
    forced = h <= MIN_BLK and w <= MIN_BLK
    if not forced:
        s = splits[si[0]]; si[0] += 1
        if s == 1:
            hh = (h + 1) // 2; ww = (w + 1) // 2
            for dr, dh in ((0, hh), (hh, h - hh)):
                for dc, dw in ((0, ww), (ww, w - ww)):
                    if dh > 0 and dw > 0:
                        _leaves_dec(r0 + dr, c0 + dc, dh, dw, splits, si, leaves)
            return
    leaves.append((r0, c0, h, w))

def _iter_tiles(H, W):
    for r0 in range(0, H, TILE):
        for c0 in range(0, W, TILE):
            yield r0, c0, min(TILE, H - r0), min(TILE, W - c0)

def rgb_to_ycocg(img):
    R = img[:, :, 0].astype(np.int32); G = img[:, :, 1].astype(np.int32); B = img[:, :, 2].astype(np.int32)
    Co = R - B
    t = B + (Co >> 1)
    Cg = G - t
    Y = t + (Cg >> 1)
    return [Y, Co, Cg]

def ycocg_to_rgb(planes):
    Y, Co, Cg = planes
    t = Y - (Cg >> 1)
    G = Cg + t
    B = t - (Co >> 1)
    R = Co + B
    return np.stack([R, G, B], axis=2).astype(np.uint8)

def _encode_channel(chan, dcd):
    H, W = chan.shape
    splits = []; lmodes_l = []; lmeans_l = []; lwidx_l = []
    res_by_ctx = [[] for _ in range(NCTX)]
    for r0, c0, h, w in _iter_tiles(H, W):
        tile = chan[r0:r0 + h, c0:c0 + w].astype(np.int32)
        leaves = []
        _leaves_enc(tile, 0, 0, h, w, splits, leaves)
        lv = np.array(leaves, np.int32).reshape(-1, 4)
        cm = _med_cost_map(tile, dcd, COST2)
        dcost, mcost, mval, wcost, widx, w0, w1 = _leaf_stats(tile, lv, cm, WMASKS, COST2)
        cost = np.stack([mcost + MEAN_PEN, dcost + MED_PEN, wcost + WED_PEN])
        choice = np.argmin(cost, axis=0).astype(np.int32)
        modes = np.zeros((h, w), np.uint8); means = np.zeros((h, w), np.int32)
        _fill_maps(lv, choice, mval, widx, w0, w1, WMASKS, modes, means)
        res, ctx = _enc_res_ctx(tile, modes, means, dcd, CTX_BOUNDS)
        for k in range(NCTX):
            res_by_ctx[k].append(res[ctx == k])
        lmodes_l.append(choice)
        for i in range(lv.shape[0]):
            m = choice[i]
            if m == 0:
                lmeans_l.append(mval[i])
            elif m == 2:
                lwidx_l.append(widx[i]); lmeans_l.append(w0[i]); lmeans_l.append(w1[i])
    resctx = [np.concatenate(res_by_ctx[k]) if res_by_ctx[k] else np.zeros(0, np.int32) for k in range(NCTX)]
    return (np.array(splits, np.int32), np.concatenate(lmodes_l),
            np.array(lmeans_l, np.int32), np.array(lwidx_l, np.int32), resctx)

def compress_image(img):
    H, W, C = img.shape
    out = bytearray(MAGIC + _u32(H) + _u32(W) + bytes([C]))
    planes = rgb_to_ycocg(img) if C == 3 else [img[:, :, i].astype(np.int32) for i in range(C)]
    for ch in range(C):
        splits, lmodes, lmeans, lwidx, resctx = _encode_channel(planes[ch], DC_DEFAULT[ch])
        for arr in (splits, lmodes, lmeans, lwidx, *resctx):
            _put_arr(out, arr)
    return bytes(out)

def decompress_image(data):
    assert data[:4] == MAGIC
    off = 4
    H, off = _ru32(data, off)
    W, off = _ru32(data, off)
    C = data[off]; off += 1
    planes = []
    for ch in range(C):
        splits, off = _get_arr(data, off)
        lmodes, off = _get_arr(data, off)
        lmeans, off = _get_arr(data, off)
        lwidx, off = _get_arr(data, off)
        rc = []
        for k in range(NCTX):
            a, off = _get_arr(data, off)
            rc.append(a.astype(np.int32))
        cbase = np.zeros(NCTX, np.int64)
        for k in range(1, NCTX):
            cbase[k] = cbase[k - 1] + rc[k - 1].size
        cdata = np.concatenate(rc) if any(a.size for a in rc) else np.zeros(1, np.int32)
        ccur = np.zeros(NCTX, np.int64)
        plane = np.empty((H, W), np.int32)
        si = [0]; mcur = 0; wcur = 0; lcur = 0
        dcd = DC_DEFAULT[ch]
        for r0, c0, h, w in _iter_tiles(H, W):
            leaves = []
            _leaves_dec(0, 0, h, w, splits, si, leaves)
            lv = np.array(leaves, np.int32).reshape(-1, 4)
            n = lv.shape[0]
            lm = lmodes[lcur:lcur + n].astype(np.int32); lcur += n
            mval, widx, w0, w1, mcur, wcur = _expand_streams(lm, lmeans.astype(np.int32), lwidx.astype(np.int32), mcur, wcur)
            modes = np.zeros((h, w), np.uint8); means = np.zeros((h, w), np.int32)
            _fill_maps(lv, lm, mval, widx, w0, w1, WMASKS, modes, means)
            plane[r0:r0 + h, c0:c0 + w] = _dec_res_ctx(modes, means, dcd, CTX_BOUNDS, cdata, cbase, ccur)
        planes.append(plane)
    if C == 3:
        return ycocg_to_rgb(planes)
    return np.stack(planes, axis=2).astype(np.uint8)

def _dc_med_res(dc):
    a = np.zeros_like(dc); a[:, 1:] = dc[:, :-1]
    b = np.zeros_like(dc); b[1:, :] = dc[:-1, :]
    cc = np.zeros_like(dc); cc[1:, 1:] = dc[:-1, :-1]
    mx = np.maximum(a, b); mn = np.minimum(a, b)
    med = np.where(cc >= mx, mn, np.where(cc <= mn, mx, a + b - cc))
    return (dc - med).reshape(-1)

@njit(cache=True)
def _dc_unmed(res, nbr, nbc):
    dc = np.zeros((nbr, nbc), np.int32)
    for r in range(nbr):
        for c in range(nbc):
            a = dc[r, c - 1] if c > 0 else 0
            b = dc[r - 1, c] if r > 0 else 0
            cc = dc[r - 1, c - 1] if (r > 0 and c > 0) else 0
            dc[r, c] = res[r * nbc + c] + _med(a, b, cc)
    return dc.reshape(-1)

def _extract_markers(raw):
    segs = bytearray(); i = 2
    while i + 4 <= len(raw):
        if raw[i] != 0xFF:
            break
        m = raw[i + 1]
        if m == 0xDA:
            break
        L = int.from_bytes(raw[i + 2:i + 4], "big")
        if 0xE0 <= m <= 0xEF or m == 0xFE:
            segs += raw[i:i + 2 + L]
        i += 2 + L
    return bytes(segs)

def _dct_blob(planes, qt, qmap, markers):
    out = bytearray(MAGICJ)
    mz = zlib.compress(markers, 9)
    out += _u32(len(mz)) + mz
    out += bytes([len(planes), qt.shape[0]]) + qt.astype("<u2").tobytes() + bytes(int(q) for q in qmap)
    for P in planes:
        out += _u32(P.shape[0]) + _u32(P.shape[1])
        C = P.reshape(-1, 64).astype(np.int32)
        _put_arr(out, _dc_med_res(P[:, :, 0, 0].astype(np.int32)))
        for k in range(1, 64):
            _put_arr(out, C[:, k])
    return bytes(out)

def compress_jpeg(path):
    raw = open(path, "rb").read()
    d = jpeglib.read_dct(path)
    planes = [p for p in (d.Y, d.Cb, d.Cr) if p is not None]
    qmap = list(getattr(d, "quant_tbl_no", None) if getattr(d, "quant_tbl_no", None) is not None else [0, 1, 1][:len(planes)])
    return _dct_blob(planes, d.qt.astype(np.uint16), qmap, _extract_markers(raw))

_F_RGB2YCC = np.array([[0.299, 0.587, 0.114],
                       [-0.168735892, -0.331264108, 0.5],
                       [0.5, -0.418687589, -0.081312411]])

_CB = 13; _PB = 2
_FX = (2446, 3196, 4433, 6270, 7373, 9633, 12299, 15137, 16069, 16819, 20995, 25172)

def _fix16(x):
    return int(x * 65536 + 0.5)

_CRR = np.array([(_fix16(1.40200) * (i - 128) + 32768) >> 16 for i in range(256)], np.int64)
_CBB = np.array([(_fix16(1.77200) * (i - 128) + 32768) >> 16 for i in range(256)], np.int64)
_CBG = np.array([(-_fix16(0.34414)) * (i - 128) for i in range(256)], np.int64)
_CRG = np.array([(-_fix16(0.71414)) * (i - 128) + 32768 for i in range(256)], np.int64)

def _pair_tables():
    d1 = (_CRR[:, None] - _CBB[None, :]).astype(np.int64)
    lo = int(d1.min())
    flat = d1.reshape(-1)
    order = np.argsort(flat, kind="stable")
    pair_cr = (order // 256).astype(np.int32)
    pair_cb = (order % 256).astype(np.int32)
    sortd = flat[order]
    span = int(d1.max()) - lo + 1
    start = np.zeros(span + 1, np.int64)
    np.cumsum(np.bincount((sortd - lo).astype(np.int64), minlength=span), out=start[1:])
    return pair_cr, pair_cb, start, lo

_PCR, _PCB, _PSTART, _D1LO = _pair_tables()
_CRRn, _CBBn, _CBGn, _CRGn = _CRR, _CBB, _CBG, _CRG

@njit(cache=True)
def _islow_block(coef, quant, out):
    ws = np.empty(64, np.int64)
    F0298, F0390, F0541, F0765, F0899, F1175, F1501, F1847, F1961, F2053, F2562, F3072 = _FX
    for c in range(8):
        z2 = np.int64(coef[2 * 8 + c]) * quant[2, c]
        z3 = np.int64(coef[6 * 8 + c]) * quant[6, c]
        z1 = (z2 + z3) * F0541
        t2 = z1 + z3 * (-F1847)
        t3 = z1 + z2 * F0765
        z2 = np.int64(coef[0 * 8 + c]) * quant[0, c]
        z3 = np.int64(coef[4 * 8 + c]) * quant[4, c]
        t0 = (z2 + z3) << _CB
        t1 = (z2 - z3) << _CB
        t10 = t0 + t3; t13 = t0 - t3; t11 = t1 + t2; t12 = t1 - t2
        t0 = np.int64(coef[7 * 8 + c]) * quant[7, c]
        t1 = np.int64(coef[5 * 8 + c]) * quant[5, c]
        t2 = np.int64(coef[3 * 8 + c]) * quant[3, c]
        t3 = np.int64(coef[1 * 8 + c]) * quant[1, c]
        z1 = t0 + t3; z2 = t1 + t2; z3 = t0 + t2; z4 = t1 + t3
        z5 = (z3 + z4) * F1175
        t0 = t0 * F0298; t1 = t1 * F2053; t2 = t2 * F3072; t3 = t3 * F1501
        z1 = z1 * (-F0899); z2 = z2 * (-F2562); z3 = z3 * (-F1961); z4 = z4 * (-F0390)
        z3 += z5; z4 += z5
        t0 += z1 + z3; t1 += z2 + z4; t2 += z2 + z3; t3 += z1 + z4
        sh = _CB - _PB; half = np.int64(1) << (sh - 1)
        ws[0 * 8 + c] = (t10 + t3 + half) >> sh
        ws[7 * 8 + c] = (t10 - t3 + half) >> sh
        ws[1 * 8 + c] = (t11 + t2 + half) >> sh
        ws[6 * 8 + c] = (t11 - t2 + half) >> sh
        ws[2 * 8 + c] = (t12 + t1 + half) >> sh
        ws[5 * 8 + c] = (t12 - t1 + half) >> sh
        ws[3 * 8 + c] = (t13 + t0 + half) >> sh
        ws[4 * 8 + c] = (t13 - t0 + half) >> sh
    for r in range(8):
        z2 = ws[r * 8 + 2]; z3 = ws[r * 8 + 6]
        z1 = (z2 + z3) * F0541
        t2 = z1 + z3 * (-F1847)
        t3 = z1 + z2 * F0765
        z2 = ws[r * 8 + 0]; z3 = ws[r * 8 + 4]
        t0 = (z2 + z3) << _CB
        t1 = (z2 - z3) << _CB
        t10 = t0 + t3; t13 = t0 - t3; t11 = t1 + t2; t12 = t1 - t2
        t0 = ws[r * 8 + 7]; t1 = ws[r * 8 + 5]; t2 = ws[r * 8 + 3]; t3 = ws[r * 8 + 1]
        z1 = t0 + t3; z2 = t1 + t2; z3 = t0 + t2; z4 = t1 + t3
        z5 = (z3 + z4) * F1175
        t0 = t0 * F0298; t1 = t1 * F2053; t2 = t2 * F3072; t3 = t3 * F1501
        z1 = z1 * (-F0899); z2 = z2 * (-F2562); z3 = z3 * (-F1961); z4 = z4 * (-F0390)
        z3 += z5; z4 += z5
        t0 += z1 + z3; t1 += z2 + z4; t2 += z2 + z3; t3 += z1 + z4
        sh = _CB + _PB + 3; half = np.int64(1) << (sh - 1)
        for k, v in ((0, t10 + t3), (7, t10 - t3), (1, t11 + t2), (6, t11 - t2),
                     (2, t12 + t1), (5, t12 - t1), (3, t13 + t0), (4, t13 - t0)):
            s = ((v + half) >> sh) + 128
            out[r * 8 + k] = 0 if s < 0 else (255 if s > 255 else s)

@njit(cache=True)
def _sim_bad(K, quant, b, img, r0, c0, ycc):
    for ch in range(3):
        _islow_block(K[ch, b], quant[ch], ycc[ch])
    bad = 0
    for y in range(8):
        for x in range(8):
            p = y * 8 + x
            yy = ycc[0, p]; cb = ycc[1, p]; cr = ycc[2, p]
            R = yy + _CRRn[cr]
            G = yy + ((_CBGn[cb] + _CRGn[cr]) >> 16)
            B = yy + _CBBn[cb]
            R = 0 if R < 0 else (255 if R > 255 else R)
            G = 0 if G < 0 else (255 if G > 255 else G)
            B = 0 if B < 0 else (255 if B > 255 else B)
            if R != img[r0 + y, c0 + x, 0] or G != img[r0 + y, c0 + x, 1] or B != img[r0 + y, c0 + x, 2]:
                bad += 1
    return bad

@njit(cache=True)
def _refine_all(K, quant, flip, img, nbr, nbc):
    ycc = np.empty((3, 64), np.int64)
    tot = 0
    nfix = 0
    for b in range(nbr * nbc):
        i = b // nbc; j = b % nbc
        bad = _sim_bad(K, quant, b, img, i * 8, j * 8, ycc)
        if bad == 0:
            continue
        guard = 0
        improved = True
        while bad > 0 and improved and guard < 12:
            improved = False; guard += 1
            for ch in range(3):
                for p in range(64):
                    f = flip[ch, b, p]
                    if f == 0:
                        continue
                    K[ch, b, p] += f
                    nb = _sim_bad(K, quant, b, img, i * 8, j * 8, ycc)
                    if nb < bad:
                        bad = nb; improved = True; nfix += 1
                    else:
                        K[ch, b, p] -= f
        tot += bad
    return tot, nfix

@njit(cache=True)
def _up_at(C, r, c):
    hh, ww = C.shape
    ir = r >> 1
    r1 = ir - 1 if (r & 1) == 0 else ir + 1
    if r1 < 0:
        r1 = 0
    if r1 > hh - 1:
        r1 = hh - 1
    ic = c >> 1
    c1 = ic - 1 if (c & 1) == 0 else ic + 1
    if c1 < 0:
        c1 = 0
    if c1 > ww - 1:
        c1 = ww - 1
    this = np.int64(C[ir, ic]) * 3 + np.int64(C[r1, ic])
    other = np.int64(C[ir, c1]) * 3 + np.int64(C[r1, c1])
    if (c & 1) == 0:
        return (this * 3 + other + 8) >> 4
    return (this * 3 + other + 7) >> 4

@njit(cache=True)
def _local_up_bad(C, F, i, j):
    Hf, Wf = F.shape
    bad = 0
    for r in range(max(0, 2 * i - 1), min(Hf, 2 * i + 3)):
        for c in range(max(0, 2 * j - 1), min(Wf, 2 * j + 3)):
            if _up_at(C, r, c) != F[r, c]:
                bad += 1
    return bad

@njit(cache=True)
def _solve_down(F):
    Hf, Wf = F.shape
    hh, ww = Hf // 2, Wf // 2
    Cf = np.empty((hh, ww), np.float64)
    for i in range(hh):
        for j in range(ww):
            Cf[i, j] = (F[2 * i, 2 * j] + F[2 * i, 2 * j + 1] + F[2 * i + 1, 2 * j] + F[2 * i + 1, 2 * j + 1]) * 0.25
    adj = np.empty((hh, ww), np.float64)
    for _ in range(30):
        adj[:, :] = 0.0
        for r in range(Hf):
            ir = r >> 1
            r1 = ir - 1 if (r & 1) == 0 else ir + 1
            if r1 < 0:
                r1 = 0
            if r1 > hh - 1:
                r1 = hh - 1
            for c in range(Wf):
                ic = c >> 1
                c1 = ic - 1 if (c & 1) == 0 else ic + 1
                if c1 < 0:
                    c1 = 0
                if c1 > ww - 1:
                    c1 = ww - 1
                e = F[r, c] - (9.0 * Cf[ir, ic] + 3.0 * Cf[r1, ic] + 3.0 * Cf[ir, c1] + Cf[r1, c1]) / 16.0
                adj[ir, ic] += 0.5625 * e
                adj[r1, ic] += 0.1875 * e
                adj[ir, c1] += 0.1875 * e
                adj[r1, c1] += 0.0625 * e
        Cf += 0.45 * adj * 0.25
    C = np.empty((hh, ww), np.int64)
    for i in range(hh):
        for j in range(ww):
            v = int(np.floor(Cf[i, j] + 0.5))
            C[i, j] = 0 if v < 0 else (255 if v > 255 else v)
    for _ in range(12):
        changed = False
        for i in range(hh):
            for j in range(ww):
                b0 = _local_up_bad(C, F, i, j)
                if b0 == 0:
                    continue
                bestd = 0
                bestb = b0
                for d in (1, -1, 2, -2):
                    nv = C[i, j] + d
                    if nv < 0 or nv > 255:
                        continue
                    C[i, j] = nv
                    b = _local_up_bad(C, F, i, j)
                    C[i, j] = nv - d
                    if b < bestb:
                        bestb = b
                        bestd = d
                if bestd != 0:
                    C[i, j] += bestd
                    changed = True
        if not changed:
            break
    return C

@njit(cache=True)
def _up_mismatch(C, F):
    Hf, Wf = F.shape
    m = np.zeros((Hf, Wf), np.uint8)
    for r in range(Hf):
        for c in range(Wf):
            if _up_at(C, r, c) != F[r, c]:
                m[r, c] = 1
    return m

@njit(cache=True)
def _rbad420(img, Ys, Cbs, Crs, r0, c0, r1, c1):
    bad = 0
    for r in range(r0, r1):
        for c in range(c0, c1):
            y = Ys[r, c]
            cb = _up_at(Cbs, r, c)
            cr = _up_at(Crs, r, c)
            R = y + _CRRn[cr]
            G = y + ((_CBGn[cb] + _CRGn[cr]) >> 16)
            B = y + _CBBn[cb]
            R = 0 if R < 0 else (255 if R > 255 else R)
            G = 0 if G < 0 else (255 if G > 255 else G)
            B = 0 if B < 0 else (255 if B > 255 else B)
            if R != img[r, c, 0] or G != img[r, c, 1] or B != img[r, c, 2]:
                bad += 1
    return bad

@njit(cache=True)
def _blk_into(K, quant, plane, r0, c0, tmp, old):
    for p in range(64):
        old[p] = plane[r0 + p // 8, c0 + p % 8]
    _islow_block(K, quant, tmp)
    for p in range(64):
        plane[r0 + p // 8, c0 + p % 8] = tmp[p]

@njit(cache=True)
def _blk_restore(plane, r0, c0, old):
    for p in range(64):
        plane[r0 + p // 8, c0 + p % 8] = old[p]

@njit(cache=True)
def _sim_planes420(KY, KCb, KCr, qY, qC, H, W):
    nbcY = W // 8
    nbrY = H // 8
    hh, ww = H // 2, W // 2
    Ys = np.empty((H, W), np.int64)
    Cbs = np.empty((hh, ww), np.int64)
    Crs = np.empty((hh, ww), np.int64)
    tmp = np.empty(64, np.int64)
    for bi in range(nbrY):
        for bj in range(nbcY):
            _islow_block(KY[bi * nbcY + bj], qY, tmp)
            for p in range(64):
                Ys[bi * 8 + p // 8, bj * 8 + p % 8] = tmp[p]
    nbcC = ww // 8
    for bi in range(hh // 8):
        for bj in range(nbcC):
            _islow_block(KCb[bi * nbcC + bj], qC, tmp)
            for p in range(64):
                Cbs[bi * 8 + p // 8, bj * 8 + p % 8] = tmp[p]
            _islow_block(KCr[bi * nbcC + bj], qC, tmp)
            for p in range(64):
                Crs[bi * 8 + p // 8, bj * 8 + p % 8] = tmp[p]
    return Ys, Cbs, Crs

@njit(cache=True)
def _refine420(KY, KCb, KCr, qY, qC, fY, fCb, fCr, img, Ys, Cbs, Crs):
    H, W, _ = img.shape
    nbcY = W // 8
    nbcC = W // 16
    nMbr = H // 16
    nMbc = W // 16
    tmp = np.empty(64, np.int64)
    old = np.empty(64, np.int64)
    tot = 0
    nfix = 0
    for mi in range(nMbr):
        for mj in range(nMbc):
            r0 = mi * 16
            c0 = mj * 16
            bad = _rbad420(img, Ys, Cbs, Crs, r0, c0, r0 + 16, c0 + 16)
            if bad == 0:
                continue
            guard = 0
            improved = True
            while bad > 0 and improved and guard < 10:
                improved = False
                guard += 1
                for dy in range(2):
                    for dx in range(2):
                        bi = mi * 2 + dy
                        bj = mj * 2 + dx
                        b = bi * nbcY + bj
                        for p in range(64):
                            f = fY[b, p]
                            if f == 0:
                                continue
                            KY[b, p] += f
                            _blk_into(KY[b], qY, Ys, bi * 8, bj * 8, tmp, old)
                            nb = _rbad420(img, Ys, Cbs, Crs, r0, c0, r0 + 16, c0 + 16)
                            if nb < bad:
                                bad = nb
                                improved = True
                                nfix += 1
                            else:
                                KY[b, p] -= f
                                _blk_restore(Ys, bi * 8, bj * 8, old)
                cb = mi * nbcC + mj
                rr0 = r0 - 1 if r0 > 0 else 0
                cc0 = c0 - 1 if c0 > 0 else 0
                rr1 = r0 + 17 if r0 + 17 < H else H
                cc1 = c0 + 17 if c0 + 17 < W else W
                for p in range(64):
                    f = fCb[cb, p]
                    if f == 0:
                        continue
                    b0 = _rbad420(img, Ys, Cbs, Crs, rr0, cc0, rr1, cc1)
                    KCb[cb, p] += f
                    _blk_into(KCb[cb], qC, Cbs, mi * 8, mj * 8, tmp, old)
                    nb = _rbad420(img, Ys, Cbs, Crs, rr0, cc0, rr1, cc1)
                    if nb < b0:
                        improved = True
                        nfix += 1
                        bad = _rbad420(img, Ys, Cbs, Crs, r0, c0, r0 + 16, c0 + 16)
                    else:
                        KCb[cb, p] -= f
                        _blk_restore(Cbs, mi * 8, mj * 8, old)
                for p in range(64):
                    f = fCr[cb, p]
                    if f == 0:
                        continue
                    b0 = _rbad420(img, Ys, Cbs, Crs, rr0, cc0, rr1, cc1)
                    KCr[cb, p] += f
                    _blk_into(KCr[cb], qC, Crs, mi * 8, mj * 8, tmp, old)
                    nb = _rbad420(img, Ys, Cbs, Crs, rr0, cc0, rr1, cc1)
                    if nb < b0:
                        improved = True
                        nfix += 1
                        bad = _rbad420(img, Ys, Cbs, Crs, r0, c0, r0 + 16, c0 + 16)
                    else:
                        KCr[cb, p] -= f
                        _blk_restore(Crs, mi * 8, mj * 8, old)
            tot += bad
    return tot, nfix

@njit(cache=True)
def _invert_ycc_nb(img, pair_cr, pair_cb, pstart, d1lo):
    H, W, _ = img.shape
    ycc = np.empty((H, W, 3), np.int16)
    exact = np.zeros((H, W), np.uint8)
    for r in range(H):
        for c in range(W):
            R = np.int64(img[r, c, 0]); G = np.int64(img[r, c, 1]); B = np.int64(img[r, c, 2])
            fy = 0.299 * R + 0.587 * G + 0.114 * B
            fcb = -0.168735892 * R - 0.331264108 * G + 0.5 * B + 128.0
            fcr = 0.5 * R - 0.418687589 * G - 0.081312411 * B + 128.0
            found = False
            if 0 < R < 255 and 0 < G < 255 and 0 < B < 255:
                idx = R - B - d1lo
                bd = 1e18
                by = 0; bcb = 0; bcr = 0
                for t in range(pstart[idx], pstart[idx + 1]):
                    cr = pair_cr[t]; cb = pair_cb[t]
                    y = R - _CRRn[cr]
                    if y < 0 or y > 255:
                        continue
                    if G != y + ((_CBGn[cb] + _CRGn[cr]) >> 16):
                        continue
                    dd = (y - fy) ** 2 + (cb - fcb) ** 2 + (cr - fcr) ** 2
                    if dd < bd:
                        bd = dd; by = y; bcb = cb; bcr = cr
                if bd < 1e17:
                    ycc[r, c, 0] = by; ycc[r, c, 1] = bcb; ycc[r, c, 2] = bcr
                    exact[r, c] = 1
                    found = True
            if not found:
                v0 = int(round(fy)); v1 = int(round(fcb)); v2 = int(round(fcr))
                ycc[r, c, 0] = 0 if v0 < 0 else (255 if v0 > 255 else v0)
                ycc[r, c, 1] = 0 if v1 < 0 else (255 if v1 > 255 else v1)
                ycc[r, c, 2] = 0 if v2 < 0 else (255 if v2 > 255 else v2)
    return ycc, exact

def _dct_matrix():
    A = np.zeros((8, 8))
    for u in range(8):
        cu = np.sqrt(0.5) if u == 0 else 1.0
        for x in range(8):
            A[u, x] = 0.5 * cu * np.cos((2 * x + 1) * u * np.pi / 16)
    return A

_DCTA = _dct_matrix()

def _blocks(plane):
    H, W = plane.shape
    return plane.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)

def _plane_dct(plane):
    return _DCTA @ (_blocks(plane) - 128.0) @ _DCTA.T

_IJG_LUMA = np.array([16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,
                      14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,
                      49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99], np.int64).reshape(8, 8)
_IJG_CHROMA = np.array([17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,
                        47,66,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,
                        99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99], np.int64).reshape(8, 8)

def _qfilt(vals):
    base = vals[np.abs(vals) > 3.0]
    if base.size > 3000:
        base = base[:: base.size // 3000 + 1]
    return base

def _q_pos_ok(vals, q):
    if vals.size < 8:
        return True
    dev = (vals - q * np.round(vals / q)) ** 2
    k = max(1, int(0.98 * vals.size))
    tm = float(np.partition(dev, k - 1)[:k].mean())
    if np.sqrt(tm) > 0.15 * q + 0.8:
        return False
    var = max(float(np.var(vals)), 1e-9)
    return max(tm - 0.13, 0.0) / min(q * q / 12.0, var) < 0.35

def _fit_ijg(pre, base):
    best_t = None
    best_s = 1e18
    for Q in range(1, 101):
        S = 5000 // Q if Q < 50 else 200 - 2 * Q
        tbl = np.clip((base * S + 50) // 100, 1, 255)
        tot = 0.0
        wsum = 0.0
        for (v, u), vals in pre:
            if vals.size < 8:
                continue
            q = int(tbl[v, u])
            if q < 3:
                continue
            dev = (vals - q * np.round(vals / q)) ** 2
            k = max(1, int(0.98 * vals.size))
            tm = float(np.partition(dev, k - 1)[:k].mean())
            var = max(float(np.var(vals)), 1e-9)
            sc = max(tm - 0.13, 0.0) / min(q * q / 12.0, var)
            w = min(vals.size, 500)
            tot += min(sc, 3.0) * w
            wsum += w
        if wsum >= 100 and tot / wsum < 0.15:
            return tbl.astype(np.int32), tot / wsum
    return None, 1e18

def _q_scores(base, qlo, qhi, trim):
    k = max(1, int(trim * base.size))
    var = max(float(np.var(base)), 1e-9)
    ss = np.full(qhi + 1, 1e18)
    ok = np.zeros(qhi + 1, bool)
    for q in range(qlo, qhi + 1):
        dev = (base - q * np.round(base / q)) ** 2
        tm = float(np.partition(dev, k - 1)[:k].mean())
        ss[q] = max(tm - 0.13, 0.0) / min(q * q / 12.0, var)
        ok[q] = np.sqrt(tm) <= 0.15 * q + 0.8
    return ss, ok

def _q_pick(ss, ok):
    if not ok.any():
        return 1
    smin = ss[ok].min()
    thr = max(smin * 1.5, smin + 0.03)
    best = 1
    for q in range(1, ss.shape[0]):
        if ok[q] and ss[q] <= thr:
            best = q
    return best

def _estimate_q(vals):
    base = vals[np.abs(vals) > 3.0]
    if base.size > 15000:
        base = base[:: base.size // 15000 + 1]
    if base.size == 0:
        return 64
    if base.size < 8 or float(np.var(base)) <= 1e-9:
        return min(max(int(round(float(np.abs(base).min()))), 1), 128)
    ss, ok = _q_scores(base, 1, 128, 0.98)
    best = _q_pick(ss, ok)
    if best <= 4:
        b2 = vals[np.abs(vals) > 1.2]
        if b2.size > 15000:
            b2 = b2[:: b2.size // 15000 + 1]
        if b2.size >= 8 and float(np.var(b2)) > 1e-9:
            ss2, ok2 = _q_scores(b2, 1, 12, 0.95)
            best = _q_pick(ss2, ok2)
    return best

@njit(cache=True)
def _sample_bad444(K, quant, img, nbr, nbc, step):
    ycc = np.empty((3, 64), np.int64)
    tot = 0
    cnt = 0
    for b in range(0, nbr * nbc, step):
        i = b // nbc
        j = b % nbc
        tot += _sim_bad(K, quant, b, img, i * 8, j * 8, ycc)
        cnt += 64
    return tot / cnt

@njit(cache=True)
def _rbad_sample420(img, Ys, Cbs, Crs, step):
    H, W, _ = img.shape
    tot = 0
    cnt = 0
    for r0 in range(0, H - 15, 16 * step):
        for c0 in range(0, W - 15, 16 * step):
            tot += _rbad420(img, Ys, Cbs, Crs, r0, c0, r0 + 16, c0 + 16)
            cnt += 256
    return tot / cnt

def _attempt_420(img, ycc, exact, tmp, verbose=False, use_ijg=True):
    H, W, _ = img.shape
    Cbf = ycc[:, :, 1].astype(np.int64)
    Crf = ycc[:, :, 2].astype(np.int64)
    Cbh = _solve_down(Cbf)
    Crh = _solve_down(Crf)
    mm = (_up_mismatch(Cbh, Cbf) | _up_mismatch(Crh, Crf)) > 0
    if verbose:
        print(f"  420: downsample risolto, {int(mm.sum())} campioni non verificati")
    if float(mm.mean()) > 0.10:
        return None
    nbrY, nbcY = H // 8, W // 8
    nY = nbrY * nbcY
    nbrC, nbcC = H // 16, W // 16
    nC = nbrC * nbcC
    okbY = _blocks(exact).reshape(-1, 64).min(axis=1).astype(bool)
    okC = (exact.astype(bool) & (~mm))
    okbC = okC.reshape(nbrC, 16, nbcC, 16).min(axis=(1, 3)).reshape(-1)
    D_Y = _plane_dct(ycc[:, :, 0].astype(np.float64))
    D_Cb = _plane_dct(Cbh.astype(np.float64))
    D_Cr = _plane_dct(Crh.astype(np.float64))
    if int(okbY.sum()) < 1000:
        okbY = np.ones(nY, bool)
    if int(okbC.sum()) < 500:
        okbC = np.ones(nC, bool)
    qt = np.zeros((2, 8, 8), np.int32)

    def _sel(Dc, okm, v, u):
        m = okm.copy()
        if v == 0:
            m &= np.abs(Dc)[:, 1:, :].sum(axis=(1, 2)) > 2.0
        if u == 0:
            m &= np.abs(Dc)[:, :, 1:].sum(axis=(1, 2)) > 2.0
        if int(m.sum()) < 24:
            m = okm
        return Dc[m, v, u]

    if use_ijg:
        preL = [((v, u), _qfilt(_sel(D_Y, okbY, v, u))) for v in range(8) for u in range(8)]
        preC = [((v, u), _qfilt(np.concatenate([_sel(D_Cb, okbC, v, u), _sel(D_Cr, okbC, v, u)]))) for v in range(8) for u in range(8)]
        tL, sL = _fit_ijg(preL, _IJG_LUMA)
        tC, sC = _fit_ijg(preC, _IJG_CHROMA)
        if tL is None or tC is None:
            return None
        dL = dict(preL); dC = dict(preC)
        for v in range(8):
            for u in range(8):
                qt[0, v, u] = tL[v, u] if _q_pos_ok(dL[(v, u)], int(tL[v, u])) else _estimate_q(_sel(D_Y, okbY, v, u))
                qt[1, v, u] = tC[v, u] if _q_pos_ok(dC[(v, u)], int(tC[v, u])) else _estimate_q(np.concatenate([_sel(D_Cb, okbC, v, u), _sel(D_Cr, okbC, v, u)]))
    else:
        for v in range(8):
            for u in range(8):
                qt[0, v, u] = _estimate_q(_sel(D_Y, okbY, v, u))
                qt[1, v, u] = _estimate_q(np.concatenate([_sel(D_Cb, okbC, v, u), _sel(D_Cr, okbC, v, u)]))
    if int((qt > 1).sum()) < 6:
        return None
    if verbose:
        print(f"  420: qt stimata (dc luma={qt[0,0,0]}, dc chroma={qt[1,0,0]})")
    qmap = [0, 1, 1]
    quantY = qt[0].astype(np.int64)
    quantC = qt[1].astype(np.int64)

    def _mk(D, q, n):
        e = D.reshape(n, 64) / q.reshape(64).astype(np.float64)[None]
        Kc = np.round(e)
        r = e - Kc
        fl = np.zeros((n, 64), np.int8)
        border = np.abs(r) >= 0.22
        fl[border] = np.sign(r[border]).astype(np.int8)
        return np.ascontiguousarray(Kc.astype(np.int16)), fl

    KY, fY = _mk(D_Y, quantY, nY)
    KCb, fCb = _mk(D_Cb, quantC, nC)
    KCr, fCr = _mk(D_Cr, quantC, nC)
    del D_Y, D_Cb, D_Cr
    Ys, Cbs, Crs = _sim_planes420(KY, KCb, KCr, quantY, quantC, H, W)
    if _rbad_sample420(img, Ys, Cbs, Crs, 7) > 0.45:
        return None
    bad, nfix = _refine420(KY, KCb, KCr, quantY, quantC, fY, fCb, fCr, img, Ys, Cbs, Crs)
    if verbose:
        print(f"  420: refine {nfix} coefficienti corretti, {bad} pixel residui")
    if bad * 25 > img.size:
        return None
    planes = [KY.reshape(nbrY, nbcY, 8, 8).astype(np.int16),
              KCb.reshape(nbrC, nbcC, 8, 8).astype(np.int16),
              KCr.reshape(nbrC, nbcC, 8, 8).astype(np.int16)]
    d = jpeglib.from_dct(Y=planes[0], Cb=planes[1], Cr=planes[2],
                         qt=qt.astype(np.uint16), quant_tbl_no=qmap)
    d.write_dct(tmp)
    rgb2 = np.asarray(Image.open(tmp).convert("RGB"))
    os.remove(tmp)
    di = np.flatnonzero(img.reshape(-1) != rgb2.reshape(-1))
    if verbose:
        print(f"  420: verifica finale, {di.size} valori in patch")
    if di.size * 4 > img.size:
        return None
    return planes, qt.astype(np.uint16), qmap, di, img.reshape(-1)[di]

def _inv_blob(inv, verbose=False):
    planes, qt, qmap, di, dv = inv
    out = bytearray(_dct_blob(planes, qt, qmap, b""))
    z = zlib.compress(di.astype("<u4").tobytes() + dv.astype("u1").tobytes(), 9)
    out += _u32(di.size) + _u32(len(z)) + z
    if verbose:
        print(f"    blob {len(out)/1e3:.0f} KB (patch {di.size} valori)")
    return bytes(out)

def _try_jpeg_invert(img, tmp, verbose=False):
    H, W, _ = img.shape
    if H % 8 or W % 8:
        return None
    ycc, exact = _invert_ycc_nb(img, _PCR, _PCB, _PSTART, _D1LO)
    if verbose:
        print(f"  ycc esatta: {100*exact.mean():.2f}% dei pixel")
    blobs = []
    clean = False
    for ijg in (True, False):
        if clean:
            break
        r = _attempt_444(img, ycc, exact, tmp, verbose, use_ijg=ijg)
        if r is not None:
            clean = clean or r[3].size < (H * W) // 100
            blobs.append(_inv_blob(r, verbose))
    if (not clean) and H % 16 == 0 and W % 16 == 0:
        for ijg in (True, False):
            if clean:
                break
            r = _attempt_420(img, ycc, exact, tmp, verbose, use_ijg=ijg)
            if r is not None:
                clean = clean or r[3].size < (H * W) // 100
                blobs.append(_inv_blob(r, verbose))
    if not blobs:
        return None
    return min(blobs, key=len)

def _attempt_444(img, ycc, exact, tmp, verbose=False, use_ijg=True):
    H, W, _ = img.shape
    okb = _blocks(exact).reshape(-1, 64).min(axis=1).astype(bool)
    nbr, nbc = H // 8, W // 8
    n = nbr * nbc
    D = [_plane_dct(ycc[:, :, c].astype(np.float64)) for c in range(3)]
    if int(okb.sum()) < 1000:
        okb = np.ones(n, bool)
    qt = np.zeros((2, 8, 8), np.int32)
    ev = [np.abs(Dc)[:, 1:, :].sum(axis=(1, 2)) > 2.0 for Dc in D]
    eh = [np.abs(Dc)[:, :, 1:].sum(axis=(1, 2)) > 2.0 for Dc in D]

    def _sel(c, v, u):
        m = okb.copy()
        if v == 0:
            m &= ev[c]
        if u == 0:
            m &= eh[c]
        if int(m.sum()) < 24:
            m = okb
        return D[c][m, v, u]

    if use_ijg:
        preL = [((v, u), _qfilt(_sel(0, v, u))) for v in range(8) for u in range(8)]
        preC = [((v, u), _qfilt(np.concatenate([_sel(1, v, u), _sel(2, v, u)]))) for v in range(8) for u in range(8)]
        tL, sL = _fit_ijg(preL, _IJG_LUMA)
        tC, sC = _fit_ijg(preC, _IJG_CHROMA)
        if tL is None or tC is None:
            return None
        dL = dict(preL); dC = dict(preC)
        for v in range(8):
            for u in range(8):
                qt[0, v, u] = tL[v, u] if _q_pos_ok(dL[(v, u)], int(tL[v, u])) else _estimate_q(_sel(0, v, u))
                qt[1, v, u] = tC[v, u] if _q_pos_ok(dC[(v, u)], int(tC[v, u])) else _estimate_q(np.concatenate([_sel(1, v, u), _sel(2, v, u)]))
    else:
        for v in range(8):
            for u in range(8):
                qt[0, v, u] = _estimate_q(_sel(0, v, u))
                qt[1, v, u] = _estimate_q(np.concatenate([_sel(1, v, u), _sel(2, v, u)]))

    if int((qt > 1).sum()) < 6:
        return None
    if verbose:
        print(f"  qt stimata (dc luma={qt[0,0,0]}, dc chroma={qt[1,0,0]})")
    qmap = [0, 1, 1]
    K = np.empty((3, n, 64), np.int16)
    flip = np.zeros((3, n, 64), np.int8)
    quant = np.empty((3, 8, 8), np.int64)
    for c in range(3):
        q = qt[qmap[c]].astype(np.float64)
        e = D[c].reshape(n, 64) / q.reshape(64)[None]
        Kc = np.round(e)
        r = e - Kc
        K[c] = Kc.astype(np.int16)
        fl = np.zeros((n, 64), np.int8)
        border = np.abs(r) >= 0.22
        fl[border] = np.sign(r[border]).astype(np.int8)
        flip[c] = fl
    del D
    quant[0] = qt[0]; quant[1] = qt[1]; quant[2] = qt[1]
    if _sample_bad444(K, quant, img, nbr, nbc, 37) > 0.45:
        return None
    bad, nfix = _refine_all(K, quant, flip, img, nbr, nbc)
    if verbose:
        print(f"  refine: {nfix} coefficienti corretti, {bad} pixel residui")
    if bad * 25 > img.size:
        return None
    planes = [K[c].reshape(nbr, nbc, 8, 8).astype(np.int16) for c in range(3)]
    d = jpeglib.from_dct(Y=planes[0], Cb=planes[1], Cr=planes[2],
                         qt=qt.astype(np.uint16), quant_tbl_no=qmap)
    d.write_dct(tmp)
    rgb2 = np.asarray(Image.open(tmp).convert("RGB"))
    os.remove(tmp)
    di = np.flatnonzero(img.reshape(-1) != rgb2.reshape(-1))
    if di.size * 4 > img.size:
        return None
    return planes, qt.astype(np.uint16), qmap, di, img.reshape(-1)[di]

def compress_pixels(img, tmp="_inv_tmp.jpg", verbose=False):
    dct_blob = _try_jpeg_invert(img, tmp, verbose=verbose)
    gen_blob = compress_image(img)
    if dct_blob is not None and len(dct_blob) < len(gen_blob):
        return dct_blob
    return gen_blob

def decompress_any(blob, tmp="_dec_tmp.jpg"):
    if blob[:4] == MAGICJ:
        _, off = decompress_jpeg(blob, tmp)
        arr = np.array(Image.open(tmp).convert("RGB"))
        os.remove(tmp)
        if off < len(blob):
            npatch, off = _ru32(blob, off)
            zlen, off = _ru32(blob, off)
            if npatch:
                rawp = zlib.decompress(blob[off:off + zlen])
                di = np.frombuffer(rawp, "<u4", npatch)
                dv = np.frombuffer(rawp, "u1", npatch, npatch * 4)
                arr.reshape(-1)[di.astype(np.int64)] = dv
        return arr
    return decompress_image(blob)

def decompress_jpeg(data, out_jpg):
    off = 4
    mlen, off = _ru32(data, off)
    markers = zlib.decompress(data[off:off + mlen]); off += mlen
    nch = data[off]; nqt = data[off + 1]; off += 2
    qt = np.frombuffer(data, "<u2", nqt * 64, off).reshape(nqt, 8, 8).astype(np.uint16); off += nqt * 128
    qmap = [data[off + i] for i in range(nch)]; off += nch
    planes = []
    for _ in range(nch):
        nbr, off = _ru32(data, off)
        nbc, off = _ru32(data, off)
        n = nbr * nbc
        C = np.empty((n, 64), np.int32)
        dres, off = _get_arr(data, off)
        C[:, 0] = _dc_unmed(dres.astype(np.int32), nbr, nbc)
        for k in range(1, 64):
            C[:, k], off = _get_arr(data, off)
        planes.append(C.reshape(nbr, nbc, 8, 8).astype(np.int16))
    d = jpeglib.from_dct(Y=planes[0], Cb=planes[1] if nch > 1 else None,
                         Cr=planes[2] if nch > 2 else None, qt=qt, quant_tbl_no=qmap)
    tmp = out_jpg + ".tmp"
    d.write_dct(tmp)
    body = open(tmp, "rb").read()
    os.remove(tmp)
    with open(out_jpg, "wb") as f:
        f.write(body[:2] + markers + body[2:])
    return out_jpg, off

@njit(cache=True)
def _rc_encode(ctxs, ks, mant, probs, kmax):
    n = ctxs.shape[0]
    out = np.empty(n * 8 + 4096, np.uint8)
    op = 0
    low = np.uint64(0); rng = np.uint64(0xFFFFFFFF)
    cache = np.uint64(0); csize = np.uint64(1)
    TOP = np.uint64(1 << 24); M32 = np.uint64(0xFFFFFFFF)
    for i in range(n):
        ctx = ctxs[i]; k = ks[i]
        for pos in range(kmax + 1):
            bit = 1 if pos < k else 0
            pidx = ctx * (kmax + 1) + pos
            p = np.uint64(probs[pidx])
            bound = (rng >> np.uint64(11)) * p
            if bit == 0:
                rng = bound
                probs[pidx] = np.int32(p + ((np.uint64(2048) - p) >> np.uint64(5)))
            else:
                low += bound; rng -= bound
                probs[pidx] = np.int32(p - (p >> np.uint64(5)))
            while rng < TOP:
                if (low < np.uint64(0xFF000000)) or (low > M32):
                    temp = cache
                    while True:
                        out[op] = np.uint8((temp + (low >> np.uint64(32))) & np.uint64(0xFF)); op += 1
                        temp = np.uint64(0xFF); csize -= np.uint64(1)
                        if csize == 0:
                            break
                    cache = (low >> np.uint64(24)) & np.uint64(0xFF)
                csize += np.uint64(1)
                low = (low << np.uint64(8)) & M32; rng = (rng << np.uint64(8)) & M32
            if bit == 0:
                break
        if k >= 1:
            m = mant[i]
            for j in range(k - 2, -1, -1):
                rng = rng >> np.uint64(1)
                if (m >> j) & 1:
                    low += rng
                while rng < TOP:
                    if (low < np.uint64(0xFF000000)) or (low > M32):
                        temp = cache
                        while True:
                            out[op] = np.uint8((temp + (low >> np.uint64(32))) & np.uint64(0xFF)); op += 1
                            temp = np.uint64(0xFF); csize -= np.uint64(1)
                            if csize == 0:
                                break
                        cache = (low >> np.uint64(24)) & np.uint64(0xFF)
                    csize += np.uint64(1)
                    low = (low << np.uint64(8)) & M32; rng = (rng << np.uint64(8)) & M32
    for _ in range(5):
        if (low < np.uint64(0xFF000000)) or (low > M32):
            temp = cache
            while True:
                out[op] = np.uint8((temp + (low >> np.uint64(32))) & np.uint64(0xFF)); op += 1
                temp = np.uint64(0xFF); csize -= np.uint64(1)
                if csize == 0:
                    break
            cache = (low >> np.uint64(24)) & np.uint64(0xFF)
        csize += np.uint64(1)
        low = (low << np.uint64(8)) & M32
    return out[:op].copy()

@njit(cache=True)
def _q1(g, tb):
    a = g if g >= 0 else -g
    lev = 0
    while lev < tb.shape[0] and a >= tb[lev]:
        lev += 1
    return lev if g >= 0 else -lev

@njit(cache=True)
def _prep_plane(P, tb):
    H, W = P.shape
    n = H * W
    pred = np.empty(n, np.int32); ctx = np.empty(n, np.int32); s = np.empty(n, np.int8)
    i = 0
    for r in range(H):
        for c in range(W):
            a = P[r, c - 1] if c > 0 else (P[r - 1, c] if r > 0 else 0)
            b = P[r - 1, c] if r > 0 else a
            cc = P[r - 1, c - 1] if (r > 0 and c > 0) else b
            dd = P[r - 1, c + 1] if (r > 0 and c + 1 < W) else b
            q1 = _q1(dd - b, tb); q2 = _q1(b - cc, tb); q3 = _q1(cc - a, tb)
            sg = 1
            if q1 < 0 or (q1 == 0 and q2 < 0) or (q1 == 0 and q2 == 0 and q3 < 0):
                sg = -1; q1 = -q1; q2 = -q2; q3 = -q3
            ctx[i] = (q1 * RAW_SPAN + (q2 + RAW_Q)) * RAW_SPAN + (q3 + RAW_Q)
            pred[i] = _med(a, b, cc); s[i] = sg; i += 1
    return pred, ctx, s

@njit(cache=True)
def _decode_plane(data, H, W, probs, bias, tb, kmax):
    rng = np.uint64(0xFFFFFFFF); code = np.uint64(0)
    pos = 1
    TOP = np.uint64(1 << 24); M32 = np.uint64(0xFFFFFFFF)
    for _ in range(4):
        code = ((code << np.uint64(8)) | np.uint64(data[pos])) & M32; pos += 1
    rec = np.empty((H, W), np.int32)
    for r in range(H):
        for c in range(W):
            a = rec[r, c - 1] if c > 0 else (rec[r - 1, c] if r > 0 else 0)
            b = rec[r - 1, c] if r > 0 else a
            cc = rec[r - 1, c - 1] if (r > 0 and c > 0) else b
            dd = rec[r - 1, c + 1] if (r > 0 and c + 1 < W) else b
            q1 = _q1(dd - b, tb); q2 = _q1(b - cc, tb); q3 = _q1(cc - a, tb)
            sg = 1
            if q1 < 0 or (q1 == 0 and q2 < 0) or (q1 == 0 and q2 == 0 and q3 < 0):
                sg = -1; q1 = -q1; q2 = -q2; q3 = -q3
            ctx = (q1 * RAW_SPAN + (q2 + RAW_Q)) * RAW_SPAN + (q3 + RAW_Q)
            k = 0
            for p_ in range(kmax + 1):
                pidx = ctx * (kmax + 1) + p_
                p = np.uint64(probs[pidx])
                bound = (rng >> np.uint64(11)) * p
                if code < bound:
                    rng = bound
                    probs[pidx] = np.int32(p + ((np.uint64(2048) - p) >> np.uint64(5)))
                    bit = 0
                else:
                    code -= bound; rng -= bound
                    probs[pidx] = np.int32(p - (p >> np.uint64(5)))
                    bit = 1
                while rng < TOP:
                    rng = (rng << np.uint64(8)) & M32
                    code = ((code << np.uint64(8)) | np.uint64(data[pos])) & M32; pos += 1
                if bit == 0:
                    break
                k += 1
            if k == 0:
                u = 0
            else:
                m = 1
                for _ in range(k - 1):
                    rng = rng >> np.uint64(1)
                    bit = 0
                    if code >= rng:
                        code -= rng; bit = 1
                    m = (m << 1) | bit
                    while rng < TOP:
                        rng = (rng << np.uint64(8)) & M32
                        code = ((code << np.uint64(8)) | np.uint64(data[pos])) & M32; pos += 1
                u = m
            e2 = (u >> 1) if (u & 1) == 0 else -((u + 1) >> 1)
            errf = e2 + bias[ctx]
            rec[r, c] = _med(a, b, cc) + sg * errf
    return rec

def _split_bayer(m):
    return [m[0::2, 0::2], m[0::2, 1::2], m[1::2, 0::2], m[1::2, 1::2]]

def compress_raw(mosaic):
    H, W = mosaic.shape
    out = bytearray(MAGICR + _u32(H) + _u32(W))
    for P in _split_bayer(mosaic.astype(np.int32)):
        pred, ctx, s = _prep_plane(np.ascontiguousarray(P), RAW_TB)
        errf = (P.reshape(-1) - pred) * s
        bias = np.zeros(RAW_NCTX, np.int32)
        order = np.argsort(ctx, kind="stable")
        cs = ctx[order]; es = errf[order]
        bnd = np.concatenate(([0], np.flatnonzero(np.diff(cs)) + 1, [cs.size]))
        for j in range(bnd.size - 1):
            grp = es[bnd[j]:bnd[j + 1]]
            if grp.size:
                bias[cs[bnd[j]]] = int(round(np.median(grp)))
        e2 = errf - bias[ctx]
        u = np.where(e2 >= 0, 2 * e2, -2 * e2 - 1).astype(np.int64)
        k = np.zeros(u.size, np.int32)
        nz = u > 0
        k[nz] = np.floor(np.log2(u[nz])).astype(np.int32) + 1
        mant = np.where(k >= 1, u - (np.int64(1) << np.maximum(k - 1, 0)), 0).astype(np.int32)
        probs = np.full(RAW_NCTX * (KMAX + 1), 1024, np.int32)
        blob = _rc_encode(ctx.astype(np.int32), k, mant, probs, KMAX)
        bz = zlib.compress(bias.astype("<i2").tobytes(), 9)
        out += _u32(len(bz)) + bz + _u32(len(blob)) + blob.tobytes()
    return bytes(out)

def decompress_raw(data):
    off = 4
    H, off = _ru32(data, off)
    W, off = _ru32(data, off)
    m = np.empty((H, W), np.uint16)
    shapes = [((H + 1) // 2, (W + 1) // 2), ((H + 1) // 2, W // 2),
              (H // 2, (W + 1) // 2), (H // 2, W // 2)]
    slots = [(np.s_[0::2, 0::2]), (np.s_[0::2, 1::2]), (np.s_[1::2, 0::2]), (np.s_[1::2, 1::2])]
    for (hh, ww), sl in zip(shapes, slots):
        bl, off = _ru32(data, off)
        bias = np.frombuffer(zlib.decompress(data[off:off + bl]), "<i2").astype(np.int32); off += bl
        pl, off = _ru32(data, off)
        blob = np.frombuffer(data, np.uint8, pl, off); off += pl
        payload = np.concatenate([blob, np.zeros(16, np.uint8)])
        probs = np.full(RAW_NCTX * (KMAX + 1), 1024, np.int32)
        rec = _decode_plane(payload, hh, ww, probs, bias, RAW_TB, KMAX)
        m[sl] = rec.astype(np.uint16)
    return m

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    base = path.rsplit(".", 1)[0]
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("dng", "cr2", "cr3", "nef", "arw", "raw", "rw2", "orf", "raf", "pef", "srw"):
        import rawpy
        mosaic = np.ascontiguousarray(rawpy.imread(path).raw_image_visible)
        H, W = mosaic.shape
        raw = mosaic.size * 2
        print(f"{path}: mosaico {W}x{H}  grezzo {raw/1e6:.2f} MB")
        t = time.time(); blob = compress_raw(mosaic); te = time.time() - t
        t = time.time(); dec = decompress_raw(blob); td = time.time() - t
        with open(base + ".tso", "wb") as f:
            f.write(blob)
        print(f"encode {te:.1f}s  decode {td:.1f}s")
        print(f".tso = {len(blob)/1e6:.2f} MB   ratio {raw/len(blob):.2f}x   ({8*len(blob)/mosaic.size:.3f} bit/px)")
        print(f"LOSSLESS bit-per-bit: {np.array_equal(mosaic, dec)}")
        try:
            import imagecodecs
            jxl = len(imagecodecs.jpegxl_encode(mosaic, lossless=True, effort=7))
            j2k = len(imagecodecs.jpeg2k_encode(mosaic, level=0, reversible=True))
            print(f"JPEG-XL={jxl/1e6:.2f}MB  JPEG2000={j2k/1e6:.2f}MB  Tsiro={len(blob)/1e6:.2f}MB")
            print(f"Tsiro vs JPEG2000: {100*(1-len(blob)/j2k):.1f}% piu' piccolo | vs JPEG-XL: {100*(1-len(blob)/jxl):.1f}%")
        except Exception:
            pass
        sys.exit(0)
    head = open(path, "rb").read(2)
    force_pixel = len(sys.argv) > 2 and sys.argv[2] == "pixel"
    if head == b"\xFF\xD8" and not force_pixel:
        src = os.path.getsize(path)
        t = time.time(); blob = compress_jpeg(path); te = time.time() - t
        with open(base + ".tso", "wb") as f:
            f.write(blob)
        t = time.time(); decompress_jpeg(blob, base + "_decoded.jpg"); td = time.time() - t
        a = np.asarray(Image.open(path).convert("RGB"))
        b = np.asarray(Image.open(base + "_decoded.jpg").convert("RGB"))
        print(f"{path}: {a.shape[1]}x{a.shape[0]}  jpg={src/1e6:.2f} MB")
        print(f"encode {te:.1f}s  decode {td:.1f}s")
        print(f".tso = {len(blob)/1e6:.2f} MB   ({100*len(blob)/src:.1f}% del jpg originale)")
        print(f"PIXEL IDENTICI: {np.array_equal(a, b)}   maxdiff={int(np.abs(a.astype(np.int16)-b.astype(np.int16)).max())}")
    else:
        img = np.asarray(Image.open(path).convert("RGB"))
        H, W, C = img.shape
        raw = H * W * C
        print(f"{path}: {W}x{H} {C}ch  raw={raw/1e6:.1f} MB  (input: solo array di pixel)")
        t = time.time(); blob = compress_pixels(img, verbose=True); te = time.time() - t
        t = time.time(); dec = decompress_any(blob); td = time.time() - t
        with open(base + ".tso", "wb") as f:
            f.write(blob)
        mode = "DCT-inversa" if blob[:4] == MAGICJ else "pixel"
        print(f"modalita' {mode}   encode {te:.1f}s  decode {td:.1f}s")
        print(f".tso = {len(blob)/1e6:.2f} MB   ratio {raw/len(blob):.2f}x")
        print(f"IDENTICO bit-per-bit: {np.array_equal(img, dec)}")
        import io
        buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG", optimize=True)
        png = buf.tell()
        buf = io.BytesIO(); Image.fromarray(img).save(buf, "WEBP", lossless=True, quality=100)
        webp = buf.tell()
        try:
            import imagecodecs
            jxl = len(imagecodecs.jpegxl_encode(img, lossless=True, effort=3))
            jxl = min(jxl, len(imagecodecs.jpegxl_encode(img, lossless=True, effort=7)))
        except Exception:
            jxl = 0
        line = f"PNG={png/1e6:.2f}MB  WebP={webp/1e6:.2f}MB"
        if jxl:
            line += f"  JPEG-XL={jxl/1e6:.2f}MB"
        print(line + f"  Tsiro={len(blob)/1e6:.2f}MB")
        if jxl:
            print(f"Tsiro e' {jxl/len(blob):.1f}x piu' piccolo di JPEG XL lossless")
