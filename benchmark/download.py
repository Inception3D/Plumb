#!/usr/bin/env python3
"""Download the 50 source files of bench50 into DATA_ROOT (the layout benchmark/build.py reads), verified and resumable.

  python benchmark/download.py --data-root DIR [--spec benchmark/bench50.json] [--hf-endpoint URL] [--tasks ID ...]
                               [--workers N] [--verify]

PhysInOne: the scene ZIP from its Hugging Face shard repository at the pinned revision, checked by the LFS sha256.
Physion v1.5: the HDF5 member of the category's test-set ZIP on S3, read with HTTP range requests (the archives, up to
200 GB, are never downloaded whole), inflated and checked by the ZIP CRC32 and size. Interrupted transfers resume from
<file>.part (<file>.compressed.part for Physion); complete files are skipped (--verify checks them again).
Proxies are taken from the usual environment variables (HTTPS_PROXY, ...).
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests

HERE = Path(__file__).resolve().parent
CHUNK = 16 << 20  # bytes per range request
_local = threading.local()


def session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def get(url, headers, retries=6):
    for attempt in range(retries):
        try:
            r = session().get(url, headers={"Accept-Encoding": "identity", **headers}, stream=True, timeout=(30, 120))
            if r.status_code == 412:
                raise RuntimeError(f"{url} changed since the spec was written (ETag)")
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30))


def fetch(url, part, size, offset=0, headers=None):
    """append bytes [offset + len(part), offset + size) of url to part, CHUNK bytes per range request"""
    failures = 0
    while (have := part.stat().st_size if part.exists() else 0) < size:
        end = min(size, have + CHUNK)
        try:
            with get(url, {"Range": f"bytes={offset + have}-{offset + end - 1}", **(headers or {})}) as r:
                if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {offset + have}-"):
                    raise RuntimeError(f"{url}: the server ignored the range request")
                with part.open("ab") as f:
                    for chunk in r.iter_content(1 << 16):
                        if f.tell() + len(chunk) > end:
                            raise RuntimeError(f"{url}: the server sent more bytes than requested")
                        f.write(chunk)
            failures = 0
        except requests.RequestException:  # resume where the transfer stopped
            failures += 1
            if failures > 10:
                raise
            time.sleep(min(2 ** failures, 60))
    if part.stat().st_size != size:
        raise RuntimeError(f"{part} is larger than expected")


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_crc32(path):
    crc = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc


def physinone(rec, dest, endpoint):
    url = f"{endpoint}/datasets/{rec['repo']}/resolve/{rec['revision']}/{quote(rec['path'])}"
    part = dest.with_name(dest.name + ".part")
    fetch(url, part, rec["size"])
    if file_sha256(part) != rec["sha256"]:
        part.unlink()
        raise RuntimeError(f"{dest.name}: sha256 mismatch (partial file removed)")
    part.replace(dest)


def physion(rec, dest):
    if rec["compression"] not in (0, 8):
        raise RuntimeError(f"{rec['member']}: unsupported ZIP compression {rec['compression']}")
    name = rec["member"].encode()
    match = {"If-Match": rec["etag"]}
    with get(rec["url"], {"Range": f"bytes={rec['offset']}-{rec['offset'] + 29 + len(name)}", **match}) as r:
        head = r.content
    if r.status_code != 206 or len(head) != 30 + len(name) or head[:4] != b"PK\x03\x04" or head[30:] != name:
        raise RuntimeError(f"{rec['url']}: no local header of {rec['member']} at offset {rec['offset']}")
    name_len, extra_len = struct.unpack_from("<HH", head, 26)
    comp, part = dest.with_name(dest.name + ".compressed.part"), dest.with_name(dest.name + ".part")
    fetch(rec["url"], comp, rec["compressed_size"], rec["offset"] + 30 + name_len + extra_len, match)
    crc = size = 0
    dec = zlib.decompressobj(-15) if rec["compression"] == 8 else None
    with comp.open("rb") as src, part.open("wb") as out:
        while chunk := src.read(1 << 20):
            data = dec.decompress(chunk) if dec else chunk
            out.write(data); crc = zlib.crc32(data, crc); size += len(data)
        if dec:
            data = dec.flush()
            out.write(data); crc = zlib.crc32(data, crc); size += len(data)
            if not dec.eof:
                raise RuntimeError(f"{dest.name}: truncated deflate stream")
    if (size, crc) != (rec["size"], rec["crc32"]):
        comp.unlink(); part.unlink()
        raise RuntimeError(f"{dest.name}: CRC32/size mismatch (partial files removed)")
    part.replace(dest)
    comp.unlink()


def verify(scene, dest):
    rec = scene["download"]
    if scene["dataset"] == "physinone":
        return file_sha256(dest) == rec["sha256"]
    return file_crc32(dest) == rec["crc32"]


def download(scene, data_root, endpoint, check):
    rec = scene["download"]; dest = data_root / scene["source"]
    if not dest.resolve().is_relative_to(data_root):
        raise ValueError(f"unsafe destination {scene['source']}")
    if dest.exists():
        if dest.stat().st_size != rec["size"]:
            raise RuntimeError(f"{dest} exists with another size; remove it")
        if check and not verify(scene, dest):
            raise RuntimeError(f"{dest} fails the checksum; remove it")
        return "verified" if check else "present"
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if scene["dataset"] == "physinone":
        physinone(rec, dest, endpoint)
    else:
        physion(rec, dest)
    return f"downloaded {rec.get('compressed_size', rec['size']) / 1e6:.0f} MB in {time.time() - t0:.0f} s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--spec", type=Path, default=HERE / "bench50.json")
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                    help="Hugging Face endpoint, e.g. https://hf-mirror.com (default: $HF_ENDPOINT or https://huggingface.co)")
    ap.add_argument("--tasks", nargs="+", metavar="ID", help="only the sources of these job ids")
    ap.add_argument("--workers", type=int, default=4, help="files downloaded in parallel")
    ap.add_argument("--verify", action="store_true", help="also check the checksums of files already present")
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text(encoding="utf-8"))
    scenes = spec["scenes"]
    if a.tasks:
        unknown = set(a.tasks) - {s["job_id"] for s in scenes}
        if unknown:
            ap.error(f"unknown job ids: {sorted(unknown)}")
        scenes = [s for s in scenes if s["job_id"] in a.tasks]
    data_root = a.data_root.resolve(); data_root.mkdir(parents=True, exist_ok=True)
    total = sum(s["download"].get("compressed_size", s["download"]["size"]) for s in scenes
                if not (data_root / s["source"]).exists())
    print(f"{len(scenes)} source files, {total / 1e9:.1f} GB to download into {data_root}", flush=True)
    failed = []

    def one(scene):
        try:
            status = download(scene, data_root, a.hf_endpoint.rstrip("/"), a.verify)
        except Exception as exc:  # reported at the end; the other files go on
            failed.append(scene["source"]); status = f"FAILED: {type(exc).__name__}: {exc}"
        print(f"{scene['source']}: {status}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, scenes))
    print(f"{len(scenes) - len(failed)}/{len(scenes)} files complete" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
