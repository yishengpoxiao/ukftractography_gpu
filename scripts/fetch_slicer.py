#!/usr/bin/env python3
"""Fetch a Windows Slicer installer and verify the official SHA-512 checksum."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


def download(version, output):
    query = dict(os='win', stability='release')
    if version != 'latest':
        query['version'] = version
    page = urllib.request.urlopen('https://download.slicer.org/?' + urllib.parse.urlencode(query), timeout=30).read().decode()
    item = re.search(r'href="/bitstream/([a-f0-9]+)" class="button-download', page)[1]
    api = 'https://slicer-packages.kitware.com/api/v1/'
    files = json.load(urllib.request.urlopen(api+'item/'+item+'/files', timeout=30))
    package, = [file for file in files if file['name'].endswith('-win-amd64.exe')]
    if version != 'latest' and package['name'] != 'Slicer-'+version+'-win-amd64.exe':
        raise ValueError('The download server returned a different Slicer version: '+package['name'])
    digest = hashlib.sha512()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as target, urllib.request.urlopen(api+'file/'+package['_id']+'/download', timeout=60) as source:
        while chunk := source.read(1024*1024):
            digest.update(chunk)
            target.write(chunk)
    if output.stat().st_size != package['size'] or digest.hexdigest() != package['sha512']:
        output.unlink()
        raise ValueError('Slicer installer size/SHA-512 verification failed.')
    print('Verified official installer:', package['name'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='latest')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    download(args.version, args.output)
