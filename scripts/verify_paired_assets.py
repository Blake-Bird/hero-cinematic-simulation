#!/usr/bin/env python3
"""Check the atlas's published local links, image headers, and MP4 range support."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen

from PIL import Image


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = set()

    def handle_starttag(self, tag, attrs):
        self.references.update(value for key, value in attrs
                               if key in ('href', 'src', 'poster', 'data-film') and value)


def public_path(path, root):
    parts = path.relative_to(root).parts
    return not any(part.startswith('.') for part in parts) and parts[0] != 'Source'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    parser.add_argument('--origin', required=True)
    parser.add_argument('--posters', action='store_true')
    args = parser.parse_args()
    root = args.atlas.resolve()
    pages = [p for p in root.rglob('*.html') if public_path(p, root)]
    films = sorted(p for p in (root / '04_Movies').rglob('*.mp4') if public_path(p, root)
                   and not p.name.endswith('.staged.mp4'))
    failures, images, targets = [], set(), set()
    for page in pages:
        links = Assets()
        links.feed(page.read_text())
        for reference in links.references:
            url = urlsplit(reference)
            if url.scheme or url.netloc or not url.path:
                continue
            local = unquote(url.path)
            path = root / local.lstrip('/') if local.startswith('/') else page.parent / local
            if not path.exists():
                failures.append({'page': str(page.relative_to(root)), 'missing': reference})
                continue
            targets.add(path.resolve())
            if path.suffix.lower() in ('.png', '.jpg', '.webp'):
                images.add(path.resolve())
    for path in sorted(images):
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            failures.append({'image': str(path), 'error': str(error)})

    def inspect_movie(path):
        url_path = path.relative_to(root).as_posix()
        try:
            probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,pix_fmt,width,height:format=duration', '-of', 'json', str(path)])
            info = json.loads(probe)
            stream, = info['streams']
            if stream['codec_name'] != 'h264' or stream['pix_fmt'] != 'yuv420p':
                raise ValueError(f'Expected H.264/yuv420p, got {stream}')
            duration = float(info['format']['duration'])
            request = Request(args.origin.rstrip('/') + '/' + quote(url_path), headers={'Range': 'bytes=16-79'})
            with urlopen(request, timeout=15) as response:
                if response.status != 206 or response.headers.get_content_type() != 'video/mp4' or len(response.read()) != 64:
                    raise ValueError(f'Expected 206 MP4 range response, got {response.status} {response.headers}')
            poster = path.with_suffix('.poster.jpg')
            if args.posters and not poster.exists():
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(duration * .58), '-i', str(path),
                                '-frames:v', '1', '-vf', 'scale=640:-2', '-update', '1', str(poster)], check=True)
            return {'film': url_path, 'duration_seconds': duration, **stream, 'range_response': 206}
        except Exception as error:
            return {'film': url_path, 'error': str(error)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        movies = list(pool.map(inspect_movie, films))
    failures.extend(r for r in movies if 'error' in r)
    audit = {'html_pages': len(pages), 'unique_link_targets': len(targets), 'verified_images': len(images),
             'films': movies, 'failures': failures}
    (root / '07_Status_and_Provenance/asset_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'Checked {len(pages)} pages, {len(targets)} link targets, {len(images)} images and {len(movies)} movies. Failures {len(failures)}.')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
