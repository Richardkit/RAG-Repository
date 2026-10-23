#!/usr/bin/env python
# Downloads 3RScan public data release (modified to place files only into existing scan_id dirs)

import sys
import argparse
import os
import tempfile
import re
import zipfile

if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    import urllib.request as urllib
else:
    import urllib

BASE_URL = 'http://campar.in.tum.de/public_datasets/3RScan/'
DATA_URL = BASE_URL + 'Dataset/'
TOS_URL = 'http://campar.in.tum.de/public_datasets/3RScan/3RScanTOU.pdf'
TEST_FILETYPES = ['mesh.refined.v2.obj', 'mesh.refined.mtl', 'mesh.refined_0.png', 'sequence.zip']
# We only provide semantic annotations for the train and validation scans as well as the for the
# reference scans in the test set.
FILETYPES = TEST_FILETYPES + ['labels.instances.annotated.v2.ply', 'mesh.refined.0.010000.segs.v2.json', 'semseg.v2.json']

RELEASE = 'release_scans.txt'
HIDDEN_RELEASE = 'test_rescans.txt'
RELEASE_SIZE = '~94GB'

id_reg = re.compile(r"[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}")

def get_scans(scan_file):
    scan_lines = urllib.urlopen(scan_file)
    scans = []
    for scan_line in scan_lines:
        scan_line = scan_line.decode('utf8').rstrip('\n')
        match = id_reg.search(scan_line)
        if match:
            scans.append(match.group())
    return scans

def download_file(url, out_file):
    """
    Download URL to out_file (atomic temp -> rename).
    If file exists, skip.
    """
    print(url)
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        os.close(fh)
        urllib.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)

def safe_extract_zip(zip_path, target_dir):
    """
    Safely extract a zip to target_dir (prevents path traversal).
    """
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            member_path = os.path.join(target_dir, member.filename)
            abs_target = os.path.abspath(target_dir)
            abs_member = os.path.abspath(member_path)
            if not abs_member.startswith(abs_target + os.sep) and abs_member != abs_target:
                raise Exception(f"Unsafe zip member path: {member.filename}")
        zf.extractall(target_dir)

def download_scan(scan_id, scan_dir, file_types):
    """
    Only operate if scan_dir already exists.
    Do NOT create missing scan directories (skip those scans).
    For sequence.zip, extract to scan_dir/sequence/ and remove the zip afterwards.
    """
    if not os.path.isdir(scan_dir):
        print(f"SKIP: scan dir not found, will not download: {scan_id}")
        return

    print('Downloading 3RScan scan ' + scan_id + ' ...')
    for ft in file_types:
        url = DATA_URL + '/' + scan_id + '/' + ft
        # default path inside scan_dir
        out_file = os.path.join(scan_dir, ft)

        if ft == 'sequence.zip':
            # keep the zip in scan_dir temporarily, then extract to scan_dir/sequence/
            zip_path = os.path.join(scan_dir, 'sequence.zip')
            download_file(url, zip_path)

            seq_dir = os.path.join(scan_dir, 'sequence')
            try:
                print(f'\tExtracting {zip_path} -> {seq_dir}')
                safe_extract_zip(zip_path, seq_dir)
                # remove zip after successful extraction
                try:
                    os.remove(zip_path)
                    print(f'\tRemoved {zip_path}')
                except OSError:
                    pass
            except Exception as e:
                print(f'ERROR: failed to extract {zip_path}: {e}')
        else:
            download_file(url, out_file)

    print('Downloaded scan ' + scan_id)

def download_release(release_scans, out_root, file_types):
    """
    Iterate scans but ONLY download into existing scan_id folders under out_root.
    Never create new scan_id folders.
    """
    print('Downloading 3RScan release to ' + out_root + '...')
    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_root, scan_id)
        # only proceed if scan directory exists already (your filtered set)
        if os.path.isdir(scan_out_dir):
            download_scan(scan_id, scan_out_dir, file_types)
        else:

            os.makedirs(scan_out_dir, exist_ok=True)
            download_scan(scan_id, scan_out_dir, file_types)
            print(f"Created missing scan dir {scan_out_dir} and downloaded {scan_id}.")
            # print(f"SKIP: {scan_id} does not exist under {out_root}, not downloading.")
    print('Downloaded 3RScan release.')

def download_tfrecord(url, out_dir, file):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, file)
    download_file(url + '/' + file, out_file)

def main():
    parser = argparse.ArgumentParser(description='Downloads 3RScan public data release.')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--id', help='specific scan id to download')
    parser.add_argument('--type', help='specific file type to download')
    args = parser.parse_args()

    print('By pressing any key to continue you confirm that you have agreed to the 3RScan terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')

    release_scans = get_scans(BASE_URL + RELEASE)
    test_scans = get_scans(BASE_URL + HIDDEN_RELEASE)
    file_types = FILETYPES
    file_types_test = TEST_FILETYPES

    if args.type:  # download file type
        file_type = args.type
        if file_type == 'tfrecords':
            download_tfrecord(BASE_URL, args.out_dir, 'val-scans.tfrecords')
            download_tfrecord(BASE_URL, args.out_dir, 'train-scans.tfrecords')
            return
        elif file_type not in FILETYPES:
            print('ERROR: Invalid file type: ' + file_type)
            return
        file_types = [file_type]
        file_types_test = [file_type] if file_type in TEST_FILETYPES else []

    if args.id:  # download single scan
        scan_id = args.id
        if scan_id not in release_scans and scan_id not in test_scans:
            print('ERROR: Invalid scan id: ' + scan_id)
        else:
            scan_dir = os.path.join(args.out_dir, scan_id)
            # Only if the scan folder already exists
            if os.path.isdir(scan_dir):
                if scan_id in release_scans:
                    download_scan(scan_id, scan_dir, file_types)
                elif scan_id in test_scans:
                    download_scan(scan_id, scan_dir, file_types_test)
            else:
                print(f"SKIP: target dir {scan_dir} does not exist; not downloading {scan_id}.")
    else:  # download entire release into existing scan folders only
        if len(file_types) == len(FILETYPES):
            print('WARNING: You are downloading the entire 3RScan release which requires ' + RELEASE_SIZE + ' of space.')
        else:
            print('WARNING: You are downloading all 3RScan scans of type ' + file_types[0])
        print('Note: Only existing scan directories under the output root will be used. Others will be skipped.')
        print('***')
        print('Press any key to continue, or CTRL-C to exit.')
        _ = input('')
        download_release(release_scans, args.out_dir, file_types)
        download_release(test_scans, args.out_dir, file_types_test)

if __name__ == "__main__":
    main()
