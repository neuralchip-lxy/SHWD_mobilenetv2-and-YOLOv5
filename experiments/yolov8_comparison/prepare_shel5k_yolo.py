"""Create a separate six-class YOLO dataset from the saved SHEL5K audit.

Keep originals intact. Split similar-image groups together (72/8/20 target).
This is a custom reproducible split, not the paper's official split.
"""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import yaml
from experiment_paths import path_for, resolve_path

NAMES = ['helmet', 'head_with_helmet', 'person_with_helmet', 'head', 'person_no_helmet', 'face']


def prepare(audit_dir, output):
    inventory = json.loads((audit_dir/'inventory.json').read_text(encoding='utf-8'))
    audit = json.loads((audit_dir/'audit.json').read_text(encoding='utf-8'))
    overlap = json.loads((audit_dir/'overlap.json').read_text(encoding='utf-8'))
    if set(audit['objects']) - set(NAMES) - {'person'}:
        raise ValueError(f'Unexpected class names: {audit["objects"]}')
    if audit['dimension_mismatch']:
        raise ValueError('Resolve XML/image dimensions before conversion.')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    excluded = [dict(id=r['id'],reason='ambiguous leftover person class; entire image excluded') for r in inventory if any(o['name'] not in NAMES for o in r['labels'])]
    inventory = [r for r in inventory if all(o['name'] in NAMES for o in r['labels'])]
    parent = {r['id']: r['id'] for r in inventory}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a != b: parent[max(a,b)] = min(a,b)
    for pair in overlap['within']:
        if pair['a'] in parent and pair['b'] in parent: union(pair['a'],pair['b'])
    exact = {}
    for r in inventory:
        if r['pixel_hash'] in exact: union(r['id'],exact[r['pixel_hash']])
        else: exact[r['pixel_hash']]=r['id']
    groups=defaultdict(list)
    for r in inventory: groups[find(r['id'])].append(r)
    shuffled=list(groups.values())
    random.Random(20260920).shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    ratios={'train':.72,'val':.08,'test':.20}
    assigned={s:[] for s in ratios}
    # Put large duplicate groups first, minimizing normalized capacity occupancy.
    for group in shuffled:
        split=min(ratios, key=lambda s:(len(assigned[s])+len(group)/2)/(len(inventory)*ratios[s]))
        assigned[split].extend(group)
    per_split={}; manifest=[]; corrections=[]
    # Validate and prepare every label before writing dataset files.
    labels={}
    for split, records in assigned.items():
        counts=Counter()
        for r in records:
            w,h=r['size']; rows=[]; seen=set()
            for obj in r['labels']:
                original=obj['box']; x1,y1,x2,y2=original
                x1,x2=max(0,min(w,x1)),max(0,min(w,x2))
                y1,y2=max(0,min(h,y1)),max(0,min(h,y2))
                if x2<=x1 or y2<=y1: raise ValueError(f'Degenerate box: {r["id"]} {obj}')
                box=[x1,y1,x2,y2]
                if box != original: corrections.append(dict(id=r['id'],old=original,new=box))
                cls=NAMES.index(obj['name']); item=(cls,*box)
                if item in seen:
                    corrections.append(dict(id=r['id'],duplicate_label_removed=list(item)))
                    continue
                seen.add(item)
                rows.append(f'{cls} {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
                counts[obj['name']]+=1
            labels[r['id']]='\n'.join(rows)+'\n'
            manifest.append(dict(id=r['id'],split=split,group=find(r['id']),pixel_hash=r['pixel_hash'],
                                 xml_sha256=hashlib.sha256(Path(r['xml']).read_bytes()).hexdigest()))
        if set(counts)!=set(NAMES): raise ValueError(f'Missing class in {split}: {counts}')
        per_split[split]=dict(images=len(records),objects=dict(counts))
    output.mkdir(parents=True)
    for split,records in assigned.items():
        image_dir=output/'images'/split; image_dir.mkdir(parents=True)
        label_dir=output/'labels'/split; label_dir.mkdir(parents=True)
        for r in records:
            shutil.copy2(r['image'],image_dir/Path(r['image']).name)
            (label_dir/(r['id']+'.txt')).write_text(labels[r['id']],encoding='utf-8')
    (output/'shel5k.yaml').write_text(yaml.safe_dump(dict(path='.',train='images/train',val='images/val',test='images/test',nc=6,names=NAMES),sort_keys=False),encoding='utf-8')
    (output/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    summary=dict(classes=NAMES,excluded_images=excluded,split_seed=20260920,target_ratios=ratios,split=per_split,groups=len(groups),
                 largest_group=max(map(len,groups.values())),corrections=corrections,overlap_audit=audit,
                 coordinate_rule='Use XML boundary values directly, clip to [0,W] and [0,H], no arbitrary one-pixel shift.',
                 scope='Independent six-class training benchmark; not a leakage-free SHWD-to-SHEL5K cross-domain test.',
                 grouping='Exact decoded pixels and pHash candidate pairs <=4 stay in the same split; heuristic does not guarantee all near duplicates found.')
    (output/'preparation_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    shutil.copy2(audit_dir/'overlap.json',output/'overlap_with_SHWD_and_internal.json')
    print(json.dumps(summary,indent=2),flush=True)
    print('DATA:',output/'shel5k.yaml',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,default=path_for('shel5k_audit'))
    parser.add_argument('--output',type=Path,default=path_for('shel5k_yolo'))
    args=parser.parse_args()
    prepare(resolve_path(args.audit_dir),resolve_path(args.output))
