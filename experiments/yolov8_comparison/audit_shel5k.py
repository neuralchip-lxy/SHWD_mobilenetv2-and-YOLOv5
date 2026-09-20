import json, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image

ROOT=Path('E:/datasets/SHEL5K_original/9rcv8mm682-4/Safety Helmet Wearing Dataset')
SHWD=Path('E:/codex_work/experiment of mobilenet2 and YOLO5/SHWD_YOLO/images')
OUT=Path('E:/experiment_M2Y5/shel5k_preparation')
OUT.mkdir(exist_ok=True)

def hashes(path):
    with Image.open(path) as im:
        im=im.convert('RGB'); im.load()
        pixel=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        gray=np.array(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
        dct=cv2.dct(gray)[:8,:8].flatten()[1:]
        bits=dct>np.median(dct)
        ph=sum(int(b)<<i for i,b in enumerate(bits))
        size=im.size
    return pixel,ph,size

class BK:
    def __init__(self): self.tree=None
    def add(self,h,item):
        if self.tree is None: self.tree=[h,[item],{}]; return
        node=self.tree
        while True:
            d=(h^node[0]).bit_count()
            if d==0: node[1].append(item); return
            if d not in node[2]: node[2][d]=[h,[item],{}]; return
            node=node[2][d]
    def query(self,h,r=4):
        stack=[self.tree] if self.tree else []; found=[]
        while stack:
            node=stack.pop(); d=(h^node[0]).bit_count()
            if d<=r: found.extend((item,d) for item in node[1])
            stack.extend(child for dist,child in node[2].items() if d-r<=dist<=d+r)
        return found

def main():
    records=[]; objects=Counter(); bad=[]; dims=[]
    for i,p in enumerate(sorted((ROOT/'Annotations').glob('*.xml'))):
        t=ET.parse(p).getroot(); image=ROOT/'Images'/t.findtext('filename')
        pixel,ph,size=hashes(image); w,h=size
        if size!=(int(t.findtext('size/width')),int(t.findtext('size/height'))): dims.append(p.name)
        labels=[]
        for o in t.findall('object'):
            name=o.findtext('name'); box=[float(o.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
            if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h): bad.append([p.name,name,box,size])
            labels.append(dict(name=name,box=box,difficult=o.findtext('difficult','0'))); objects[name]+=1
        records.append(dict(id=p.stem,image=str(image),xml=str(p),pixel_hash=pixel,phash=ph,size=size,labels=labels))
        if i%1000==0: print('SHEL5K',i,flush=True)
    tree=BK(); shwd=[]
    for i,p in enumerate(sorted(SHWD.rglob('*'))):
        if p.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp'): continue
        pixel,ph,size=hashes(p); tree.add(ph,len(shwd)); shwd.append(dict(path=str(p),pixel_hash=pixel,phash=ph))
        if len(shwd)%2000==0: print('SHWD',len(shwd),flush=True)
    exact=defaultdict(list)
    for r in shwd: exact[r['pixel_hash']].append(r['path'])
    cross=[]; within=[]; own=BK()
    for r in records:
        matches=tree.query(r['phash'])
        if matches or r['pixel_hash'] in exact:
            cross.append(dict(id=r['id'],exact=exact.get(r['pixel_hash'],[]),candidates=[dict(path=shwd[j]['path'],distance=d) for j,d in matches]))
        within.extend(dict(a=other,b=r['id'],distance=d) for other,d in own.query(r['phash']))
        own.add(r['phash'],r['id'])
    report=dict(images=len(records),objects=dict(objects),bad_boxes=bad,dimension_mismatch=dims,
                cross_exact_images=sum(bool(r['exact']) for r in cross),cross_candidate_images=len(cross),within_candidate_pairs=len(within),
                phash_rule='63 AC DCT bits, Hamming <=4; candidates, not proof of identity')
    (OUT/'inventory.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    (OUT/'overlap.json').write_text(json.dumps(dict(cross=cross,within=within),ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Read-only SHEL5K XML/image and duplicate audit.')
    parser.add_argument('--source',type=Path,default=ROOT)
    parser.add_argument('--shwd-images',type=Path,default=SHWD)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    ROOT,SHWD,OUT=args.source,args.shwd_images,args.output
    OUT.mkdir(parents=True,exist_ok=True)
    main()
