import json,subprocess,re,sys,os
rows=[]
for n in range(int(sys.argv[1]),int(sys.argv[2])+1):
    f=f'pr{n}.json'
    if not os.path.exists(f):
        r=subprocess.run(['gh','pr','view',str(n),'--repo','Layr-Labs/mlxfast-qwen38-27b-tf-engine','--json','title,comments'],capture_output=True,text=True)
        if r.returncode: continue
        open(f,'w').write(r.stdout)
    d=json.load(open(f))
    m={}
    for c in d['comments']:
        for k,v in re.findall(r'^\| (\w+) \| (.*?) \|$',c['body'],re.M): m[k]=v
    if 'score' not in m: continue
    def g(k):
        try: return float(m[k])
        except: return None
    t=d['title']; sid=re.search(r'submission ([0-9a-f]{8})',t); who=re.search(r'@([\w-]+)',t)
    rows.append(dict(pr=n,id=sid.group(1) if sid else '?',who=who.group(1) if who else '?',score=g('score'),box=m.get('baseline_box'),
      cd=g('candidate_leg_decode_seconds_per_token'),cw=g('candidate_leg_decode_window_seconds_per_token'),cp=g('candidate_leg_prefill_seconds_per_token'),
      bd=g('baseline_leg_decode_seconds_per_token'),bw=g('baseline_leg_decode_window_seconds_per_token'),bp=g('baseline_leg_prefill_seconds_per_token')))
json.dump(rows,open('rows.json','w'),indent=0)
for r in rows:
    if r['cd'] is None: continue
    print(f"{r['pr']} {r['id']} {r['who'][:12]:12s} {r['score']:.5f} {r['box'] or '':22s} cand pre {1/r['cp']:7.1f} dec {1/r['cd']:6.2f} win {1/r['cw']:6.2f} | ctrl pre {1/r['bp']:7.1f} dec {1/r['bd']:6.2f}")
