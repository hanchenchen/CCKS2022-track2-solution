import json

pvs_path = "pvs.json"
pvs = json.load(open(pvs_path))
print(len(pvs))
ps = sorted(pvs, key=lambda k: len(pvs[k]), reverse=True)
num = 10
for p in ps[:num]:
    print(p, len(pvs[p]))
print(ps[:num])
