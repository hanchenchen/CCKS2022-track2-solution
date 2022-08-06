import json
import unicodedata


def read_info(json_path):
    items = [json.loads(line) for line in open(json_path).readlines()]
    ret = {}
    for item in items:
        item_id = item["item_id"]
        ret[item_id] = item
    return ret


def read_pair(json_path):
    items = [json.loads(line) for line in open(json_path).readlines()]
    return items


def clean_str(s):
    # https://blog.csdn.net/Owen_goodman/article/details/107783304
    return unicodedata.normalize("NFKC", s.replace("#", "").lower())


def str2dict(s):
    d = {}
    s_split = s.split(";")
    for sub_str in s_split:
        sub_str_split = sub_str.split(":")
        p = sub_str_split[0]
        v = ":".join(sub_str_split[1:])
        if p not in d:
            d[p] = set()
        d[p].add(v)
    for k in d:
        d[k] = ",".join(d[k])
    return d
