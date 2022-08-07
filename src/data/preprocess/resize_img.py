import argparse
import glob
import multiprocessing
import os

from PIL import Image, ImageFile
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir")
parser.add_argument("--dst_dir")
parser.add_argument("--img_size", type=int)
args = parser.parse_args()


ImageFile.LOAD_TRUNCATED_IMAGES = True

os.makedirs(args.dst_dir, exist_ok=True)

path_list = []
path_list += glob.glob(f"{args.src_dir}/item_train_images/*")
path_list += glob.glob(f"{args.src_dir}/item_valid_images/*")
path_list += glob.glob(f"{args.src_dir}/item_test_images/*")
print(len(path_list))  # 103452
suffix_list = [os.path.splitext(path)[-1] for path in path_list]
print(set(suffix_list))  # {'.bmp', '.jpg', '.SS2', '.gif', '.png', '.jpeg'}


def extract_frames(path):
    name, suffix = os.path.splitext(os.path.basename(path))
    img = Image.open(path).convert("RGB")
    img = img.resize((args.img_size, args.img_size))
    img.save(f"{args.dst_dir}/{name}.jpg")


# for path in tqdm(path_list):
#     extract_frames(path)
pbar = tqdm(total=len(path_list))
update = lambda *args: pbar.update()
pool = multiprocessing.Pool(64)
for path in path_list:
    pool.apply_async(extract_frames, (path,), callback=update)
print("Start")
pool.close()
pool.join()
print("Done")
print(len(os.listdir(args.dst_dir)))  # 91906
