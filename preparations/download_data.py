import os
import requests
import shutil
import kagglehub

from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


# { dset_name: [download_type, dset_id (or url)] }
datasets = {
    "lion": ["huggingface", "daybreaksly/LION-data-train"],
    # "coco-2014": ["kagglehub", "jeffaudi/coco-2014-dataset-for-yolov3"],
    # "coco-2017": ["kagglehub", "rafaelpadilla/coco2017"],
    # "okvqa-2014": ["url", "http://images.cocodataset.org/zips/train2014.zip"],
    # "okvqa-2014": ["kagglehub", "power0341/ocr-vqa-200k-full"],
    # "textcaps": [
    #     "url",
    #     "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
    # "textcaps": ["kagglehub", "xuantruongdang99/textcaps-vi"],
    # ],
    # "vqav2-2014": ["url", "http://images.cocodataset.org/zips/train2014.zip"],
    # "visual_genome": ["kagglehub", "mathurinache/visual-genome"],
}
hf_cache_dir = Path("~/.cache/huggingface/datasets")
kh_cache_dir = Path("~/.cache/kagglehub/datasets")
dest_path = "images"


def check_dir(dir: str = dest_path):
    if not os.path.exists(dir):
        print(f"❌ No such directory found: {dir}")
        os.makedirs(dir)
        print(f"└── ✅ Directory made at: {dir}")

    return Path(dir)


def download_huggingface_dataset(dataset_id: str):
    dataset = load_dataset(dataset_id, split="train")
    dataset_path = hf_cache_dir / ""


def download_kaggle_dataset(dataset_name: str, dataset_id: str):
    path = kagglehub.dataset_download(dataset_id)
    # print(f"📀 {dataset_name} downloaded\n└── ✅ Saved at: {path}")

    dest_root = Path(f"images/{dataset_name}")
    path = Path(path)
    subdirs = [dir for dir in path.iterdir() if dir.is_dir()]
    # print(f"📁 Copying downloaded files to {dest_root}...")
    for _, dir in enumerate(subdirs, 1):
        dest_path = dest_root / dir.name
        shutil.copytree(src=dir, dst=dest_path, dirs_exist_ok=True)
        # if i == len(subdirs):
        #     print(f"└── ✅ {dir.name} copied")
        # else:
        #     print(f"├── ✅ {dir.name} copied")

    return path


def download_url_dataset(dataset_name: str, url: str):
    with requests.get(url, stream=True) as req:
        req.raise_for_status()
        total = int(req.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=out_path
        ) as pbar:
            for part in req.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
                    pbar.update(len(part))


def download(url: str, out_path: str, chunk=1 << 20):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=out_path
        ) as pbar:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
                    pbar.update(len(part))
    return out_path


if __name__ == "__main__":
    for dset_name, dset_info in tqdm(datasets.items()):
        download_type = dset_info[0]
        dset_id = dset_info[1]
        if download_type == "huggingface":
            download_huggingface_dataset(dset_id)
        elif download_type == "kagglehub":
            download_kaggle_dataset(dset_name, dset_id)
        elif download_type == "url":
            download_url_dataset(dset_name, dset_id)
