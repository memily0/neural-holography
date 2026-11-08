"""Download the original datasets with published archive checksums.

Checksum sources are torchvision v0.21.0 datasets/{mnist,cifar}.py.
The script uses only the Python standard library; inference never needs it.
"""

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class Config:
    root: Path = Path("data")
    datasets: tuple = ("cifar10", "mnist", "fashionmnist")
    verify_only: bool = False


MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
FASHION_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
CIFAR_FILES = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
}
CIFAR_URLS = (
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://zenodo.org/records/10089977/files/cifar-10-python.tar.gz?download=1",
)


def stream_md5(stream):
    digest = hashlib.md5()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def file_md5(path):
    with Path(path).open("rb") as stream:
        return stream_md5(stream)


def download(urls, path, checksum, verify_only=False):
    path = Path(path)
    if path.exists() and file_md5(path) == checksum:
        return {"path": str(path), "md5": checksum, "status": "verified"}
    if verify_only:
        raise RuntimeError(f"Missing or checksum-mismatched archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    errors = []
    for url in urls:
        try:
            request = Request(url, headers={"User-Agent": "neural-holography-dataset-downloader/1.0"})
            with urlopen(request, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            actual = file_md5(partial)
            if actual != checksum:
                raise ValueError(f"Expected MD5 {checksum}, received {actual}")
            partial.replace(path)
            return {"path": str(path), "url": url, "md5": checksum, "status": "downloaded"}
        except (OSError, URLError, ValueError) as error:
            errors.append(f"{url}: {error}")
            partial.unlink(missing_ok=True)
    raise RuntimeError("Download failed:\n" + "\n".join(errors))


def unpack_gzip(path, verify_only=False):
    path = Path(path)
    output = path.with_suffix("")
    with gzip.open(path, "rb") as stream:
        checksum = stream_md5(stream)
    if output.exists() and file_md5(output) == checksum:
        return output
    if verify_only:
        raise RuntimeError(f"Missing or corrupted uncompressed file: {output}")
    partial = output.with_name(output.name + ".part")
    with gzip.open(path, "rb") as source, partial.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    if file_md5(partial) != checksum:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Uncompressed checksum mismatch: {output}")
    partial.replace(output)
    return output


def prepare_idx(root, dataset, files, base_url, verify_only=False):
    directory = root / dataset / "raw"
    records = []
    for filename, checksum in files.items():
        path = directory / filename
        record = download((base_url + filename,), path, checksum, verify_only)
        raw = unpack_gzip(path, verify_only)
        record.update({"uncompressed_path": str(raw), "uncompressed_md5": file_md5(raw)})
        records.append(record)
    return records


def prepare_cifar(root, verify_only=False):
    path = root / "cifar-10-python.tar.gz"
    record = download(CIFAR_URLS, path, "c58f30108f718f92721af3b95e74349a", verify_only)
    directory = root / "cifar-10-batches-py"
    if not verify_only:
        directory.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(path, "r:gz") as archive:
        for filename, checksum in CIFAR_FILES.items():
            output = directory / filename
            if not output.exists() or file_md5(output) != checksum:
                if verify_only:
                    raise RuntimeError(f"Missing or corrupted CIFAR file: {output}")
                # Read known regular members only; no extractall, links or arbitrary paths.
                member = archive.getmember("cifar-10-batches-py/" + filename)
                if not member.isfile():
                    raise RuntimeError(f"Expected a regular archive member: {member.name}")
                partial = output.with_name(output.name + ".part")
                with archive.extractfile(member) as source, partial.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                if file_md5(partial) != checksum:
                    partial.unlink(missing_ok=True)
                    raise RuntimeError(f"Extracted checksum mismatch: {output}")
                partial.replace(output)
            extracted.append({"path": str(output), "md5": checksum})
    record["extracted"] = extracted
    return [record]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", choices=("cifar10", "mnist", "fashionmnist"), default=list(Config.datasets))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    cfg = Config(args.root, tuple(args.datasets), args.verify_only)
    records = {}
    if "cifar10" in cfg.datasets:
        records["cifar10"] = prepare_cifar(cfg.root, cfg.verify_only)
    if "mnist" in cfg.datasets:
        records["mnist"] = prepare_idx(cfg.root, "MNIST", MNIST_FILES, "https://ossci-datasets.s3.amazonaws.com/mnist/", cfg.verify_only)
    if "fashionmnist" in cfg.datasets:
        records["fashionmnist"] = prepare_idx(cfg.root, "FashionMNIST", FASHION_FILES, "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/", cfg.verify_only)
    print(json.dumps({"root": str(cfg.root.resolve()), "verify_only": cfg.verify_only, "datasets": records}, indent=2))


if __name__ == "__main__":
    main()
