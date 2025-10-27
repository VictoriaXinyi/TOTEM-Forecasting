import argparse
import numpy as np
import os
from lib.utils.env import get_default_output_dir, ensure_dir
from tqdm.auto import tqdm

def _load_split_data(file_path):
    try:
        arr = np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"加载失败: {file_path}. 请确保该 .npy 文件存在且未损坏。原始错误: {e}") from e
    try:
        arr = np.asarray(arr, dtype=np.float32)
    except Exception:
        dtype = getattr(arr, 'dtype', 'unknown')
        raise RuntimeError(f"数据类型不兼容: {file_path}. 期望数值数组可转换为 float32，实际 dtype: {dtype}")
    if arr.ndim != 2:
        raise ValueError(f"期望二维数组形状 [N, T]，实际维度为 {arr.ndim}: {file_path}")
    return arr

def create_full_data(args, split='train'):
    datasets = ['weather', 'electricity', 'traffic', 'ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']
    arrays = []
    missing = []
    time_dims = []
    for name in tqdm(datasets, desc=f"{split}", unit="dataset"):
        file_path = os.path.join(args.root_path, name, f"{split}_data_x.npy")
        if not os.path.exists(file_path):
            missing.append(file_path)
            continue
        arr = _load_split_data(file_path)
        arrays.append(arr)
        time_dims.append(arr.shape[1])

    if missing:
        details = "\n".join(missing)
        raise FileNotFoundError(f"以下 {split} 分割文件缺失，无法合并:\n{details}")

    if arrays and len(set(time_dims)) > 1:
        details = ", ".join([f"{datasets[i]}:T={time_dims[i]}" for i in range(len(time_dims))])
        raise ValueError(f"{split} 数据的时间维度不一致，无法拼接。请检查各数据集的 pred_len 是否统一。详情: {details}")

    data = np.concatenate(tuple(arrays), axis=0)

    ensure_dir(args.save_path)
    np.save(os.path.join(args.save_path, f"{split}_data_x.npy"), data)


def main(args):
    create_full_data(args, 'train')
    create_full_data(args, 'val')
    create_full_data(args, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root_path', type=str, default='', help='root path of the data folders')
    parser.add_argument('--save_path', type=str, default='', help='where to save the combined data')
    args = parser.parse_args()
    if args.root_path in (None, '', 'None'):
        args.root_path = os.path.join(get_default_output_dir(), 'data')
    if args.save_path in (None, '', 'None'):
        args.save_path = os.path.join(get_default_output_dir(), 'data', 'all')
    ensure_dir(args.save_path)
    main(args)