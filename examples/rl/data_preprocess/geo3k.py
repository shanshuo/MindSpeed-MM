# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os
import shutil
import logging
import datasets

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

_HDFS_PREFIX = "hdfs://"

_HDFS_BIN_PATH = shutil.which('hdfs')


def exists(path: str, **kwargs) -> bool:
    r"""Works like os.path.exists() but supports hdfs.

    Test whether a path exists. Returns False for broken symbolic links.

    Args:
        path (str): path to test

    Returns:
        bool: True if the path exists, False otherwise
    """
    if _is_non_local(path):
        return _exists(path, **kwargs)
    return os.path.exists(path)


def _exists(file_path: str):
    """ hdfs capable to check whether a file_path is exists """
    if file_path.startswith("hdfs"):
        return _run_cmd(_hdfs_cmd(f"-test -e {file_path}")) == 0
    return os.path.exists(file_path)


def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    r"""Works like os.makedirs() but supports hdfs.

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.

    Args:
        name (str): directory to create
        mode (int): file mode bits
        exist_ok (bool): if True, do not raise an exception if the directory already exists
        kwargs: keyword arguments for hdfs

    """
    if _is_non_local(name):
        # TODO(haibin.lin):
        # - handle OSError for hdfs(?)
        # - support exist_ok for hdfs(?)
        _mkdir(name, **kwargs)
    else:
        os.makedirs(name, mode=mode, exist_ok=exist_ok)


def _mkdir(file_path: str) -> bool:
    """hdfs mkdir"""
    if file_path.startswith("hdfs"):
        _run_cmd(_hdfs_cmd(f"-mkdir -p {file_path}"))
    else:
        os.makedirs(file_path, exist_ok=True)
    return True


def copy(src: str, dst: str, **kwargs) -> bool:
    r"""Works like shutil.copy() for file, and shutil.copytree for dir, and supports hdfs.

    Copy data and mode bits ("cp src dst"). Return the file's destination.
    The destination may be a directory.
    If source and destination are the same file, a SameFileError will be
    raised.

    Arg:
        src (str): source file path
        dst (str): destination file path
        kwargs: keyword arguments for hdfs copy

    Returns:
        str: destination file path

    """
    if _is_non_local(src) or _is_non_local(dst):
        # TODO(haibin.lin):
        # - handle SameFileError for hdfs files(?)
        # - return file destination for hdfs files
        return _copy(src, dst)
    else:
        if os.path.isdir(src):
            return shutil.copytree(src, dst, **kwargs)
        else:
            return shutil.copy(src, dst, **kwargs)


def _copy(from_path: str, to_path: str, timeout: int = None) -> bool:
    if to_path.startswith("hdfs"):
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(_hdfs_cmd(f"-cp -f {from_path} {to_path}"), timeout=timeout)
        else:
            returncode = _run_cmd(_hdfs_cmd(f"-put -f {from_path} {to_path}"), timeout=timeout)
    else:
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(_hdfs_cmd(f"-get \
                {from_path} {to_path}"), timeout=timeout)
        else:
            try:
                shutil.copy(from_path, to_path)
                returncode = 0
            except shutil.SameFileError:
                returncode = 0
            except Exception as e:
                logger.warning(f"copy {from_path} {to_path} failed: {e}")
                returncode = -1
    return returncode == 0


def _run_cmd(cmd: str, timeout=None):
    return os.system(cmd)


def _hdfs_cmd(cmd: str) -> str:
    return f"{_HDFS_BIN_PATH} dfs {cmd}"


def _is_non_local(path: str):
    return path.startswith(_HDFS_PREFIX)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/geo3k')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--local_data', default=None)

    args = parser.parse_args()

    data_source = 'hiyouga/geometry3k'

    if args.local_data is None:
        dataset = datasets.load_dataset(data_source)
    else:
        dataset = datasets.load_dataset(args.local_data)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = (
        r'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. '
        r'The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.'
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            problem = example.pop('problem')
            prompt = problem + ' ' + instruction_following
            answer = example.pop('answer')
            images = example.pop('images')

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "images": images,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": problem,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)