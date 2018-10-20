import pytest, fastai, shutil
from fastai.datasets import *
from fastai.datasets import Config, _expand_path, _url2tgz, _url2path
from pathlib import Path

Config.DEFAULT_CONFIG_PATH = 'config_test/test.yml'
Config.DEFAULT_CONFIG = { 'data_path': 'config_test/data' }


def clean_path(path):
    path = Path(path)
    if path.is_file(): path.unlink()
    else: shutil.rmtree(path)

@pytest.mark.parametrize("dataset", [
    'adult', 'mnist', 'movie_lens',
    # 'imdb',  # imdb fails unless 'en' spacy language is available
])

def test_get_samples(dataset, tmpdir):
    method = f'get_{dataset}'
    df = getattr(URLs, method)()
    assert df is not None

def test_creates_config():
    config_path = _expand_path(Config.DEFAULT_CONFIG_PATH)
    clean_path(config_path)
    assert not config_path.exists(), "config path should not exist"
    config = Config.get(config_path)
    assert config_path.exists(), "Config.get should create config if it doesn't exist"
    assert config == Config.DEFAULT_CONFIG
    assert Config.get_key('data_path') == 'config_test/data', "get_key returned wrong data_path"

def test_untar_dest():
    dest = Path('test_data')
    path = untar_data(URLs.MNIST_TINY, dest=dest)
    assert path.exists()
    