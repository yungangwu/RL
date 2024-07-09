import io
import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ModelStorage(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    async def save(self, name: str, version: int, data: bytes):
        pass

    @abstractmethod
    async def load(self, name: str, version: int) -> bytes:
        pass

class FileModelStorage(ModelStorage):
    def __init__(self, dir: str = '.') -> None:
        super().__init__()
        self._dir = dir

    def _get_dir(self, name: str):
        return os.path.join(self._dir, name)

    def _get_file_name(self, name: str, version: int):
        return f'{name}_model_{version}.mdl'

    async def save(self, name: str, version: int, data: bytes):
        file_dir = self._get_dir(name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        file_name = os.path.join(file_dir, self._get_file_name(name, version))
        with open(file_name, 'wb') as f:
            f.write(data)
        logger.info(f'save model [{name}] version [{version}] to file: {file_name}.')

    async def load(self, name: str, version: int) -> bytes:
        file_name = os.path.join(self._get_dir(name), self._get_file_name(name, version))
        if os.path.exists(file_name):
            logger.info(f'load model [{name}] version [{version}] from file: {file_name}.')
            with open(file_name, 'rb') as f:
                return f.read()