import logging
import io
from krl.model.model_storage import FileModelStorage

logger = logging.getLogger(__name__)

class ModelCollection(object):
    def __init__(self, model_saver) -> None:
        super().__init__()
        self._latest_version = 0
        self._model_saver = model_saver

    def add_model(self, data: bytes):
        self._latest_version += 1
        self._model = io.BytesIO(data)
        return self._latest_version

    def set_model(self, version: int, data: bytes):
        self._model = io.BytesIO(data)
        self._latest_version = version

    def get_latest_version(self):
        return self._latest_version

    def get_model(self):
        return self._model

class ModelSaver(object):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self._config = config
        storage = config.get('storage', {})
        storage_type = storage.get('type', 'file')
        if storage_type == 'file':
            dir = storage.get('dir', './saved_models')
            self._storage = FileModelStorage(dir)
        else:
            raise RuntimeError(f'model storage type {storage_type} is invalid.')
        self._model_collections = {}

        self._startup = config.get('startup', None)

    async def startup(self):
        if self._startup:
            for name, options in self._startup.items():
                version = options.get('version', 1)
                data = await self.load(name=name, version=version)
                if data:
                    self[name].set_model(version, data)

    async def save(self, name: str, version: int, data: bytes):
        await self._storage.save(name, version, data)

    async def load(self, name: str, version: int):
        return await self._storage.load(name, version)

    async def get_model(self, key: str, version: int):
        collection: ModelCollection = self[key]
        if version == collection.get_latest_version():
            return collection.get_model()
        else:
            data = await self.load(key, version)
            return io.BytesIO(data)