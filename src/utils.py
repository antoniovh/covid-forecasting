import pathlib
import os
# import shutil


class Paths:
    def __init__(self):
        b = pathlib.Path(__file__).resolve()
        self._base = b.parents[1]

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, val):
        self._base = pathlib.Path(val)

    @property
    def data(self):
        return self.base / 'data'

    @property
    def rawdir(self):
        return self.data / "raw"

    @property
    def processed(self):
        return self.data / "processed"

    @property
    def interim(self):
        return self.data / "interim"

    @property
    def models(self):
        return self.base / 'models'

    @property
    def source(self):
        return self.base / 'src'

    def __str__(self):
        return f"""
                Base path: {self.base}
                Data path: {self.data}
              Source path: {self.source}
              Models path: {self.models}
        """

    def check_dirs(self):
        print("Checking directories...")
        self.base.exists() or os.makedirs(self.base)
        self.data.exists() or os.makedirs(self.data)
        self.interim.exists() or os.makedirs(self.interim)
        self.rawdir.exists() or os.makedirs(self.rawdir)
        self.processed.exists() or os.makedirs(self.processed)
        self.source.exists() or os.makedirs(self.source)
        self.models.exists() or os.makedirs(self.models)
        print('Ok!')
        print(Paths())
        return None


PATHS = Paths()