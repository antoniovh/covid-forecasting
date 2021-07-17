import pathlib
import os
# import shutil


class Paths():
    def __init__(self):
        b = pathlib.Path(__file__).resolve()
        self._base = b.parents[1] / 'data'

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, val):
        self._base = pathlib.Path(val)

    @property
    def rawdir(self):
        return self.base / "raw"

    @property
    def processed(self):
        return self.base / "processed"

    @property
    def interim(self):
        return self. base / "interim"

    def __str__(self):
        return f"""
                 Base path: {self.base}
        Data download path: {self.rawdir}
        Processed data path: {self.processed}
        """

    def check_dirs(self):
        print("Checking directories...")
        print(PATHS)
        self.base.exists() or os.makedirs(self.base)
        self.interim.exists() or os.makedirs(self.interim)
        self.rawdir.exists() or os.makedirs(self.rawdir)
        self.processed.exists() or os.makedirs(self.processed)
        return None


PATHS = Paths()
