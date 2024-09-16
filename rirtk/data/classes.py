import re
from pathlib import Path


class Classes:
    silence_name = '<sil>'

    def __init__(self, classes, utid_regexp=None):
        self.utid_regexp = None if utid_regexp is None else re.compile(utid_regexp)
        classes = Path(classes)
        assert classes.is_file(), f'File {classes} does not exist'
        lines = classes.read_text().strip().split('\n')
        self.name_id = dict()
        self.id_name = dict()
        self.silence_id = None
        index = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            name = parts[0]
            assert name not in self.name_id, f'Duplicate class ID found {name} in file {classes}'
            self.name_id[name] = index
            self.id_name[index] = name
            if name == self.silence_name:
                self.silence_id = index
            index += 1
        assert self.silence_id is not None, f'Failed to find silence class with name {silence} in file {classes}'
        self.num_classes = index

    def name_by_id(self, clsid):
        assert clsid in self.id_name, f'Failed to find class ID for class {clsid}'
        return self.id_name[clsid]

    def id_by_name(self, name):
        assert name in self.name_id, f'Failed to find class with name {name}'
        return self.name_id[name]

    def name_by_utid(self, utid):
        match = self.utid_regexp.match(utid)
        assert match is not None, f'Failed to parse utterance ID {utid}'
        name = match.group(1)
        assert name in self.name_id, f'Failed to find class with name {name}'
        return name

    def id_by_utid(self, utid):
        return self.id_by_name(self.name_by_utid(utid))
