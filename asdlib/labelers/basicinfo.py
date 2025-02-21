from typing import List

from asdlib.datasets.collators import get_dcase_info
from sklearn.preprocessing import LabelEncoder

from .base import LabelerBase


class BasicInfoLabeler(LabelerBase):
    def __init__(self, label_name: str):
        # "machine_attr_domain""
        self.label_name_split = []
        for l in label_name.split("_"):
            self.label_name_split.append(l.replace("domain", "is_target"))
        self.le = LabelEncoder()

    def get_label(self, path: str) -> str:
        label_split: List[str] = []
        for l in self.label_name_split:
            label_split.append(str(get_dcase_info(path, l)))
        return "---".join(label_split)

    def fit(self, all_path_list: List[str]) -> None:
        all_label_list = []
        for path in all_path_list:
            all_label_list.append(self.get_label(path))
        self.le.fit(all_label_list)
        self.num_class = len(self.le.classes_)

    def trans(self, path: str) -> int:
        return self.le.transform([self.get_label(path)])[0].item()  # type: ignore
