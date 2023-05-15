from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

from lib import CatPolicy, NumPolicy, TaskType

GESTURE = 'gesture'
CHURN = 'churn'
CALIFORNIA = 'california'
HOUSE = 'house'
ADULT = 'adult'
DIAMOND = 'diamond'
OTTO = 'otto'
HIGGS_SMALL = 'higgs-small'
BLACK_FRIDAY = 'black-friday'
FB_COMMENTS = 'fb-comments'
WEATHER_SMALL = 'weather-small'
COVTYPE = 'covtype'
MICROSOFT = 'microsoft'


@dataclass(frozen=True)
class DatasetInfo:
    batch_size: int
    num_policy: Optional[NumPolicy]
    task_type: TaskType
    train_size: int
    val_size: int
    test_size: int
    n_num_features: int
    n_bin_features: int
    cat_cardinalities: list[int]
    n_classes: Optional[int]

    @property
    def size(self) -> int:
        return self.train_size + self.val_size + self.test_size

    @property
    def n_cat_features(self) -> int:
        return len(self.cat_cardinalities)

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_bin_features + self.n_cat_features

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS


DATASETS = {
    # size < 10000
    GESTURE: DatasetInfo(
        batch_size=128,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.MULTICLASS,
        train_size=6318,
        val_size=1580,
        test_size=1975,
        n_num_features=32,
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=5,
    ),
    # size < 50000
    CHURN: DatasetInfo(
        batch_size=256,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.BINCLASS,
        train_size=6400,
        val_size=1600,
        test_size=2000,
        n_num_features=7,
        n_bin_features=3,
        cat_cardinalities=[9, 16, 7, 15, 6, 5, 42],
        n_classes=2,
    ),
    CALIFORNIA: DatasetInfo(
        batch_size=256,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.REGRESSION,
        train_size=13209,
        val_size=3303,
        test_size=4128,
        n_num_features=8,
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=None,
    ),
    HOUSE: DatasetInfo(
        batch_size=256,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.REGRESSION,
        train_size=14581,
        val_size=3646,
        test_size=4557,
        n_num_features=16,
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=None,
    ),
    ADULT: DatasetInfo(
        batch_size=256,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.BINCLASS,
        train_size=26048,
        val_size=6513,
        test_size=16281,
        n_num_features=6,
        n_bin_features=1,
        cat_cardinalities=[9, 16, 7, 15, 6, 5, 42],
        n_classes=2,
    ),
    # size < 200000
    DIAMOND: DatasetInfo(
        batch_size=512,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.REGRESSION,
        train_size=34521,
        val_size=8631,
        test_size=10788,
        n_num_features=6,
        n_bin_features=0,
        cat_cardinalities=[5, 7, 8],
        n_classes=None,
    ),
    OTTO: DatasetInfo(
        batch_size=512,
        num_policy=None,
        task_type=TaskType.MULTICLASS,
        train_size=39601,
        val_size=9901,
        test_size=12376,
        n_num_features=93,
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=9,
    ),
    HIGGS_SMALL: DatasetInfo(
        batch_size=512,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.BINCLASS,
        train_size=62751,
        val_size=15688,
        test_size=19610,
        n_num_features=28,
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=2,
    ),
    BLACK_FRIDAY: DatasetInfo(
        batch_size=512,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.REGRESSION,
        train_size=106764,
        val_size=26692,
        test_size=33365,
        n_num_features=4,
        n_bin_features=1,
        cat_cardinalities=[2, 7, 3, 5],
        n_classes=None,
    ),
    FB_COMMENTS: DatasetInfo(
        batch_size=512,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.REGRESSION,
        train_size=157638,
        val_size=19722,
        test_size=19720,
        n_num_features=36,
        n_bin_features=14,
        cat_cardinalities=[9, 16, 7, 15, 6, 5, 42],
        n_classes=None,
    ),
    # size < 2000000
    WEATHER_SMALL: DatasetInfo(
        batch_size=1024,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.REGRESSION,
        train_size=296554,
        val_size=47373,
        test_size=53172,
        n_num_features=118,
        n_bin_features=1,
        cat_cardinalities=[],
        n_classes=None,
    ),
    COVTYPE: DatasetInfo(
        batch_size=1024,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.MULTICLASS,
        train_size=371847,
        val_size=92962,
        test_size=116203,
        n_num_features=10,
        n_bin_features=44,
        cat_cardinalities=[],
        n_classes=7,
    ),
    MICROSOFT: DatasetInfo(
        batch_size=1024,
        num_policy=NumPolicy.QUANTILE,
        task_type=TaskType.REGRESSION,
        train_size=723412,
        val_size=235259,
        test_size=241521,
        n_num_features=131,
        n_bin_features=5,
        cat_cardinalities=[],
        n_classes=None,
    ),
}
assert all(
    # must be either both True or both False
    v.is_binclass == (v.n_classes == 2)
    for v in DATASETS.values()
)


def fill_dl_config(config: dict[str, Any], force: bool) -> None:
    c = config.get('space', config)
    info = DATASETS[c['data']['path'].rsplit('/', 1)[1]]

    method = cast(
        Callable[[dict, str, Any], Any], dict.__setitem__ if force else dict.setdefault
    )
    method(c, 'batch_size', info.batch_size)
    method(c, 'patience', 16)
    method(c, 'n_epochs', float('inf'))

    method(
        c['data'],
        'num_policy',
        None if info.num_policy is None else info.num_policy.value,
    )
    method(
        c['data'],
        'cat_policy',
        CatPolicy.ORDINAL.value if info.n_cat_features else None,
    )
    method(c['data'], 'y_policy', 'standard' if info.is_regression else None)
