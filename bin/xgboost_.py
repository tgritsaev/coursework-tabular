# XGBoost
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

import delu
import lib
from lib import KWArgs


# DO NOT modify the config neither in __post_init__ nor during execution
@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: KWArgs  # xgboost.{XGBClassifier,XGBRegressor}
    fit: KWArgs  # xgboost.{XGBClassifier,XGBRegressor}.fit

    def __post_init__(self):
        assert 'random_state' not in self.model
        if isinstance(self.data, dict):
            assert self.data['cat_policy'] in [None, 'one-hot']
        assert (
            'early_stopping_rounds' in self.fit
        ), 'XGBoost does not automatically use the best model, so early stopping must be used'
        use_gpu = self.model.get('tree_method') == 'gpu_hist'
        if use_gpu:
            assert os.environ.get('CUDA_VISIBLE_DEVICES')


@lib.has_logger
def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    # >>> start
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    logger = lib.get_logger(output / '0.log')
    # all modifications to `config` must be done BEFORE creating the report
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.improve_reproducibility(C.seed)

    # >>> data
    dataset = C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    dataset = dataset.merge_num_bin()
    assert set(dataset.data.keys()) == {'X_num', 'Y'}, set(dataset.data.keys())
    assert dataset.X_num is not None  # for type checker

    # >>> model
    model_kwargs = C.model | {'random_state': C.seed}
    if dataset.is_regression:
        model = XGBRegressor(**model_kwargs)
        predict = model.predict
        fit_extra_kwargs = {}
    else:
        model = XGBClassifier(**model_kwargs, disable_default_eval_metric=True)
        if dataset.is_multiclass:
            predict = model.predict_proba
            fit_extra_kwargs = {'eval_metric': 'merror'}
        else:
            predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
            fit_extra_kwargs = {'eval_metric': 'error'}
    report['prediction_type'] = None if dataset.is_regression else 'probs'

    # >>> training
    logger.info('training...')
    with delu.Timer() as timer:
        model.fit(
            dataset.X_num['train'],
            dataset.Y['train'],
            eval_set=[(dataset.X_num['val'], dataset.Y['val'])],
            **C.fit,
            **fit_extra_kwargs,
        )
    report['time'] = str(timer)

    # >>> finish
    logger.info('finishing...')
    model.save_model(str(output / "model.xgbm"))
    np.save(output / "feature_importances.npy", model.feature_importances_)
    predictions = {k: predict(v) for k, v in dataset.X_num.items()}
    report['metrics'] = dataset.calculate_metrics(
        predictions, report['prediction_type']  # type: ignore[code]
    )
    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.run_Function_cli(main)
