# this file must not import anything from lib except for lib/env.py

import argparse
import dataclasses
import datetime
import enum
import functools
import importlib
import inspect
import json
import os
import pickle
import shutil
import sys
import time
import typing
import warnings
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Optional, Type, TypeVar, Union

import numpy as np
import tomli
import tomli_w
import torch
from loguru._logger import Logger

import delu

from . import env

# ======================================================================================
# >>> logging <<<
# ======================================================================================
_LOGGER: Optional[Logger] = None
_logfiles = set()


def get_logger(path: Union[None, str, Path] = None) -> Logger:
    global _LOGGER
    format_ = '<level>{elapsed}</level> | {message}'

    if _LOGGER is None:
        from loguru import logger

        logger.remove()
        timer = run_timer()
        logger = logger.patch(
            lambda x: x.update(elapsed=datetime.timedelta(seconds=int(timer())))  # type: ignore[code]
        )
        logger.add(sys.stderr, format=format_)
        _LOGGER = logger  # type: ignore[code]

    assert _LOGGER is not None
    if path is not None:
        path = env.get_path(path)
        assert path not in _logfiles
        _LOGGER.add(path, format=format_)  # type: ignore[code]
        _logfiles.add(path)

    return _LOGGER


def has_logger(f):
    @functools.wraps(f)
    def new_f(*args, use_current_logger: bool = False, **kwargs):
        if use_current_logger:
            return f(*args, **kwargs)
        else:
            global _LOGGER
            logger = _LOGGER
            try:
                _LOGGER = None
                return f(*args, **kwargs)
            finally:
                _LOGGER = logger

    return new_f


# ======================================================================================
# >>> types <<<
# ======================================================================================
KWArgs = dict[str, Any]
JSONDict = dict[str, Any]  # must be JSON-serializable
T = TypeVar('T')


# ======================================================================================
# >>> enums <<<
# ======================================================================================
class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'


class Part(enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


# ======================================================================================
# >>> IO <<<
# ======================================================================================
def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


# ======================================================================================
# >>> Function <<<
# ======================================================================================
# "Function" is any function with the following signature:
# Function: (
#     config: JSONDict,
#     output: Union[str, Path],
#     *,
#     force = False,
#     [continue_ = False]
# ) -> Optional[JSONDict]
Function = Callable[..., Optional[JSONDict]]


def start(
    output: Union[str, Path], force: bool = False, continue_: bool = False
) -> bool:
    """Start Function."""
    print_sep('=')
    print(f'[>>>] {datetime.datetime.now()}')
    output = env.get_path(output)
    logger = get_logger()

    if output.exists():
        if force:
            logger.warning('Removing the existing output')
            shutil.rmtree(output)
            output.mkdir()
            return True
        elif not continue_:
            backup_output(output)
            logger.warning(f'Already exists! | {env.try_get_relative_path(output)}\n')
            return False
        elif output.joinpath('DONE').exists():
            backup_output(output)
            logger.info('Already done!\n')
            return False
        else:
            logger.info('Continuing with the existing output')
            return True
    else:
        logger.info('Creating the output')
        output.mkdir()
        return True


def make_config(Config: Type[T], config: JSONDict) -> T:
    assert is_dataclass(Config) or Config is dict

    if isinstance(config, Config):
        the_config = config
    else:
        assert is_dataclass(Config)

        def _from_dict(datacls: type[T], data: dict) -> T:
            # this is an intentionally restricted parsing which
            # supports only nested (optional) dataclasses,
            # but not unions and collections thereof
            assert is_dataclass(datacls)
            data = deepcopy(data)
            for field in dataclasses.fields(datacls):
                if field.name not in data:
                    continue
                if is_dataclass(field.type):
                    data[field.name] = _from_dict(field.type, data[field.name])
                # check if Optional[<dataclass>]
                elif (
                    typing.get_origin(field.type) is Union
                    and len(typing.get_args(field.type)) == 2
                    and typing.get_args(field.type)[1] is type(None)
                    and is_dataclass(typing.get_args(field.type)[0])
                ):
                    if data[field.name] is not None:
                        data[field.name] = _from_dict(
                            typing.get_args(field.type)[0], data[field.name]
                        )
                else:
                    # in this case, we do nothing and hope for good luck
                    pass

            return datacls(**data)

        the_config = _from_dict(Config, config)

    print_sep()
    pprint(
        asdict(the_config) if is_dataclass(the_config) else the_config,
        sort_dicts=False,
        width=100,
    )
    print_sep()
    return the_config


def create_report(config: JSONDict) -> JSONDict:
    # report is just a JSON-serializable Python dictionary
    # for storing arbitrary information about a given run.

    report = {}
    # If this snippet succeeds, then report['function'] will the full name of the
    # function relative to the project directory (e.g. "bin.catboost_.main")
    try:
        caller_frame = inspect.stack()[1]
        caller_relative_path = env.get_path(caller_frame.filename).relative_to(
            env.PROJECT_DIR
        )
        caller_module = str(caller_relative_path.with_suffix('')).replace('/', '.')
        caller_function_qualname = f'{caller_module}.{caller_frame.function}'
        import_(caller_function_qualname)
        report['function'] = caller_function_qualname
    except Exception as err:
        warnings.warn(
            f'The key "function" will be missing in the report. Reason: {err}'
        )
    report['environment'] = (
        {
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpus': delu.hardware.get_gpus_info(),
            'torch.version.cuda': torch.version.cuda,  # type: ignore[code]
            'torch.backends.cudnn.version()': torch.backends.cudnn.version(),  # type: ignore[code]
            'torch.cuda.nccl.version()': torch.cuda.nccl.version(),  # type: ignore[code]
        }
        if torch.cuda.is_available()
        else {}
    )

    def jsonify(value):
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        elif isinstance(value, list):
            return [jsonify(x) for x in value]
        elif isinstance(value, dict):
            return {k: jsonify(v) for k, v in value.items()}
        else:
            return '<nonserializable>'

    report['config'] = jsonify(config)
    return report


def summarize(report: JSONDict) -> JSONDict:
    summary = {'function': report.get('function')}

    if 'best' in report:
        summary['best'] = summarize(report['best'])
    else:
        env = report.get('environment')
        if env is not None:
            summary['devices'] = (
                [
                    env['gpus']['devices'][i]['name']
                    for i in map(int, env['CUDA_VISIBLE_DEVICES'].split(','))
                ]
                if 'CUDA_VISIBLE_DEVICES' in env
                else ['CPU']
            )

    for key in ['n_parameters', 'best_epoch', 'tuning_time', 'trial_id']:
        if key in report:
            summary[key] = deepcopy(report[key])

    metrics = report.get('metrics')
    if metrics is not None and 'score' in next(iter(metrics.values())):
        summary['scores'] = {part: metrics[part]['score'] for part in metrics}

    for key in ['n_completed_trials', 'time']:
        if key in report:
            summary[key] = deepcopy(report[key])

    return summary


def run_Function_cli(function: Function, argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE')
    parser.add_argument('--force', action='store_true')
    if 'continue_' in inspect.signature(function).parameters:
        can_continue_ = True
        parser.add_argument('--continue', action='store_true', dest='continue_')
    else:
        can_continue_ = False
    args = parser.parse_args(*(() if argv is None else (argv,)))

    # >>> snippet for the internal infrastructure, ignore it
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if snapshot_dir and Path(snapshot_dir).joinpath('CHECKPOINTS_RESTORED').exists():
        assert can_continue_ and args.continue_
    # <<<

    config_path = env.get_path(args.config)
    assert config_path.exists()
    function(
        load_config(config_path),
        config_path.with_suffix(''),
        force=args.force,
        **({'continue_': args.continue_} if can_continue_ else {}),
    )


_LAST_SNAPSHOT_TIME = None


def backup_output(output: Path) -> None:
    """
    This is a function for the internal infrastructure, ignore it.
    """
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output.relative_to(env.PROJECT_DIR)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output = dir_ / relative_output_dir
        prev_backup_output = new_output.with_name(new_output.name + '_prev')
        new_output.parent.mkdir(exist_ok=True, parents=True)
        if new_output.exists():
            new_output.rename(prev_backup_output)
        shutil.copytree(output, new_output)
        # the case for evaluate.py which automatically creates configs
        if output.with_suffix('.toml').exists():
            shutil.copyfile(
                output.with_suffix('.toml'), new_output.with_suffix('.toml')
            )
        if prev_backup_output.exists():
            shutil.rmtree(prev_backup_output)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        import nirvana_dl.snapshot  # type: ignore[code]

        nirvana_dl.snapshot.dump_snapshot()
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')


def finish(output: Path, report: JSONDict) -> None:
    dump_json(report, output / 'report.json')
    json_output_path = os.environ.get('JSON_OUTPUT_FILE')
    if json_output_path:
        try:
            key = str(output.relative_to(env.PROJECT_DIR))
        except ValueError:
            pass
        else:
            json_output_path = Path(json_output_path)
            try:
                json_data = json.loads(json_output_path.read_text())
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                json_data = {}
            json_data[key] = load_report(output)
            json_output_path.write_text(json.dumps(json_data, indent=4))
        shutil.copyfile(
            json_output_path,
            os.path.join(os.environ['SNAPSHOT_PATH'], 'json_output.json'),
        )

    output.joinpath('DONE').touch()
    backup_output(output)
    print_sep()
    print(f'[<<<] {datetime.datetime.now()}')
    try:
        print_summary(output)
    except FileNotFoundError:
        pass
    print_sep()
    get_logger().info(f'Done! | {env.try_get_relative_path(output)}')


# ======================================================================================
# >>> output <<<
# ======================================================================================
_TOML_CONFIG_NONE = '__null__'


def _process_toml_config(data, load) -> JSONDict:
    if load:
        # replace _TOML_CONFIG_NONE with None
        condition = lambda x: x == _TOML_CONFIG_NONE  # noqa
        value = None
    else:
        # replace None with _TOML_CONFIG_NONE
        condition = lambda x: x is None  # noqa
        value = _TOML_CONFIG_NONE

    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)  # type: ignore[code]


def load_config(output_or_config_path: Union[str, Path]) -> JSONDict:
    with open(env.get_path(output_or_config_path).with_suffix('.toml'), 'rb') as f:
        return _process_toml_config(tomli.load(f), True)


def dump_config(config: JSONDict, output_or_config_path: Union[str, Path]) -> None:
    path = env.get_path(output_or_config_path).with_suffix('.toml')
    with open(path, 'wb') as f:
        tomli_w.dump(_process_toml_config(config, False), f)
    assert config == load_config(path)  # sanity check


def load_report(output: Union[str, Path]) -> JSONDict:
    return load_json(env.get_path(output) / 'report.json')


def dump_report(report: JSONDict, output: Union[str, Path]) -> None:
    dump_json(report, env.get_path(output) / 'report.json')


def load_summary(output: Union[str, Path]) -> JSONDict:
    return load_json(env.get_path(output) / 'summary.json')


def print_summary(output: Union[str, Path]):
    pprint(load_summary(output), sort_dicts=False, width=60)


def dump_summary(summary: JSONDict, output: Union[str, Path]) -> None:
    dump_json(summary, env.get_path(output) / 'summary.json')


def load_predictions(output: Union[str, Path]) -> dict[str, np.ndarray]:
    x = np.load(env.get_path(output) / 'predictions.npz')
    return {key: x[key] for key in x}


def dump_predictions(
    predictions: dict[str, np.ndarray], output: Union[str, Path]
) -> None:
    np.savez(env.get_path(output) / 'predictions.npz', **predictions)


def get_checkpoint_path(output: Union[str, Path]) -> Path:
    return env.get_path(output) / 'checkpoint.pt'


def load_checkpoint(output: Union[str, Path], **kwargs) -> JSONDict:
    return torch.load(get_checkpoint_path(output), **kwargs)


def dump_checkpoint(checkpoint: JSONDict, output: Union[str, Path], **kwargs) -> None:
    torch.save(checkpoint, get_checkpoint_path(output), **kwargs)


# ======================================================================================
# >>> other <<<
# ======================================================================================
def run_timer():
    timer = delu.Timer()
    timer.run()
    return timer


def import_(qualname: str) -> Any:
    # example: import_('bin.catboost_.main')
    try:
        module, name = qualname.rsplit('.', 1)
        return getattr(importlib.import_module(module), name)
    except Exception as err:
        raise ValueError(f'Cannot import "{qualname}"') from err


def get_device() -> torch.device:
    if torch.cuda.is_available():
        assert os.environ.get('CUDA_VISIBLE_DEVICES') is not None
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def is_oom_exception(err: RuntimeError) -> bool:
    return any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


def run_cli(fn: Callable, argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    for name, arg in inspect.signature(fn).parameters.items():
        origin = typing.get_origin(arg.annotation)
        if origin is Union:
            # must be optional
            assert len(typing.get_args(arg.annotation)) == 2 and (
                typing.get_args(arg.annotation)[1] is type(None)
            )
            assert arg.default is None
            type_ = typing.get_args(arg.annotation)[0]
        else:
            assert origin is None
            type_ = arg.annotation

        has_default = arg.default is not inspect.Parameter.empty
        if arg.annotation is bool:
            if not has_default or not arg.default:
                parser.add_argument('--' + name, action='store_true')
            else:
                parser.add_argument('--no-' + name, action='store_false', dest=name)
        else:
            assert type_ in [int, float, str, bytes, Path] or issubclass(
                type_, enum.Enum
            )
            parser.add_argument(
                ('--' if has_default else '') + name,
                type=((lambda x: bytes(x, 'utf8')) if type_ is bytes else type_),
                **({'default': arg.default} if has_default else {}),
            )
    args = parser.parse_args(*(() if argv is None else (argv,)))
    return fn(**vars(args))


def print_sep(ch='-'):
    print(ch * 100)
