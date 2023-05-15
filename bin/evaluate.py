import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional

import lib


def main(
    path: Path,
    n_seeds: int,
    function: Optional[str] = None,
    *,
    force: bool = False,
):
    path = lib.get_path(path)
    if path.name.endswith('-tuning'):
        from_tuning = True
        assert function is None
        assert (path / 'DONE').exists()

        tuning_report = lib.load_report(path)
        function_qualname = tuning_report['config']['function']
        template_config = tuning_report['best']['config']

        path = path.with_name(path.name.replace('tuning', 'evaluation'))
        path.mkdir(exist_ok=True)
    else:
        from_tuning = False
        assert path.name.endswith('-evaluation')
        assert function is not None
        function_qualname = function
        template_config = lib.load_config(path / '0.toml')

    function_: lib.Function = lib.import_(function_qualname)
    for seed in range(n_seeds):
        config = deepcopy(template_config)
        config['seed'] = seed
        if 'catboost' in function_qualname:
            config['model']['task_type'] = 'CPU'  # this is crucial for good results
            config['model'].setdefault('thread_count', 4)
        config_path = path / f'{seed}.toml'
        try:
            if seed > 0 or from_tuning:
                lib.dump_config(config, config_path)
            function_(config, config_path.with_suffix(''), force=force)
        except Exception:
            if seed > 0 or from_tuning:
                config_path.unlink(True)
            shutil.rmtree(config_path.with_suffix(''), True)
            raise


if __name__ == '__main__':
    lib.run_cli(main)
