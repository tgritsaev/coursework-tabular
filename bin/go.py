from pathlib import Path
from typing import Optional

import bin.ensemble
import bin.evaluate
import bin.tune
import lib


def main(
    path: Path,  # "a/b/c/0-tuning" OR "a/b/c/0-evaluation"
    function: Optional[str] = None,
    n_seeds: int = 15,
    n_ensembles: int = 3,
    ensemble_size: int = 5,
    *,
    continue_: bool = False,
    force: bool = False,
):
    path = lib.get_path(path)
    if path.name.endswith(('-tuning', '-tuning.toml')):
        assert function is None
        tuning_output = path.with_suffix('')
        tuning_config = lib.load_config(path)
        bin.tune.main(tuning_config, tuning_output, continue_=continue_, force=force)
        evaluation_input = tuning_output
        evaluation_dir = tuning_output.with_name(
            tuning_output.name.replace('tuning', 'evaluation')
        )
    else:
        assert function is not None
        evaluation_input = path / '0.toml'
        evaluation_dir = path.parent

    bin.evaluate.main(evaluation_input, n_seeds, function, force=force)
    bin.ensemble.main(evaluation_dir, n_ensembles, ensemble_size, force=force)


if __name__ == '__main__':
    lib.run_cli(main)
