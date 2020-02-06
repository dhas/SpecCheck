from pathlib import Path
import prepare_datasets

plots_dir = Path('./_plots')
plots_dir.mkdir(exist_ok=True)

prepare_datasets.main(plots_dir)