from pathlib import Path
import prepare_datasets
import prepare_encoders

out_dir = Path('./_outputs')
out_dir.mkdir(exist_ok=True)

prepare_datasets.main(out_dir)
prepare_encoders.main(out_dir)