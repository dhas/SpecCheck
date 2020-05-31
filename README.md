# SpecCheck
This repository contains the code and data which can reproduce results reported in our article 'Does the data meet your expectations? Explaining bias in a dataset' submitted to NeurIPS 2020. Briefly, the idea is to explain sample representation and bias in a dataset of greyscale images of circles and squares in terms of intuitive aspects such as size, position and pixel brightness.

## Demo
1. Make sure to install dependencies listed in requirements.txt
2. We use an Nvidia GTX 1080Ti GPU, but it may be possible to replicate results without using a GPU
3. To run the full experiment from scratch, run
```bat
python run_test.py --test_config=config/test_draw_128.py
```
This will prompt the user to place source data files from Quick,Draw! with Google (https://quickdraw.withgoogle.com/data)

4. To use pre-trained models and previously collected data, download this file and untar it to SpecCheck/_tests, before running the same command
