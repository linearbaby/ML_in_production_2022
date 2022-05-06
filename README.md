# ML_production_IZ1
Project structure:
```
ml_project/
├── artifacts   <--- possible main artifacts storage
├── configs     <--- Hydra configs 
├── data        <--- datasets, data management
├── docs        <--- documentation
├── eval.py     <--- evaluation script
├── features    <--- feature processing and extruction
├── metrics     <--- possible main metrics storage
├── models      <--- model utils
├── notebooks   <--- notebooks research
├── pipeline.py <--- training script
├── settings    <--- possible global project settings
├── testing     <--- tests folder
└── utils       <--- project utils folder
```

## _Installing the repository and running the virtual environment_

Change your working directory to ( for example `cd /path/to/your/dir/` ). Execute underling code:
```
git fork https://github.com/ArtemPushPop/ML_in_production_2022.git .
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cd ml_project
```
After fishing previous steps, yo are ready for ML.

## _Training models_
Training models is as simple as running this script
```
python pipeline.py
```
This script will get model according to project configuration (default: LogReg), fit it with training data, fetched from internet, or located on your machine, and save trained model and metrics in hydra outputs folder with self-describing name.<br>
You can easily parametrize training script. For example training GaussianNB model is done with next script:
```
python pipeline.py model=gaussian_naive_bayes
```

## _Evaluating results_
For evaluating previously trained model, one need to call evaluating script 
```
python eval.py
```
This script will load most previously trained model from hydra outputs (if no models were trained, script will alert about this), predict with this model test data, loaded from internet or located on your machine, and save results in hydra outputs folder with self-describing name.<br>
Like training script, if there is a need to evaluate GaussianNB model one need to parametrize `eval.py` script
```
python eval.py model=gaussian_naive_bayes
```