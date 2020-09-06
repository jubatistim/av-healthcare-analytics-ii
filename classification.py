# %% Data import
import pandas as pd
from sklearn.model_selection import train_test_split

raw = pd.read_csv('./healthcare/train_data.csv')
raw = raw.dropna()
raw.head()

# %% raw shape
raw.shape

# %% Split datasets
train, test = train_test_split(raw, test_size=0.05)

# %% Train shape
train.shape

# %% Test shape
test.shape

# %% Setup experiment
from pycaret.classification import *

clf1 = setup(
    train, 
    target = 'Stay',
    session_id=1945,
    normalize = True, 
    # transform_target = True, 
    ignore_features = ['case_id', 'Hospital_code', 'City_Code_Hospital', 'Hospital_region_code', 'patientid', 'City_Code_Patient', 'Visitors with Patient'],
    # polynomial_features = True, 
    feature_selection = True, 
    train_size=0.8,
    log_experiment=True,
    log_plots=True,
    use_gpu=True,
    experiment_name='jb-ex-healthcare-v01'
)

# %% compare all baseline models and select top 5
top5 = compare_models(n_select = 5) 

# %% tune top 5 base models
tuned_top5 = [tune_model(i) for i in top5]

# %% ensemble top 5 tuned models
bagged_top5 = [ensemble_model(i) for i in tuned_top5]

# %% blend top 5 base models 
blender = blend_models(estimator_list = top5) 

# %% select best model 
best = automl(optimize = 'Recall')

# %% save best model
save_model(best, 'jb-model-healthcare-v01')

# %% Predictions over test

# predict over test variable
pred_test = predict_model(best, test)

# save prediction
pred_test.to_csv('pred_test.csv', index = False)

# get statistics for this prediction
# ???

# %% Predictions for submission

# predict over test_data.csv for submission
test_data = pd.read_csv('/healthcare/test_data.csv')
submit = predict_model(best, test_data)

# save prediction
submit.to_csv('pred_test.csv', index = False)

# get statistics for this prediction
# ???

# %% TODO
# pycaret.regression.plot_model
# pycaret.regression.evaluate_model
# pycaret.regression.interpret_model
# pycaret.regression.predict_model
# pycaret.regression.finalize_model