# MODAL SUBMISSION

## Model
Implements the Y-VAMPX model as described in the report.
The implementations is in `models/multimodalAttention.py`. The model used is the `MultiModalAttentionRegressor` class.


## Training
Run train.py to run usual training.
Parameters can be updated in the `.yaml` files in config/.
A checkpoint is saved every $n=5$ epochs.

## Submission 
The submission is created by `workin_create_submission.py`.

## Visualisation
Some of the visalisations are computed in the two Jupyter Notebooks `analyseDonnees.ipynb` (for data visualisation) and `analyseModele.ipynb` for model visualisation. 
The visualisation of the ResNet Model and of the attention matrices are done in `resnetvisualiation.py` and `attentionvisualisation.py`.
