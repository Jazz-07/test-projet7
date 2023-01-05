# librairies
from joblib import load
from typing import Any,Optional
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.utils import GetterDict
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
import json

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import jsbeautifier
import cssbeautifier

opts = jsbeautifier.default_options()
opts.indent_size = 2
opts.space_in_empty_paren = True
res = jsbeautifier.beautify('some javascript', opts)






# Définir un objet pour realiser les requetes
# dot notation (.)
class Data (BaseModel):
    EXT_SOURCE_1 : float
    EXT_SOURCE_2 : float
    EXT_SOURCE_3 : float
    PAYMENT_RATE : float
    DAYS_EMPLOYED :float




import re
import ast





# Class pour entrainer le modele 
class DataModel:

    def __init__(self):
        
        self.df = pd.read_csv('donnees')
    
        self.model_fname_ = 'Mon_modele.pkl'
        try:
            self.model = load(self.model_fname_)
        except Exception as _:
            self.model = self._train_model()
            json.dump(self.model, self.model_fname_)
    
    # Fonction d'entrainement du modele
    def _train_model(self):
        best_HGBC = Pipeline(steps=[('RobustScaler', RobustScaler()), 
                              ('HGBClassifier', HistGradientBoostingClassifier(learning_rate=0.1,max_depth=25,max_leaf_nodes=40,class_weight='balanced'))])
    
        x = self.df.drop('TARGET', axis=1)
        y = self.df['TARGET']
        model = best_HGBC.fit(x, y)
        return model

    # Fonction pour la prediction
    def predict_species(self, EXT_SOURCE_1,EXT_SOURCE_2, EXT_SOURCE_3, PAYMENT_RATE, DAYS_EMPLOYED):
        data_in = [[EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, PAYMENT_RATE, DAYS_EMPLOYED]]
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in).max()
        return prediction[0], probability