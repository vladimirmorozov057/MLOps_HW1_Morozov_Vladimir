import numpy as np
from flask import Flask, jsonify
from flask_restx import Api, Resource
from sklearn.preprocessing import StandardScaler
from flask_restful import reqparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import json
import os
import pickle
import ast

#Гружу модель
def load_model(params, name):

    # Загружаю параметры модели
    model_params = json.loads(json.dumps(ast.literal_eval(params)))

    print(name)
    print(model_params)

    if name == 'Gradboost':
        return GradientBoostingClassifier(**model_params)
    elif name == 'Logreg':
        return LogisticRegression(**model_params)
    #else:
        #api.abort(403, message="Модель не определена")

# Чистим датасет вилкой
def clean_data(df, for_train=True):

    df.dropna()
    np.random.seed(910)
    mask_plus = np.random.choice(np.where(df.target == 1)[0], 100000, replace=True)
    mask_zero = np.random.choice(np.where(df.target == 0)[0], 100000, replace=True)
    df = pd.concat((df.iloc[mask_plus], df.iloc[mask_zero]))
    df_target = pd.DataFrame(df['target'])
    df_target.reset_index(drop=True, inplace=True)
    df_arguments = df.drop(['target'], axis=1)
    scaler = StandardScaler()
    df_arguments = pd.DataFrame(scaler.fit_transform(df_arguments))
    df_res = pd.concat([df_target, df_arguments], axis=1)

    if for_train:
        return df_res
    else:
        return df_arguments

# Тут проходит обучение
def train_model(args):

    print(args.hyperparameters)

    base_model = load_model(args.hyperparameters, args.model_type)
    if base_model == 0:
        return 403
    data = pd.read_csv(args.csv_for_clean_and_load)
    data = clean_data(data)
    model_arguments = data.drop(['target'], axis=1)
    target_value = data['target']
    base_model.fit(model_arguments, target_value)

    return base_model

# Предсказания на тест-выборке
def make_predictions(args):

    data = pd.read_csv(args.csv_for_clean_and_load)
    data_cleaned = clean_data(data, for_train=False)

    if model_type == 403:
        return 'Модели с таким экспериментом нет!', 403
    # Предсказания
    preds = model_type.predict(data_cleaned)
    return {
               'Предсказания модели': preds.tolist()
           }, 200






    model = pickle.load(open(
        'train_res/' + args.model_type + "_" + str(
            args.experiment_id) + '.pkl',
        'rb'))

    if os.path.isfile(train_res_save):
        pickle.dump(models, open(train_res_save, 'wb'))
        return 'Модель обучена, но такой эксперимент есть', 202

    else:
        pickle.dump(models, open(train_res_save, 'wb'))
        return 'Обучение прошло успешно', 200

