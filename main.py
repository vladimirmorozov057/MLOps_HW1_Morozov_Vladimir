import numpy as np
from flask import Flask, jsonify
from flask_restx import Api, Resource
from sklearn.preprocessing import StandardScaler
from flask_restful import reqparse
from werkzeug.datastructures import FileStorage
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from werkzeug.datastructures import FileStorage
import json
import os
import pickle

# Так, сейчас я для себя буду писать комментарии, что за что отвечает
# Вот эта штука просто инициирует среду (ну или пространство на сервере, хз как сказать) модели
app = Flask(__name__)
api = Api(app)

# Вот эта штука инициирует возможность добавлений в эту среду
upload_parser = api.parser()

# Тут во сути словарик с моделями. Их будет две

models = {

    "Gradboost": {"file_with_params": "models_params/Gradboost_params.json"},

    "Logreg": {"file_with_params": "models_params/Logreg_params.json"}

}

# Это то, что именно добавляется, как это называется, какой у этого тип и обязательность
# А, ну и какая ошибка выдается, если что-то не так

upload_parser.add_argument('csv_for_clean_and_load', location='files',
                           type=FileStorage, required=True,
                           help='Заливай тренировочный датасет')

upload_parser.add_argument('model_type',
                           required=True,
                           location='args',
                           choices=list(models),
                           help='Выбирай тип модели')

upload_parser.add_argument('experiment_id',
                           type=str, default="0",
                           help='Выберай номер эксперимента')

upload_parser.add_argument('hyperparameters', required=False,
                           help='Выбирай гиперпараметры, если нужно'
                           )

# Чистим датасет вилкой

def clean_data(df, for_train=True):

    df.dropna()
    np.random.seed(910)
    mask_plus = np.random.choice(np.where(df.target == 1)[0], 100000, replace=True)
    mask_zero = np.random.choice(np.where(df.target == 0)[0], 100000, replace=True)
    df = pd.concat((df.iloc[mask_plus], df.iloc[mask_zero]))
    df_target = pd.DataFrame(df['target'])
    df_target.reset_index(drop=True, inplace=True)
    df_x = df.drop(['target'], axis=1)
    scaler = StandardScaler()
    df_x = pd.DataFrame(scaler.fit_transform(df_x))
    df_res = pd.concat([df_target, df_x], axis=1)

    if for_train:
        return df_res
    else:
        return df_x

@api.route('/train', methods=['PUT'], doc={'description': 'Запустить обучение выбранной модели на датасете'})
@api.expect(upload_parser)

class Train(Resource):

    @api.doc(params={'csv_for_clean_and_load': f'Лей файл CSV'})
    @api.doc(params={'experiment_id': f'Пиши номер эксперимента'})
    @api.doc(params={'hyperparameters': f'Впиши путь к файлу с параметрами, не учитывая корень'})

    @api.doc(responses={200: 'О, норм!'})
    @api.doc(responses={202: 'Обучение прошло, но эксперимент дублируется'})
    @api.doc(responses={403: 'Нет такого эксперимента'})

    def put(self):

        args = upload_parser.parse_args()
        base_model = self.load_model(args.hyperparameters)
        data = pd.read_csv(args.csv_for_clean_and_load)
        data = clean_data(data)
        X = data.drop(['target'], axis=1)
        y = data['target']
        models.fit(X, y)
        train_res_save = "train_res/" + args.hyperparameters[
                                           args.hyperparameters.find(
                                               '/') + 1:args.hyperparameters.rfind(
                                               '_')] + "_" + args.experiment_id + ".pkl"
        os.makedirs(
            os.path.dirname(train_res_save),
            exist_ok=True)

        # Смотрим, проводилось ли ранее обучение

        if os.path.isfile(train_res_save):
            pickle.dump(models,
                        open(train_res_save, 'wb'))
            return 'Модель обучена, но такой эксперимент есть', 202

        else:
            pickle.dump(models,
                        open(train_res_save, 'wb'))
            return 'Обучение прошло успешно', 200

    @staticmethod
    def load_model(model_path):

        # Загружаю параметры модели
        full_path = os.getcwd() + '/' + model_path

        with open(full_path, 'r') as JSON:
            model_params = json.load(JSON)
        if 'Gradboost' in model_path:
            return GradientBoostingClassifier(**model_params)
        elif 'Logreg' in model_path:
            return LogisticRegression(**model_params)
        else:
            api.abort(403, message="Модель не определена")


@api.route('/models', methods=['GET', 'DELETE'])
@api.expect(upload_parser)

class GetModels(Resource):

    def get(self):
        return jsonify(models)

    @api.doc(params={'experiment_id': f'Номер эксперимента для удаления'})
    @api.doc(params={'model_type': f'Название модели'})

    def delete(self):

        args = upload_parser.parse_args()
        delete_filename = "trained_weights/" + args.model_type + "_" + args.experiment_id + ".pkl"

        try:
            os.remove(delete_filename)
        except FileNotFoundError:
            return 'Такого файла нет', 403
        return 'Модель удалена', 200

@api.route('/predict', methods=['POST'])
@api.expect(upload_parser)

class Predict(Resource):

    @api.doc(params={'csv_for_clean_and_load': f'Файл CSV'})
    @api.doc(params={'experiment_id': f'Номер эксперимента'})
    @api.doc(params={'model_type': f'Название модели'})

    def post(self):

        args = upload_parser.parse_args()

        data = pd.read_csv(args.csv_for_clean_and_load)
        X = clean_data(data, for_train=False)

        model = pickle.load(open(
            'train_res/' + args.model_type + "_" + str(
                args.experiment_id) + '.pkl',
            'rb'))

        preds = model.predict(X)

        return {
                'Предсказания модели': preds.tolist()
               }, 200

if __name__ == '__main__':
    app.run(debug=True)

