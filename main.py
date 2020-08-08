from parser.parser import Parser
from predictions.predictor import Predictor
from utils.constants import DATASET_FILE_PATH
from utils.constants import PREPARED_DATASET_FILE_PATH
from tests.test_predict import test

TEXT = 'Ромашки цветут летом'


if __name__ == '__main__':
    # Подготовка датасета для анализа
    # parser = Parser()
    # parser.write_to_file()

    # Анализ
    predictor = Predictor()
    predictor.train_data()
    res = predictor.get_sentiment_percentage(TEXT)
    print(
        f"PROB_BAD: {res['probability_bad']}\nPROB_NORMAL: {res['probability_normal']}\nBAD_WORDS: {res['bad_words']}"
    )

    # # Тесты
    test()
