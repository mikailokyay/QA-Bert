"""
This is Question Answering main file
"""
import numpy as np
from src.train import QATrainer
from src.evaluation import evaluate


class QuestionAnswerer(QATrainer):
    """
    This is Question Answering class
    """
    def __init__(self, model_n):
        super().__init__(model_n)

    def get_predictions(self):
        """
        This is function for getting predictions and gold answers of test data
        :return: predictions and gold answers
        """
        predictions = []
        gold_answers = []
        for _, row in self.test_df.iterrows():
            text = row["context"]
            question = row["question"]
            answer_text = row["answer_text"]
            prediction = self.predict_answer(text, question)
            predictions.append(prediction)
            gold_answers.append(answer_text)
        return predictions, gold_answers

    def predict_answer(self, text, question):
        """
        This is prediction function
        :param text: context for answering question
        :param question: question to answer
        :return:
        """
        inputs = self.tokenizer.encode_plus(question, text,
                                            return_tensors='pt',
                                            max_length=512,
                                            truncation=True).to(self.device)

        outputs = self.model(**inputs)
        answer_start = np.argmax(outputs[0])
        answer_end = np.argmax(outputs[1]) + 1

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        return answer


if __name__ == "__main__":

    TRAIN_MODEL = True
    params = {
        "model_type": 'bert',
        "model_name": 'bert-base-cased',
        "batch_size": 16,
        "learning_rate": 1e-5,
        "epochs": 2,
        "save_only_last_epoch": False,
        "output_dir": '../outputs',

    }

    if TRAIN_MODEL:
        MODEL_NAME = params["model_name"]
    else:
        MODEL_NAME = f"{params['output_dir']}/bert-epoch-0-train-loss-1.5018074199706997-val-loss-1.0954303917067856"

    qa = QuestionAnswerer(MODEL_NAME)

    if TRAIN_MODEL:
        qa.train(params)

    predicted_answers, actual_answers = qa.get_predictions()
    print("evaluation results:\n", evaluate(predicted_answers, actual_answers))

    CONTEXT = """
    ???? Transformers is backed by the three most popular deep learning libraries ??? Jax, PyTorch and TensorFlow ??? with a seamless integration
    between them. It's straightforward to train your models with one before loading them for inference with the other.
    """
    QUESTION = "Which deep learning libraries back ???? Transformers?"
    print(qa.predict_answer(CONTEXT, QUESTION))
