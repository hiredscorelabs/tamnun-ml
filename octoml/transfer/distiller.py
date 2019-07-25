from sklearn.base import TransformerMixin, BaseEstimator


class Distiller(BaseEstimator, TransformerMixin):
    """
    Distills one (usually big) models knowledge to another (usually much smaller) model. Inspired by this paper https://arxiv.org/abs/1503.02531
    """
    def __init__(self, teacher_model, teacher_predict_func, student_model):
        """
        teacher_model: The model to distill (should support sklearn fit-predict interface)
        teacher_predict_func: A function to calculates the teacher models predictions. For best results, use raw outputs (logits)
        student_model: The model to train  (should support sklearn fit-predict interface).
        """
        self.teacher_model = teacher_model
        self.teacher_predict_func = teacher_predict_func
        self.student_model = student_model

    def fit(self, X, y, unlabeled_X):
        """
        Fits the student models using the teacher model's predictions

        X: Labeled X variable, will be used to train the teacher model
        y: Target variables, will be used to train the teacher model
        unlabeled_X: Will be used to train the student model using the teacher model predictions in addition to
        return: self
        """
        self.teacher_model = self.teacher_model.fit(X, y)
        unlabeled_y = self.teacher_predict_func(unlabeled_X)
        self.student_model = self.student_model.fit(unlabeled_X, unlabeled_y)

        return self

    def transform(self, X):
        """
        Uses the student model to predict target variable
        X: The input for the predictions
        return: predictions of the student model
        """
        return self.student_model.predict(X)
