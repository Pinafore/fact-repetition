from pydantic import BaseModel


class RetentionFeaturesSchema(BaseModel):
    user_count_correct: float
    user_count_wrong: float
    user_count_total: float
    user_average_overall_accuracy: float
    user_average_question_accuracy: float
    user_previous_result: float
    user_gap_from_previous: float
    question_average_overall_accuracy: float
    question_count_total: float
    question_count_correct: float
    question_count_wrong: float
