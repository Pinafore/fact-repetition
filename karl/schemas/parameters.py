from pydantic import BaseModel


class ParametersSchema(BaseModel):
    repetition_model: str = 'karl85'    # name of the repetition model
    card_embedding: float = 1           # weight on cosine distance between embeddings
    recall: float = 1                   # weight on recall probability
    recall_target: float = 0.85         # target of recall probability
    category: float = 1                 # change in category from prev
    answer: float = 1                   # reptition of the same answer
    leitner: float = 0                  # hours till leitner scheduled date
    sm2: float = 0                      # hours till sm2 scheduled date
    decay_qrep: float = 0.9             # discount factor
    cool_down: float = 1                # weight for cool down
    cool_down_time_correct: float = 20  # minutes to cool down
    cool_down_time_wrong: float = 4     # minutes to cool down
    max_recent_facts: int = 10          # num of recent facts to keep record of
