import json
from tqdm import tqdm
from collections import defaultdict
from karl.models import User, Record, Fact
from karl.util import Params, get_sessions


def leitner_update(leitner_box, fact, response: bool) -> None:
    """
    Update Leitner box and scheduled date of card.

    :param user:
    :param fact:
    :param response: CORRECT or WRONG.
    """
    # leitner boxes 1~10
    # days[0] = None as placeholder since we don't have box 0
    # days[9] and days[10] = 9999 to make it never repeat
    cur_box = leitner_box.get(fact.fact_id, None)
    if cur_box is None:
        cur_box = 1
    new_box = cur_box + (1 if response else -1)
    new_box = max(min(new_box, 10), 1)
    leitner_box[fact.fact_id] = new_box


def update_user_snapshot(session):
    '''Fill in missing fields of user snapshots'''
    leitner_box = {}  # user_id -> fact_id -> box (1~10)
    count_correct_before = {}  # user_id -> fact_id -> number of times answered correctly before
    count_wrong_before = {}  # user_id -> fact_id -> number of times answered incorrectly before
    previous_study = {}  # user_id -> fact_id -> (str(date), response)
    user_results = {}  # user_id -> list of True/False
    fact_results = {}  # fact_id -> list of True/False
    records = session.query(Record).order_by(Record.date)
    for record in tqdm(records, total=records.count()):
        if record.user_id not in leitner_box:
            leitner_box[record.user_id] = {}
        if record.user_id not in count_correct_before:
            count_correct_before[record.user_id] = {}
        if record.user_id not in count_wrong_before:
            count_wrong_before[record.user_id] = {}
        if record.user_id not in previous_study:
            previous_study[record.user_id] = {}
        if record.user_id not in user_results:
            user_results[record.user_id] = []
        if record.fact_id not in fact_results:
            fact_results[record.fact_id] = []

        record.user_snapshot = json.dumps({
            'leitner_box': leitner_box[record.user_id],
            'count_correct_before': count_correct_before[record.user_id],
            'count_wrong_before': count_wrong_before[record.user_id],
            'previous_study': previous_study[record.user_id],
            'user_results': user_results[record.user_id],
            'fact_results': fact_results[record.fact_id],
        })

        leitner_update(leitner_box[record.user_id], record.fact, record.response)

        if record.fact_id not in count_correct_before[record.user_id]:
            count_correct_before[record.user_id][record.fact_id] = 0
        if record.fact_id not in count_wrong_before[record.user_id]:
            count_wrong_before[record.user_id][record.fact_id] = 0
        if record.response:
            count_correct_before[record.user_id][record.fact_id] += 1
        else:
            count_wrong_before[record.user_id][record.fact_id] += 1

        previous_study[record.user_id][record.fact_id] = (str(record.date), record.response)
        user_results[record.user_id].append(record.response)
        fact_results[record.fact_id].append(record.response)

    session.commit()


def infer_repetition_model(params) -> str:
    if params['qrep'] == 0:
        if params['leitner'] > 0:
            return 'leitner'
        elif params['sm2'] > 0:
            return 'sm2'
        else:
            return 'unknown'
    else:
        if 'recall_target' in params:
            return 'karl' + str(int(params['recall_target'] * 100))
        else:
            return 'karl100'


def update_scheduler_snapshot(session):
    '''Fill in missing retention model name in the scheduler snapshots in each record'''
    records = session.query(Record)
    for record in tqdm(records, total=records.count()):
        params = json.loads(record.scheduler_snapshot)
        # if 'repetition_model' in params:
        #     continue
        params['repetition_model'] = infer_repetition_model(params)
        record.scheduler_snapshot = json.dumps(params)
        record.elapsed_seconds_text = record.elapsed_milliseconds_text / 1000
        record.elapsed_seconds_answer = record.elapsed_milliseconds_answer / 1000
    session.commit()

    for user in session.query(User):
        new_params = Params(
            repetition_model=infer_repetition_model(user.params.__dict__),
            qrep=user.params.qrep,
            skill=user.params.skill,
            recall=user.params.recall,
            recall_target=user.params.recall_target,
            category=user.params.category,
            answer=user.params.answer,
            leitner=user.params.leitner,
            sm2=user.params.sm2,
            decay_qrep=user.params.decay_qrep,
            decay_skill=user.params.decay_skill,
            cool_down=user.params.cool_down,
            cool_down_time_correct=user.params.cool_down_time_correct,
            cool_down_time_wrong=user.params.cool_down_time_wrong,
            max_recent_facts=user.params.max_recent_facts,
        )
        user.params = new_params
    session.commit()


def update_fact_results(session):
    for fact in session.query(Fact):
        if fact.results is None:
            fact.results = []
    session.commit()


if __name__ == '__main__':
    session = get_sessions()['prod']
    update_fact_results(session)