import json
from tqdm import tqdm
from karl.new_util import User
from karl.web import get_sessions


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
    n_users = session.query(User).count()
    for i, user in enumerate(tqdm(session.query(User), total=n_users)):
        leitner_box = {}  # fact_id -> box (1~10)
        count_correct_before = {}  # fact_id -> number of times answered correctly before
        count_wrong_before = {}  # fact_id -> number of times answered incorrectly before
        for j, record in enumerate(user.records):
            fact = record.fact
            leitner_update(leitner_box, fact, record.response)
            if fact.fact_id not in count_correct_before:
                count_correct_before[fact.fact_id] = 0
            if fact.fact_id not in count_wrong_before:
                count_wrong_before[fact.fact_id] = 0
            if record.response:
                count_correct_before[fact.fact_id] += 1
            else:
                count_wrong_before[fact.fact_id] += 1
            user_snapshot = {
                'leitner_box': leitner_box,
                'count_correct_before': count_correct_before,
                'count_wrong_before': count_wrong_before,
            }
            record.user_snapshot = json.dumps(user_snapshot)
        session.commit()


if __name__ == '__main__':
    session = get_sessions()['prod']
    update_user_snapshot(session)