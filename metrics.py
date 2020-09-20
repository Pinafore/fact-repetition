"""each metric returns a name, a description, and a scalar value"""
# think of the metrics from the perspectives of the scheduler
# if the scheduler recommended this fact and the response is X, what does it say about the scheduler?
# e.g. is it too aggressively showing difficult new facts? is it repeating easy old facts too much?
# for `learned` metric, everything is limited to the given datetime span
# it captures the number of previous unknown facts that was successfully learned using the system


import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from plotnine import *

from karl.new_util import User, Record, parse_date
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
    debug = False
    n_users = session.query(User).count()
    for i, user in enumerate(tqdm(session.query(User), total=n_users)):
        if debug and i > 10:
            break
        leitner_box = {}  # fact_id -> box (1~10)
        count_correct_before = {}  # fact_id -> number of times answered correctly before
        count_wrong_before = {}  # fact_id -> number of times answered incorrectly before
        for j, record in enumerate(user.records):
            if debug and j > 100:
                break
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


class Metric:
    name = None
    description = None
    value = 0

    def __init__(self, **kwargs):
        pass

    def update(self, record):
        pass


class n_facts_shown(Metric):

    name = 'n_facts_shown'
    description = 'Number of facts shown (including repetitions).'

    def __init__(self, **kwargs):
        self.value = 0

    def update(self, record):
        self.value += 1


class n_new_facts_shown(Metric):

    name = 'n_new_facts_shown'
    description = 'Number of new facts shown.'

    def __init__(self, **kwargs):
        self.value = 0

    def update(self, record):
        self.value += record.is_new_fact


class n_new_facts_correct(Metric):

    name = 'n_new_facts_correct'
    description = 'Number of new facts answered correctly.'

    def __init__(self, **kwargs):
        self.value = 0

    def update(self, record):
        self.value += (record.is_new_fact and record.response)


class n_new_facts_wrong(Metric):

    name = 'n_new_facts_wrong'
    description = 'Number of new facts answered incorrectly.'

    def __init__(self, **kwargs):
        self.value = 0

    def update(self, record):
        self.value += (record.is_new_fact and not record.response)


class n_old_facts_shown(Metric):

    name = 'n_old_facts_shown'
    description = 'Number of old facts reviewed.'

    def __init__(self, **kwargs):
        self.value = 0

    def update(self, record):
        self.value += not record.is_new_fact


class n_old_facts_correct(Metric):

    name = 'n_old_facts_correct'
    description = 'Number of old facts answered correctly.'

    def __init__(self, **kwargs):
        self.value = 0

    def update(self, record):
        self.value += (not record.is_new_fact and record.response)


class n_old_facts_wrong(Metric):

    name = 'n_old_facts_wrong'
    description = 'Number of old facts answered incorrectly.'

    def __init__(self, **kwargs):
        self.value = 0

    def update(self, record):
        self.value += (not record.is_new_fact and not record.response)


class n_known_facts_shown(Metric):

    name = 'n_known_facts_shown'
    description = '''Number of already-known old facts shown. These are the
        facts that the user got correct the first try (potentially before the
        datetime span). Thess cards are probably too easy.'''

    def __init__(self, **kwargs):
        self.value = 0
        self.correct_on_first_try = kwargs.pop('correct_on_first_try')

    def update(self, record):
        self.value += not record.is_new_fact and \
            self.correct_on_first_try[record.user_id][record.fact_id]


class n_known_facts_correct(Metric):

    name = 'n_known_facts_correct'
    description = 'Number of already-known old facts answered correctly (which is expected). These are the facts that the user got correct the first try (potentially before the datetime span). Thess cards are probably too easy.'

    def __init__(self, **kwargs):
        self.value = 0
        self.correct_on_first_try = kwargs.pop('correct_on_first_try')

    def update(self, record):
        self.value += not record.is_new_fact and \
            self.correct_on_first_try[record.user_id][record.fact_id] and \
            record.response


class n_known_facts_wrong(Metric):

    name = 'n_known_facts_wrong'
    description = 'Number of already-known old facts answered incorrectly (which is unexpected). These are the facts that the user got correct the first try (potentially before the datetime span). This means the user might have got it correct by being lucky.'

    def __init__(self, **kwargs):
        self.value = 0
        self.correct_on_first_try = kwargs.pop('correct_on_first_try')

    def update(self, record):
        self.value += not record.is_new_fact and \
            self.correct_on_first_try[record.user_id][record.fact_id] and \
            not record.response


class n_learned(Metric):

    name = 'n_learned'
    description = 'Number of not known facts that the user saw for the first time, but correctly answered multiple times afterwards. Specifically, we consider facts that the user got the correct answer twice consecutively.'

    def __init__(self, **kwargs):
        self.counter = {}
        self.value = 0

    def update(self, record):
        if record.is_new_fact and not record.response:
            # only consider facts not known before
            self.counter[record.fact_id] = 0
        if record.fact_id not in self.counter:
            return

        if record.response:
            self.counter[record.fact_id] += 1
            if self.counter[record.fact_id] == 2:
                self.value += 1
        else:
            self.counter[record.fact_id] = 0


class n_learned_but_forgotten(Metric):

    name = 'n_learned_but_fogotten'
    description = 'Number of learned cards answered incorrectly afterwards. See `n_learned` for definition of what is a learned card.'

    def __init__(self, **kwargs):
        self.counter = {}
        self.value = 0

    def update(self, record):
        if record.is_new_fact and not record.response:
            # only consider facts not known before
            self.counter[record.fact_id] = 0
        if record.fact_id not in self.counter:
            return

        if record.response:
            self.counter[record.fact_id] += 1
        else:
            if self.counter[record.fact_id] >= 2:
                self.value += 1
            self.counter[record.fact_id] = 0



def get_metrics(
    session,
    metric_class_list,
    user_id: str = None,
    deck_id: str = None,
    date_start: str = None,
    date_end: str = None
):
    n_users = session.query(User).count()
    correct_on_first_try = {}  # user_id -> {fact_id -> bool}
    for user in tqdm(session.query(User), total=n_users):
        correct_on_first_try[user.user_id] = {}
        for record in user.records:
            if record.fact_id in correct_on_first_try[user.user_id]:
                continue
            correct_on_first_try[user.user_id][record.fact_id] = record.response

    if date_start is None:
        date_start = '2008-06-11 08:00:00'
    if date_end is None:
        date_end = '2038-06-11 08:00:00'
    date_start = parse_date(date_start)
    date_end = parse_date(date_end)

    if user_id is not None:
        users = [session.query(User).filter(User.user_id == user_id).first()]
    else:
        users = session.query(User)

    metrics_by_user = {}
    for user in users:
        records = session.query(Record).\
            filter(Record.date >= date_start, Record.date <= date_end).\
            filter(Record.user_id == user.user_id)
        if deck_id is not None:
            records = records.filter(Record.deck_id == deck_id)

        metrics = [metric_class(correct_on_first_try=correct_on_first_try) for metric_class in metric_class_list]
        for record in records:
            for metric in metrics:
                metric.update(record)

        metrics_by_user[user.user_id] = {m.name: m.value for m in metrics}
    return metrics_by_user


if __name__ == '__main__':
    session = get_sessions()['prod']
    # update_user_snapshot(session)

    metric_class_list = [
        n_facts_shown,
        n_new_facts_shown,
        n_new_facts_correct,
        n_new_facts_wrong,
        n_new_facts_wrong,
        n_old_facts_shown,
        n_old_facts_correct,
        n_old_facts_wrong,
        n_known_facts_shown,
        n_known_facts_correct,
        n_known_facts_wrong,
        n_learned,
        n_learned_but_forgotten,
    ]

    date_stride = 3
    date_start = session.query(Record).first().date.date()
    date_end = session.query(Record).order_by(Record.date.desc()).first().date.date()
    rows = []
    for i in range((date_end - date_start).days // date_stride):
        window_start = date_start
        window_end = window_start + timedelta(days=(i + 1) * date_stride)
        metrics_by_user = get_metrics(session, metric_class_list,
                                      date_start=str(window_start), date_end=str(window_end))
        metrics_averaged = {}
        for user, metrics in metrics_by_user.items():
            for name, value in metrics.items():
                if name not in metrics_averaged:
                    metrics_averaged[name] = []
                metrics_averaged[name].append(value)
        metrics_averaged = {k: np.mean(v) for k, v in metrics_averaged.items()}
        for k, v in metrics_averaged.items():
            rows.append({
                'index': i,
                'name': k,
                'value': v,
                'date_start': window_start,
                'date_end': window_end,
            })
        print(window_start, window_end)
    
    df = pd.DataFrame(rows)
    # otherwise it won't be considered as ordered
    df.date_start = df.date_end.astype(np.datetime64)
    df.date_end = df.date_end.astype(np.datetime64)

    p = (
        ggplot(df)
        + aes(x='date_end', y='value', color='name')
        + geom_point()
        + geom_line()
        + theme(
            axis_text_x=element_text(rotation=30)
        )
    )
    p.save('output/metrics.pdf')