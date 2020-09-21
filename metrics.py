"""each metric returns a name, a description, and a scalar value"""
# think of the metrics from the perspectives of the scheduler
# if the scheduler recommended this fact and the response is X, what does it say about the scheduler?
# e.g. is it too aggressively showing difficult new facts? is it repeating easy old facts too much?
# for `learned` metric, everything is limited to the given datetime span
# it captures the number of previous unknown facts that was successfully learned using the system


import json
import bisect
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Dict
from plotnine import ggplot, aes, theme,\
    geom_point, geom_line,\
    element_text

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


class ratio_new_facts_shown(Metric):

    name = 'ratio_new_facts_shown'
    description = 'Ratio of new facts among all shown.'

    def __init__(self, **kwargs):
        self.n_fact_shown = n_facts_shown(**kwargs)
        self.n_new_facts_shown = n_new_facts_shown(**kwargs)
        self.value = 0
    
    def update(self, record):
        self.n_fact_shown.update(record)
        self.n_new_facts_shown.update(record)
        self.value = 0 if self.n_fact_shown.value == 0 else \
            (self.n_new_facts_shown.value / self.n_fact_shown.value)


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


class ratio_new_facts_correct(Metric):

    name = 'ratio_new_facts_correct'
    description = 'Ratio of new facts answered correctly among all new facts.'

    def __init__(self, **kwargs):
        self.n_new_facts_shown = n_new_facts_shown(**kwargs)
        self.n_new_facts_correct = n_new_facts_correct(**kwargs)
    
    def update(self, record):
        self.n_new_facts_shown.update(record)
        self.n_new_facts_correct.update(record)
        self.value = 0 if self.n_new_facts_shown.value == 0 else \
            (self.n_new_facts_correct.value / self.n_new_facts_shown.value)


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


class ratio_old_facts_correct(Metric):

    name = 'ratio_old_facts_correct'
    description = 'Ratio of old facts answered correctly among all old facts shown.'

    def __init__(self, **kwargs):
        self.n_old_facts_shown = n_old_facts_shown(**kwargs)
        self.n_old_facts_correct = n_old_facts_correct(**kwargs)
        self.value = 0
    
    def update(self, record):
        self.n_old_facts_shown.update(record)
        self.n_old_facts_correct.update(record)
        self.value = 0 if self.n_old_facts_shown.value == 0 else \
            (self.n_old_facts_correct.value / self.n_old_facts_shown.value)


class n_known_old_facts_shown(Metric):

    name = 'n_known_old_facts_shown'
    description = '''Number of already-known old facts shown. These are the
        facts that the user got correct the first try (potentially before the
        datetime span). Thess cards are probably too easy.'''

    def __init__(self, **kwargs):
        self.value = 0
        self.correct_on_first_try = kwargs.pop('correct_on_first_try')

    def update(self, record):
        self.value += not record.is_new_fact and \
            self.correct_on_first_try[record.user_id][record.fact_id]


class n_known_old_facts_correct(Metric):

    name = 'n_known_old_facts_correct'
    description = 'Number of already-known old facts answered correctly (which is expected). These are the facts that the user got correct the first try (potentially before the datetime span). Thess cards are probably too easy.'

    def __init__(self, **kwargs):
        self.value = 0
        self.correct_on_first_try = kwargs.pop('correct_on_first_try')

    def update(self, record):
        self.value += not record.is_new_fact and \
            self.correct_on_first_try[record.user_id][record.fact_id] and \
            record.response


class n_known_old_facts_wrong(Metric):

    name = 'n_known_old_facts_wrong'
    description = 'Number of already-known old facts answered incorrectly (which is unexpected). These are the facts that the user got correct the first try (potentially before the datetime span). This means the user might have got it correct by being lucky.'

    def __init__(self, **kwargs):
        self.value = 0
        self.correct_on_first_try = kwargs.pop('correct_on_first_try')

    def update(self, record):
        self.value += not record.is_new_fact and \
            self.correct_on_first_try[record.user_id][record.fact_id] and \
            not record.response

class ratio_known_old_facts_shown(Metric):

    name = 'ratio_known_old_facts_shown'
    description = 'Ratio of known facts shown among all old facts.'

    def __init__(self, **kwargs):
        self.n_old_facts_shown = n_old_facts_shown(**kwargs)
        self.n_known_old_facts_shown = n_known_old_facts_shown(**kwargs)
        self.value = 0

    def update(self, record):
        self.n_old_facts_shown.update(record)
        self.n_known_old_facts_shown.update(record)
        self.value = 0 if self.n_old_facts_shown.value == 0 else \
            (self.n_known_old_facts_shown.value / self.n_old_facts_shown.value)


class ratio_known_old_facts_correct(Metric):

    name = 'ratio_known_old_facts_correct'
    description = 'Ratio lf already-known old facts answered correctly among all known old facts shown.'

    def __init__(self, **kwargs):
        self.n_known_old_facts_shown = n_known_old_facts_shown(**kwargs)
        self.n_known_old_facts_correct = n_known_old_facts_correct(**kwargs)
        self.value = 0
    
    def update(self, record):
        self.n_known_old_facts_shown.update(record)
        self.n_known_old_facts_correct.update(record)
        self.value = 0 if self.n_known_old_facts_shown.value == 0 else \
            (self.n_known_old_facts_correct.value / self.n_known_old_facts_shown.value)


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


class ratio_learned(Metric):

    name = 'ratio_learned'
    description = 'Ratio of cards learned (see `n_learned`) among those not previously known.'

    def __init__(self, **kwargs):
        self.n_learned = n_learned(**kwargs)
        self.n_not_known = 0
        self.value = 0
    
    def update(self, record):
        self.n_learned.update(record)
        if record.is_new_fact and not record.response:
            self.n_not_known += 1
        self.value = 0 if self.n_not_known == 0 else (self.n_learned.value / self.n_not_known)


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


class ratio_learned_but_forgotten(Metric):

    name = 'ratio_learned_but_forgotten'
    description = 'Ratio of learned facts forgotten.'

    def __init__(self, **kwargs):
        self.n_learned = n_learned(**kwargs)
        self.n_learned_but_forgotten = n_learned_but_forgotten(**kwargs)
    
    def update(self, record):
        self.n_learned.update(record)
        self.n_learned_but_forgotten.update(record)
        self.value = 0 if self.n_learned.value == 0 else \
            (self.n_learned_but_forgotten.value / self.n_learned.value)
    

def get_metrics(
    session,
    date_start: datetime = None,
    date_end: datetime = None,
    date_stride: int = 1,
):
    metric_class_list = [
        n_facts_shown,
        ratio_new_facts_shown,
        ratio_new_facts_correct,
        n_new_facts_shown,
        n_new_facts_correct,
        n_new_facts_wrong,
        n_old_facts_shown,
        n_old_facts_correct,
        n_old_facts_wrong,
        ratio_old_facts_correct,
        n_known_old_facts_shown,
        n_known_old_facts_correct,
        n_known_old_facts_wrong,
        ratio_known_old_facts_shown,
        ratio_known_old_facts_correct,
        n_learned,
        n_learned_but_forgotten,
        ratio_learned,
        ratio_learned_but_forgotten,
    ]

    n_users = session.query(User).count()
    correct_on_first_try = {}  # user_id -> {fact_id -> bool}
    for user in tqdm(session.query(User), total=n_users):
        correct_on_first_try[user.user_id] = {}
        for record in user.records:
            if record.fact_id in correct_on_first_try[user.user_id]:
                continue
            correct_on_first_try[user.user_id][record.fact_id] = record.response

    if date_start is None:
        date_start = session.query(Record).first().date.date()
    if date_end is None:
        date_end = session.query(Record).order_by(Record.date.desc()).first().date.date()
    
    n_bins = (date_end - date_start).days // date_stride + 1
    end_dates = [date_start + i * timedelta(days=date_stride) for i in range(n_bins)]

    rows = []
    for user in tqdm(session.query(User), total=n_users):
        # TODO filter users by number of records within this period
        if len(user.records) < 20:
            continue
        repetition_model = infer_repetition_model(session, user=user, date_end=date_end)
        metrics = [metric_class(correct_on_first_try=correct_on_first_try) for metric_class in metric_class_list]
        curr_end_date = end_dates[bisect.bisect(end_dates, user.records[0].date.date())]
        for record in user.records:
            try:
                new_end_date = end_dates[bisect.bisect(end_dates, record.date.date())]
            except IndexError:
                new_end_date = end_dates[-1]
            if new_end_date != curr_end_date:
                # finished computing metrics for this date window
                for m in metrics:
                    rows.append({
                        'user_id': user.user_id,
                        'name': m.name,
                        'value': m.value,
                        'date_start': date_start,
                        'date_end': curr_end_date,
                        'repetition_model': repetition_model,
                    })
                curr_end_date = new_end_date
            for metric in metrics:
                metric.update(record)

    df = pd.DataFrame(rows)
    df.date_start = df.date_start.astype(np.datetime64)
    df.date_end = df.date_end.astype(np.datetime64)
    return df

def infer_repetition_model(session, user: User, record: Record = None, date_end: datetime = None) -> str:
    # find out the last repetition model that the user used before date_end
    # given a dictionary of params, infer what repetition model is used
    if record is None:
        records = session.query(Record).\
            filter(Record.user_id == user.user_id)
        if date_end is not None:
            records = records.filter(Record.date <= date_end)
        record = records.order_by(Record.date.desc()).first()
    if record is None:
        return None
    
    params = json.loads(record.scheduler_snapshot)
    if params['qrep'] == 0:
        if params['leitner'] > 0:
            return 'leitner'
        elif params['sm2'] > 0:
            return 'sm2'
        else:
            return 'unknown'
    else:
        if 'recall_target' in params:
            return 'karl-' + str(params['recall_target'] * 100)
        else:
            return 'karl-100'


if __name__ == '__main__':
    session = get_sessions()['prod']
    update_user_snapshot(session)
    df = get_metrics(session)
    # average for each user
    df = df.groupby(['user_id', 'name', 'date_start', 'date_end', 'repetition_model']).mean()
    # average for each repetition model
    df = df.groupby(['repetition_model', 'name', 'date_start', 'date_end']).mean().reset_index()
    for name in df.name.unique():
        # compare repetition models on each metric
        p = (
            ggplot(df[df.name == name])
            + aes(x='date_end', y='value', color='repetition_model')
            + geom_point()
            + geom_line()
            + theme(
                axis_text_x=element_text(rotation=30)
            )
        )
        p.save(f'output/{name}.pdf')

    for repetition_model in df.repetition_model.unique():
        # compare metrics for each repetition model
        p = (
            ggplot(df[df.repetition_model == repetition_model])
            + aes(x='date_end', y='value', color='name')
            + geom_point()
            + geom_line()
            + theme(
                axis_text_x=element_text(rotation=30)
            )
        )
        p.save(f'output/{repetition_model}.pdf')
