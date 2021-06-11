# %%
import pytz
import numpy as np
import pandas as pd
import altair as alt
from tqdm import tqdm
from string import Template
from dateutil.parser import parse as parse_date
from sqlalchemy import Table, MetaData
from sqlalchemy import Column, ForeignKey, Integer, Float, Boolean, String, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB

from karl.scheduler import KARLScheduler
from karl.models import User, Card, \
    SimUserCardFeatureVector, SimUserFeatureVector, SimCardFeatureVector

from karl.db.session import engine, SessionLocal
from karl.models import Record
from karl.figures import save_chart_and_pdf

alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')

# %%

class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def create_simulated_tables():
    meta = MetaData()

    Table(
        'simusercardfeaturevector', meta,
        Column('user_id', String, ForeignKey(User.id), primary_key=True, index=True),
        Column('card_id', String, ForeignKey(Card.id), primary_key=True, index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
        Column('correct_on_first_try', Boolean),
        Column('leitner_box', Integer),
        Column('leitner_scheduled_date', TIMESTAMP(timezone=True)),
        Column('sm2_efactor', Float),
        Column('sm2_interval', Float),
        Column('sm2_repetition', Integer),
        Column('sm2_scheduled_date', TIMESTAMP(timezone=True)),
    )

    Table(
        'simuserfeaturevector', meta,
        Column('user_id', String, ForeignKey(User.id), primary_key=True, index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
        Column('parameters', JSONB),
    )

    Table(
        'simcardfeaturevector', meta,
        Column('card_id', String, ForeignKey(Card.id), primary_key=True, index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
    )

    meta.create_all(engine)


def fret_board_figure(
    user_id: str,
    date_start: str = '2020-12-20',
):
    date_start = parse_date(date_start).astimezone(pytz.utc).date()

    scheduler = KARLScheduler()
    # create_simulated_tables()
    with SessionLocal() as session:
        # remove existing simulated users
        session.query(SimUserFeatureVector).delete()
        # create new simulated user
        v_user = scheduler.get_latest_user_vector(user_id, date_start).__dict__
        v_user.pop('date')
        sim_v_user = SimUserFeatureVector(**v_user)
        session.add(sim_v_user)
        # remove existing simulated cards
        session.query(SimUserCardFeatureVector).delete()
        session.query(SimCardFeatureVector).delete()
        session.commit()
        # get repetition model
        repetition_model = scheduler.get_user(user_id, session).parameters.repetition_model

    rows = []
    cards = []
    card_id_to_x_axis = {}
    index = 0
    score_types = []  # ['sum', 'leitner']
    session = SessionLocal()
    records = session.query(Record).\
        filter(Record.user_id == user_id).\
        filter(Record.date >= date_start)
    # for record in tqdm(records, total=records.count()):
    for record in records:
        if record.response is None:
            continue

        index += 1
        # if index > 20:
        #     break

        if record.card_id not in card_id_to_x_axis:
            card_id_to_x_axis[record.card_id] = len(card_id_to_x_axis)
            cards.append(record.card)
            # create new simulated card
            v_card = scheduler.get_latest_card_vector(record.card.id, record.date).__dict__
            v_card.pop('date')
            sim_v_card = SimCardFeatureVector(**v_card)
            # create new simulated user-card
            v_usercard = scheduler.get_latest_usercard_vector(user_id, record.card.id, record.date).__dict__
            v_usercard.pop('date')
            sim_v_usercard = SimUserCardFeatureVector(**v_usercard)
            session.add(sim_v_card)
            session.add(sim_v_usercard)
            session.commit()

        scores, _ = scheduler.score_user_cards(record.user, cards, record.date, session, simulated=True)
        v_usercard = session.query(SimUserCardFeatureVector).get((user_id, record.card.id))
        if v_usercard.previous_study_date is None:
            delta = None
            previous_study_date = None
            previous_study_response = None
        else:
            delta = strfdelta(record.date - v_usercard.previous_study_date, "%D days %H:%M:%S")
            previous_study_date = v_usercard.previous_study_date.strftime('%b:%d %H:%M:%S')
            previous_study_response = v_usercard.previous_study_response

        shown_card_scores = None
        for i, _ in enumerate(scores):
            scores[i]['sum'] = sum([
                record.user.parameters.__dict__.get(key, 0) * value
                for key, value in scores[i].items()
            ])
            if cards[i].id == record.card.id:
                shown_card_scores = scores[i]

        rows.append({
            'x_axis': card_id_to_x_axis[record.card_id],
            'y_axis': index,
            'type': record.response,
            'delta': delta,
            'date': record.date.strftime('%b:%d %H:%M:%S'),
            'previous_study_date': previous_study_date,
            'previous_study_response': previous_study_response,
            'cool_down': shown_card_scores['cool_down'],
        })

        for score_type in score_types:
            order = np.argsort([s[score_type] for s in scores]).tolist()
            selected_card_id = cards[order[0]].id
            assert selected_card_id in card_id_to_x_axis

            v_usercard = session.query(SimUserCardFeatureVector).get((user_id, selected_card_id))
            if v_usercard.previous_study_date is None:
                delta = None
                previous_study_date = None
                previous_study_response = None
            else:
                delta = strfdelta(record.date - v_usercard.previous_study_date, "%D days %H:%M:%S")
                previous_study_date = v_usercard.previous_study_date.strftime('%b:%d %H:%M:%S')
                previous_study_response = v_usercard.previous_study_response

            rows.append({
                'x_axis': card_id_to_x_axis[selected_card_id],
                'y_axis': index,
                'type': score_type,
                'delta': delta,
                'date': record.date.strftime('%b:%d %H:%M:%S'),
                'previous_study_date': previous_study_date,
                'previous_study_response': previous_study_response,
                'cool_down': scores[order[0]]['cool_down'],
            })

        scheduler.update_feature_vectors(record, record.date, session, simulated=True)
        session.commit()

    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_circle(size=60).encode(
        alt.X('x_axis:Q', axis=alt.Axis(title='Cards')),
        # y='utc_datetime:T',
        alt.Y('y_axis:Q', axis=alt.Axis(title='Index')),
        color=alt.Color(
            'type',
            scale=alt.Scale(
                domain=[True, False] + score_types,
                range=['#6e8eb6', '#e67375', '#f5984b', '#8bc2bc'],
            )
        ),
        tooltip=['delta', 'type', 'date', 'previous_study_date', 'previous_study_response', 'cool_down']
    ).properties(
        width=800,
        height=2000,
        title=f'User {user_id} with model {repetition_model}'
    )
    save_chart_and_pdf(chart, '/fs/www-users/shifeng/files/fret')


# %%
if __name__ == '__main__':
    '''
    date_start = '2020-12-20'
    date_start = parse_date(date_start).astimezone(pytz.utc).date()

    user_count = {}
    session = SessionLocal()
    for record in session.query(Record).filter(Record.date >= date_start).filter(Record.response is not None):
        if record.user_id not in user_count:
            user_count[record.user_id] = 0
        user_count[record.user_id] += 1
    for k, v in sorted(user_count.items(), key=lambda x: -x[1])[:10]:
        print(k, v)
    session.commit()
    session.close()
    '''

    fret_board_figure(
        user_id='548',
        date_start='2020-12-20',
    )
