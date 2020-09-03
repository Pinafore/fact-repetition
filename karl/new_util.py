import io
import json
import pytz
import numpy as np
import sqlite3
from dataclasses import dataclass
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import MutableDict, MutableList
import sqlalchemy.types as types


Base = declarative_base()


@dataclass
class Params:
    qrep: float = 1                     # cosine distance between qreps
    skill: float = 0                    # fact difficulty vs user skill level
    recall: float = 1                   # recall probability
    recall_target: float = 1            # target of recall probability
    category: float = 1                 # change in category from prev
    answer: float = 1                   # reptition of the same category
    leitner: float = 0                  # hours till leitner scheduled date
    sm2: float = 1                      # hours till sm2 scheduled date
    decay_qrep: float = 0.9             # discount factor
    decay_skill: float = 0.9            # discount factor
    cool_down: float = 1                # weight for cool down
    cool_down_time_correct: float = 20  # minutes to cool down
    cool_down_time_wrong: float = 4     # minutes to cool down
    max_recent_facts: int = 10          # num of recent facts to keep record of


class JSONEncoded(types.TypeDecorator):

    impl = types.VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class BinaryNumpy(types.TypeDecorator):

    impl = types.BINARY

    def process_bind_param(self, value, dialect):
        out = io.BytesIO()
        np.save(out, value)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def process_result_value(self, value, dialect):
        out = io.BytesIO(value)
        out.seek(0)
        return np.load(out)


class Fact(Base):
    __tablename__ = 'facts'
    fact_id = Column(String, primary_key=True)
    text = Column(String)
    answer = Column(String)
    category = Column(String)
    deck_name = Column(String, default='')
    deck_id = Column(String, default='')
    qrep = Column(BinaryNumpy, default=np.array([]))
    skill = Column(BinaryNumpy, default=np.array([]))
    results = Column(MutableList.as_mutable(JSONEncoded), default=[])


class User(Base):
    __tablename__ = 'users'
    user_id = Column(String, primary_key=True)
    recent_facts = Column(MutableList.as_mutable(JSONEncoded), default=[])
    previous_study = Column(MutableDict.as_mutable(JSONEncoded), default={})
    leitner_box = Column(MutableDict.as_mutable(JSONEncoded), default={})
    leitner_scheduled_date = Column(MutableDict.as_mutable(JSONEncoded), default={})
    sm2_efactor = Column(MutableDict.as_mutable(JSONEncoded), default={})
    sm2_interval = Column(MutableDict.as_mutable(JSONEncoded), default={})
    sm2_repetition = Column(MutableDict.as_mutable(JSONEncoded), default={})
    sm2_scheduled_date = Column(MutableDict.as_mutable(JSONEncoded), default={})
    # for computing user average accuracy
    results = Column(MutableList.as_mutable(JSONEncoded), default=[])
    # qid -> number of times user and qid correctly
    count_correct_before = Column(MutableDict.as_mutable(JSONEncoded), default={})
    # qid -> number of times user and qid incorrectly
    count_wrong_before = Column(MutableDict.as_mutable(JSONEncoded), default={})
    params = Column(MutableDict.as_mutable(JSONEncoded), default={})


class Record(Base):
    __tablename__ = 'records'
    record_id = Column(String, primary_key=True)
    debug_id = Column(String)
    user_id = Column(String, ForeignKey('users.user_id'))
    fact_id = Column(String, ForeignKey('facts.fact_id'))
    deck_id = Column(String)
    response = Column(String)
    judgement = Column(String)
    user_snapshot = Column(String)
    scheduler_snapshot = Column(String)
    fact_ids = Column(String)
    scheduler_output = Column(String)
    elapsed_seconds_text = Column(Integer)
    elapsed_seconds_answer = Column(Integer)
    elapsed_milliseconds_text = Column(Integer)
    elapsed_milliseconds_answer = Column(Integer)
    is_new_fact = Column(Boolean)
    date = Column(DateTime)

    user = relationship("User", back_populates="records")
    fact = relationship("Fact", back_populates="records")


User.records = relationship("Record", order_by=Record.date, back_populates="user")
Fact.records = relationship("Record", order_by=Record.date, back_populates="fact")


if __name__ == '__main__':
    from karl.db import SchedulerDB

    filename = 'db.sqlite.prod'

    db = SchedulerDB(filename)

    engine = create_engine(f'sqlite:///{filename}.new.db', echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    for user in db.get_user():
        new_user = User(
            user_id=user.user_id,
            recent_facts=[fact.fact_id for fact in user.recent_facts],
            previous_study={
                k: (d.astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M:%S'), r)
                for k, (d, r) in user.previous_study.items()
            },
            leitner_box=user.leitner_box,
            leitner_scheduled_date={
                k: d.astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
                for k, d in user.leitner_scheduled_date.items()
            },
            sm2_efactor=user.sm2_efactor,
            sm2_interval=user.sm2_interval,
            sm2_repetition=user.sm2_repetition,
            sm2_scheduled_date={
                k: d.astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
                for k, d in user.sm2_scheduled_date.items()
            },
            results=user.results,
            count_correct_before=user.count_correct_before,
            count_wrong_before=user.count_wrong_before,
            params=user.params.__dict__,
        )
        session.add(new_user)
    session.commit()

    for fact in db.get_fact():
        new_fact = Fact(
            fact_id=fact.fact_id,
            text=fact.text,
            answer=fact.answer,
            category=fact.category,
            deck_name=fact.deck_name,
            deck_id=fact.deck_id,
            qrep=fact.qrep,
            skill=fact.skill,
            results=fact.results,
        )
        session.add(new_fact)
    session.commit()

    for record in db.get_history():
        new_record = Record(
            record_id=record.history_id,
            debug_id=record.debug_id,
            user_id=record.user_id,
            fact_id=record.fact_id,
            deck_id=record.deck_id,
            response=record.response,
            judgement=record.judgement,
            user_snapshot=record.user_snapshot,
            scheduler_snapshot=record.scheduler_snapshot,
            fact_ids=record.fact_ids,
            scheduler_output=record.scheduler_output,
            elapsed_seconds_text=record.elapsed_seconds_text,
            elapsed_seconds_answer=record.elapsed_seconds_answer,
            elapsed_milliseconds_text=record.elapsed_milliseconds_text,
            elapsed_milliseconds_answer=record.elapsed_milliseconds_answer,
            is_new_fact=record.is_new_fact,
            date=record.date,
        )
        session.add(new_record)
    session.commit()
    session.close()
