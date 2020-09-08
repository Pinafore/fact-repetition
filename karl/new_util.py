import json
import numpy as np
import msgpack
import msgpack_numpy
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Date
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


class ParamsType(types.TypeDecorator):

    impl = types.VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps({k: v for k, v in value.__dict__.items() if not k.startswith('_')})
        else:
            return json.dumps({k: v for k, v in Params().__dict__.items() if not k.startswith('_')})

    def process_result_value(self, value, dialect):
        return Params(**json.loads(value))


class BinaryNumpy(types.TypeDecorator):

    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        return msgpack.packb(value, default=msgpack_numpy.encode)

    def process_result_value(self, value, dialect):
        return msgpack.unpackb(value, object_hook=msgpack_numpy.decode)


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
    params = Column(ParamsType)


class Record(Base):
    __tablename__ = 'records'
    record_id = Column(String, primary_key=True)
    debug_id = Column(String)
    user_id = Column(String, ForeignKey('users.user_id'))
    fact_id = Column(String, ForeignKey('facts.fact_id'))
    deck_id = Column(String)
    response = Column(Boolean)
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


class UserStat(Base):
    __tablename__ = 'user_stats'
    user_stat_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.user_id'))
    deck_id = Column(String)
    date = Column(Date)
    new_facts = Column(Integer, default=0)
    reviewed_facts = Column(Integer, default=0)
    new_correct = Column(Integer, default=0)
    reviewed_correct = Column(Integer, default=0)
    total_seen = Column(Integer, default=0)
    total_milliseconds = Column(Integer, default=0)
    total_seconds = Column(Integer, default=0)
    total_minutes = Column(Integer, default=0)
    elapsed_milliseconds_text = Column(Integer, default=0)
    elapsed_milliseconds_answer = Column(Integer, default=0)
    elapsed_seconds_text = Column(Integer, default=0)
    elapsed_seconds_answer = Column(Integer, default=0)
    elapsed_minutes_text = Column(Integer, default=0)
    elapsed_minutes_answer = Column(Integer, default=0)
    # known_rate = Column(Float)
    # new_known_rate = Column(Float)
    # review_known_rate = Column(Float)

    user = relationship("User", back_populates="user_stats")


User.records = relationship("Record", order_by=Record.date, back_populates="user")
Fact.records = relationship("Record", order_by=Record.date, back_populates="fact")
User.user_stats = relationship("UserStat", order_by=UserStat.date, back_populates="user")


if __name__ == '__main__':
    from karl.db import SchedulerDB

    filename = 'db.sqlite.prod'
    db_name = 'karl-prod'

    db = SchedulerDB(filename)

    engine = create_engine(f'postgresql+psycopg2://shifeng@localhost:5433/{db_name}?host=/fs/clip-scratch/shifeng/postgres/run')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    '''
    for user in tqdm(db.get_user()):
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
            params=user.params,
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

    for record in tqdm(db.get_history()):
        record.elapsed_seconds_text = record.elapsed_seconds_text or 0
        record.elapsed_seconds_answer = record.elapsed_seconds_answer or 0
        record.elapsed_milliseconds_text = record.elapsed_milliseconds_text or record.elapsed_seconds_text * 1000
        record.elapsed_milliseconds_answer = record.elapsed_milliseconds_answer or record.elapsed_seconds_answer * 1000
        new_record = Record(
            record_id=record.history_id,
            debug_id=record.debug_id,
            user_id=record.user_id,
            fact_id=record.fact_id,
            deck_id=record.deck_id,
            response=bool(int(record.response)),
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
    '''

    # do one pass over records and get user stats with deck_id = 'all'
    for user in tqdm(session.query(User)):
        prev_stat = None
        curr_stat = None
        deck_id = 'all'
        for record in user.records:
            if curr_stat is None:
                user_stat_id = json.dumps({
                    'user_id': user.user_id,
                    'date': str(record.date.date()),
                    'deck_id': deck_id,
                })
                curr_stat = UserStat(
                    user_stat_id=user_stat_id,
                    user_id=user.user_id,
                    deck_id=deck_id,
                    date=record.date.date(),
                    new_facts=0,
                    reviewed_facts=0,
                    new_correct=0,
                    reviewed_correct=0,
                    total_seen=0,
                    total_milliseconds=0,
                    total_seconds=0,
                    total_minutes=0,
                    elapsed_milliseconds_text=0,
                    elapsed_milliseconds_answer=0,
                    elapsed_seconds_text=0,
                    elapsed_seconds_answer=0,
                    elapsed_minutes_text=0,
                    elapsed_minutes_answer=0,
                )
            elif curr_stat.date != record.date.date():
                session.add(curr_stat)
                session.commit()
                user_stat_id = json.dumps({
                    'user_id': user.user_id,
                    'date': str(record.date.date()),
                    'deck_id': deck_id,
                })
                new_stat = UserStat(
                    user_stat_id=user_stat_id,
                    user_id=user.user_id,
                    deck_id=deck_id,
                    date=record.date.date(),
                    new_facts=curr_stat.new_facts,
                    reviewed_facts=curr_stat.reviewed_facts,
                    new_correct=curr_stat.new_correct,
                    reviewed_correct=curr_stat.reviewed_correct,
                    total_seen=curr_stat.total_seen,
                    total_milliseconds=curr_stat.total_milliseconds,
                    total_seconds=curr_stat.total_seconds,
                    total_minutes=curr_stat.total_minutes,
                    elapsed_milliseconds_text=curr_stat.elapsed_milliseconds_text,
                    elapsed_milliseconds_answer=curr_stat.elapsed_milliseconds_answer,
                    elapsed_seconds_text=curr_stat.elapsed_seconds_text,
                    elapsed_seconds_answer=curr_stat.elapsed_seconds_answer,
                    elapsed_minutes_text=curr_stat.elapsed_minutes_text,
                    elapsed_minutes_answer=curr_stat.elapsed_minutes_answer,
                )
                curr_stat = new_stat

            if record.is_new_fact:
                curr_stat.new_facts += 1
                curr_stat.new_correct += record.response
            else:
                curr_stat.reviewed_facts += 1
                curr_stat.reviewed_correct += record.response

            if record.elapsed_milliseconds_text is None:
                record.elapsed_milliseconds_text = record.elapsed_seconds_text * 1000
            if record.elapsed_milliseconds_answer is None:
                record.elapsed_milliseconds_answer = record.elapsed_seconds_answer * 1000
            total_milliseconds = record.elapsed_milliseconds_text + record.elapsed_milliseconds_answer
            curr_stat.total_seen += 1
            curr_stat.total_milliseconds += total_milliseconds
            curr_stat.total_seconds += total_milliseconds // 1000
            curr_stat.total_minutes += total_milliseconds // 60000
            curr_stat.elapsed_milliseconds_text += record.elapsed_milliseconds_text
            curr_stat.elapsed_milliseconds_answer += record.elapsed_milliseconds_answer
            curr_stat.elapsed_seconds_text += record.elapsed_milliseconds_text // 1000
            curr_stat.elapsed_seconds_answer += record.elapsed_milliseconds_answer // 1000
            curr_stat.elapsed_minutes_text += record.elapsed_milliseconds_text // 60000
            curr_stat.elapsed_minutes_answer += record.elapsed_milliseconds_answer // 60000
        if curr_stat is not None:
            session.add(curr_stat)
        session.commit()

    # do one pass over records and get user stats with deck_id = 'all'
    for user in tqdm(session.query(User)):
        # group user records by deck_id first
        # then process each deck_id
        records_by_deck_id = defaultdict(list)
        for record in user.records:
            if record.deck_id is not None:
                records_by_deck_id[record.deck_id].append(record)

        for deck_id, records in records_by_deck_id.items():
            prev_stat = None
            curr_stat = None
            for record in records:
                if curr_stat is None:
                    user_stat_id = json.dumps({
                        'user_id': user.user_id,
                        'date': str(record.date.date()),
                        'deck_id': deck_id,
                    })
                    curr_stat = UserStat(
                        user_stat_id=user_stat_id,
                        user_id=user.user_id,
                        deck_id=deck_id,
                        date=record.date.date(),
                        new_facts=0,
                        reviewed_facts=0,
                        new_correct=0,
                        reviewed_correct=0,
                        total_seen=0,
                        total_milliseconds=0,
                        total_seconds=0,
                        total_minutes=0,
                        elapsed_milliseconds_text=0,
                        elapsed_milliseconds_answer=0,
                        elapsed_seconds_text=0,
                        elapsed_seconds_answer=0,
                        elapsed_minutes_text=0,
                        elapsed_minutes_answer=0,
                    )
                elif curr_stat.date != record.date.date():
                    session.add(curr_stat)
                    session.commit()
                    user_stat_id = json.dumps({
                        'user_id': user.user_id,
                        'date': str(record.date.date()),
                        'deck_id': deck_id,
                    })
                    new_stat = UserStat(
                        user_stat_id=user_stat_id,
                        user_id=user.user_id,
                        deck_id=deck_id,
                        date=record.date.date(),
                        new_facts=curr_stat.new_facts,
                        reviewed_facts=curr_stat.reviewed_facts,
                        new_correct=curr_stat.new_correct,
                        reviewed_correct=curr_stat.reviewed_correct,
                        total_seen=curr_stat.total_seen,
                        total_milliseconds=curr_stat.total_milliseconds,
                        total_seconds=curr_stat.total_seconds,
                        total_minutes=curr_stat.total_minutes,
                        elapsed_milliseconds_text=curr_stat.elapsed_milliseconds_text,
                        elapsed_milliseconds_answer=curr_stat.elapsed_milliseconds_answer,
                        elapsed_seconds_text=curr_stat.elapsed_seconds_text,
                        elapsed_seconds_answer=curr_stat.elapsed_seconds_answer,
                        elapsed_minutes_text=curr_stat.elapsed_minutes_text,
                        elapsed_minutes_answer=curr_stat.elapsed_minutes_answer,
                    )
                    curr_stat = new_stat

                if record.is_new_fact:
                    curr_stat.new_facts += 1
                    curr_stat.new_correct += record.response
                else:
                    curr_stat.reviewed_facts += 1
                    curr_stat.reviewed_correct += record.response

                if record.elapsed_milliseconds_text is None:
                    record.elapsed_milliseconds_text = record.elapsed_seconds_text * 1000
                if record.elapsed_milliseconds_answer is None:
                    record.elapsed_milliseconds_answer = record.elapsed_seconds_answer * 1000
                total_milliseconds = record.elapsed_milliseconds_text + record.elapsed_milliseconds_answer
                curr_stat.total_seen += 1
                curr_stat.total_milliseconds += total_milliseconds
                curr_stat.total_seconds += total_milliseconds // 1000
                curr_stat.total_minutes += total_milliseconds // 60000
                curr_stat.elapsed_milliseconds_text += record.elapsed_milliseconds_text
                curr_stat.elapsed_milliseconds_answer += record.elapsed_milliseconds_answer
                curr_stat.elapsed_seconds_text += record.elapsed_milliseconds_text // 1000
                curr_stat.elapsed_seconds_answer += record.elapsed_milliseconds_answer // 1000
                curr_stat.elapsed_minutes_text += record.elapsed_milliseconds_text // 60000
                curr_stat.elapsed_minutes_answer += record.elapsed_milliseconds_answer // 60000
            if curr_stat is not None:
                session.add(curr_stat)
            session.commit()

    session.close()