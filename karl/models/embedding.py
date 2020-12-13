import msgpack
import msgpack_numpy
import sqlalchemy.types as types
from sqlalchemy import Column, ForeignKey, String

from karl.db.base_class import Base
from karl.models import Card


class BinaryNumpy(types.TypeDecorator):

    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        return msgpack.packb(value, default=msgpack_numpy.encode)

    def process_result_value(self, value, dialect):
        return msgpack.unpackb(value, object_hook=msgpack_numpy.decode)


class Embedding(Base):
    id = Column(String, ForeignKey(Card.id), primary_key=True, index=True)
    embedding = Column(BinaryNumpy, nullable=False)
