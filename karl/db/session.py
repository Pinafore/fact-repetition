from karl.config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(settings.SQLALCHEMY_DATABASE_URL, pool_pre_ping=True, pool_size=80)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
