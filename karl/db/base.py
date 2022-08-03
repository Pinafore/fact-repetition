# Import all the models, so that Base has them before being
# imported by Alembic
from karl.db.base_class import Base  # noqa
import karl.models
