#!/usr/bin/env python
# coding: utf-8

# In[1]:


from karl.db.session import SessionLocal
from karl.models import UserSnapshot, UserSnapshotV2
from tqdm import tqdm


# In[5]:


session = SessionLocal()
user_snapshots = session.query(UserSnapshot)
for snapshot in tqdm(user_snapshots, total=user_snapshots.count()):
    snapshot = snapshot.__dict__
    snapshot['schedule_request_id'] = snapshot['id']
    snapshot.pop('_sa_instance_state')
    snapshot.pop('id')
    snapshot_v2 = UserSnapshotV2(**snapshot)
    session.add(snapshot_v2)
session.commit()


# In[6]:


from karl.models import CardSnapshot, CardSnapshotV2

session = SessionLocal()
snapshots = session.query(CardSnapshot)
for snapshot in tqdm(snapshots, total=snapshots.count()):
    snapshot = snapshot.__dict__
    snapshot['schedule_request_id'] = snapshot['id']
    snapshot.pop('_sa_instance_state')
    snapshot.pop('id')
    snapshot_v2 = CardSnapshotV2(**snapshot)
    session.add(snapshot_v2)
session.commit()


# In[8]:


from karl.models import UserCardSnapshot, UserCardSnapshotV2

session = SessionLocal()
snapshots = session.query(UserCardSnapshot)
for snapshot in tqdm(snapshots, total=snapshots.count()):
    snapshot = snapshot.__dict__
    snapshot['schedule_request_id'] = snapshot['id']
    snapshot.pop('_sa_instance_state')
    snapshot.pop('id')
    snapshot_v2 = UserCardSnapshotV2(**snapshot)
    session.add(snapshot_v2)
session.commit()

