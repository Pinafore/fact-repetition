# Improve Human Learning with Representation + Spaced Repetition

## 03-07-2020
- in simulation, the probability of the card being correct should start with
  estimated from Roger's data then grow with reptition

## 03-06-2020
x dist_time is wrong, very large values
x how come timedelta is negative? timedelta is negative when card.date is
  earlier than user.last_study_date[card.card_id]
- check topic assignment, not just topic-word dist
x maybe card date should be a field but not stored in DB, so is user date
x flashcard should be renamed to schedule_request

## 03-25-2020
- finalze LDA on Quizbowl
- "point" and "10" are still in the vocab
x improve debug string
x make sure that commiting only at exit is fine
x return schedule & update details via web API for debug
- weird sequences in possibly snapshot

## 03-23-2020
x fix dist_skill
x bypass scheduler.retrieve

## 03-20-2020
x query speed optimization
x plots & probing & retention next steps
x discount
x LDA improvement
- define labels for Quizbowl sentences
x simulation

### Query speed optimization
Currently the speed of the schedule API is bad. The bottleneck is doing
retrieval on hundreds of cards for each schedule API call. For each given card,
we search for similar / identical cards in Roger’s dataset. The underlying
assumption is that Roger’s dataset is useful for retention prediction. We can
evaluate this, broken down into 4 categories:

1. seen Jeopardy cards
2. unseen Jeopardy cards
3. Quizbowl cards
4. OOD cards

1 & 2 can be tested with Wency’s train / test split and standard retention
evaluation - find the best threshold and convert to binary classification.
It’s not likely that Roger’s data is useful for 3 & 4, so we need to do OOD
detection on fall back to the trained model for those.

If it turns out retrieval on Roger’s data is good for 1 & 2, we then do
optimization. 1 is easy to speed up, we just run the matching on all of
Mattew’s cards and store the mapping as a dictionary. 2 is also not bad, we
still only need to run the retrieval for each card once, and then it can be
cached. Retrieval can always run offline async.

If it turns out retrieval on Roger’s data is not good, then we depend on the
retention model.
If it turns out that we can’t do much with the retention model, I guess our
whole project just reduces to regular spaced repetition + topical coherence?

### Discount
User’s question representation should be a discounted average of question
representations. We should store a short history of question representations in
a numpy matrix, discount factors in a numpy vector, then the current user
representation is the dot product between the two.

User’s skill level estimate is similarly a discounted average of question
skills. The difference between skill level estimates only goes up if the
response is correct.

x LDA improvement
x gensim coherence score
- labeled LDA

## 03-18-2020
x profile fastapi / flask
x gensim mallet LDA
- firebase
- long term simulation: use fake datetime, on the scale of days
x document term frequency instead of global term frequency
- category classification from topic embedding to evaluate topic model outputs
- use answers as features
- think about the bayesian non-parametric interpretation Jordan mentioned
- long-term simulation to check repetition
x transform pretrained / fine-tuned BERT to predict category directly and see
  if there is a difference between accuracy
- myleott/JGibbLabeledLDA: Labeled LDA in Java (based on JGibbLDA)
- Collaborative Topic Modeling for Recommending Scientific Articles

## 03-09-2020
x make backend persistent; sync with front-end
x removed the assumption of single user update

## 03-08-2020
### Add explanations
x explain in terms of components of the distance metric
x front-end needs to call schedule every step otherwise you don’t get the most
  up-to-date explanations / user status

### LDA
- LDA sometime disagrees with Category of the question. need to look at LDA
  experiments and see if we can rely on LDA to do difficulty / skill level
  updates.
- we can also do soft updates: update to probability of topic i is multiplied
  by assignment to topic i

### DB for scheduler
x we’ll need to have a way to synchronize scheduler DB and front-end DB

## 03-07-2020
Look into LDA: New scheduler relies on LDA for topic embedding, but I haven't
looked at how the topics look like. we need to look at LDA output and tune
several things:
1. data that it's trained on
2. number of topics
3. size of vocabulary

- lack of repetition: need discount factor or have band of difficulty estimate
  to select cards
- we are discouraging repetition of wrong cards
- simulate days between studies

## 03-04-2020
- scheduling latency: front-end, internet, or backend?
- Better difficulty estimate: Right now we're using question average accuracy
  from user records to estimate question difficulty. It doesn't work well on
  questions with low number of entries. There are several possible improvements
  to the current retention model:
- We are not doing QA, so answer can be used as a feature. Frequency of the
  answer is a strong indicator of difficulty. Use answer as a feature in
  retention model.
- We should redefine the objective. Right now we are doing user-based retention
  prediction, but we can possibly move to difficulty prediction for the general
  audience. We can directly predict the average question accuracy.
- We are not using all the data we have. All models so far are limited to
  Quizbowl or Jeopardy, we never combined the two, and there are more
  resources.
- More complicated features might be powerful, such as frequency of entities
  mentioned in a question.
- The Quizbowl training data might be problematic. Need Wenyan to recreate
- Project out difficulty and PCA again.
- Add back user features
- Matthew dataset test
- User visualization: For each user, each question he answered is a point on
  the polar plot. Red if wrong, blue if correct. This forms a heatmap that
  shows what topics the user is good / bad at.
- Discount: We probably need this to compare against Settles.
- AWS download for question and records file
- Auto-restart
