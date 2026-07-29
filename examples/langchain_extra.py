"""Installed constructor smoke example for ``experia[langchain]``."""

from experia import Learner, SimpleHeuristicEvaluator, SQLiteStore
from experia.integrations.langchain.callbacks import ExperiaCallbackHandler
from experia.integrations.langchain.retrievers import ExperiaLearningRetriever

store = SQLiteStore(":memory:")
learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
handler = ExperiaCallbackHandler(agent=learner, callback_mode="durable")
retriever = ExperiaLearningRetriever(agent=learner, limit=3)

assert handler.builder.agent is learner
assert retriever.agent is learner
assert retriever.limit == 3
