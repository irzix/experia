"""Installed constructor smoke example for ``experia[langgraph]``."""

from experia import Learner, SimpleHeuristicEvaluator, SQLiteStore
from experia.integrations.langgraph.nodes import ExperiaContextNode, ExperiaLearningNode

store = SQLiteStore(":memory:")
learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
context_node = ExperiaContextNode(agent=learner, limit=3)
learning_node = ExperiaLearningNode(agent=learner, callback_mode="durable")

assert context_node.agent is learner
assert context_node.limit == 3
assert learning_node.agent is learner
assert learning_node.callback_mode == "durable"
