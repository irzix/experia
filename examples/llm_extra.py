"""Installed constructor smoke example for the documented ``experia[llm]`` extra."""

from experia import LiteLLMEmbedder, SQLiteStore
from experia.experience.llm_evaluator import LLMEvaluator
from experia.improvement.rules import RuleGenerator

store = SQLiteStore(":memory:")
evaluator = LLMEvaluator(model="gpt-4o-mini")
embedder = LiteLLMEmbedder(model="text-embedding-3-small")
rules = RuleGenerator(store=store, model="gpt-4o-mini")

assert evaluator.model == "gpt-4o-mini"
assert embedder.model == "text-embedding-3-small"
assert rules.store is store
