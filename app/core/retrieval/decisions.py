from enum import Enum


class RetrievalDecision(str, Enum):
    ANSWERABLE = "answerable"
    ANSWERABLE_LOW_CONFIDENCE = "answerable_low_confidence"
    REFUSE_EMPTY = "refuse_empty"
    REFUSE_WEAK = "refuse_weak"
