"""Thread-local invocation metadata: target and draft run on different GPUs."""
from contextvars import ContextVar
_CONTEXT = ContextVar('attention_context')
def get_context():
    return _CONTEXT.get()
def set_context(context):
    return _CONTEXT.set(context)
def reset_context(token):
    _CONTEXT.reset(token)
