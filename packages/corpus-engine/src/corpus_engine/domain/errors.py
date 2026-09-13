"""Public engine exceptions."""


class CorpusEngineError(Exception):
    """Base exception for engine failures."""


class UnsupportedFormatError(CorpusEngineError):
    """Raised when no parser is registered for a document format."""


class MissingDependencyError(CorpusEngineError):
    """Raised when an optional adapter dependency is unavailable."""
