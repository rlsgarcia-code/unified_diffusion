class UnifiedDiffusionError(Exception):
    """Base error for the package."""


class ModelNotFoundError(UnifiedDiffusionError):
    """Raised when the requested canonical model id is unknown."""


class ProviderError(UnifiedDiffusionError):
    """Raised when a provider cannot be used or fails."""


class CacheError(UnifiedDiffusionError):
    """Raised when cache preparation or logging fails."""


class RegistryValidationError(UnifiedDiffusionError):
    """Raised when a registry entry is malformed or unsupported."""
