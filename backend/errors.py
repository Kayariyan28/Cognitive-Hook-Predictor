"""Domain errors surfaced by the TRIBE v2 backend."""


class TribeBackendError(RuntimeError):
    """Base class for expected backend failures."""


class ConfigurationError(TribeBackendError):
    """The service is not configured for reproducible model loading."""


class ModelUnavailableError(TribeBackendError):
    """The real TRIBE v2 model could not be loaded."""


class InferenceError(TribeBackendError):
    """The official TRIBE v2 inference pipeline failed."""


class PredictionValidationError(TribeBackendError, ValueError):
    """TRIBE returned output that cannot be mapped to fsaverage5 safely."""


class UploadValidationError(TribeBackendError, ValueError):
    """The uploaded file is not an accepted video input."""


class ThumbnailExtractionError(TribeBackendError, ValueError):
    """A real source-video still could not be decoded or validated."""
