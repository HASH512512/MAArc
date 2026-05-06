class IgnoreDesignantLine(ValueError):
    """Control-flow exception used by permissive AFF parsing."""

    def __init__(self) -> None:
        super().__init__("IGNORE_DESIGNANT_LINE")


class MissingDesignantChoiceError(ValueError):
    """Raised when parser hits designant notes without a decision."""

    def __init__(self) -> None:
        super().__init__("MISSING_DESIGNANT_CHOICE")
