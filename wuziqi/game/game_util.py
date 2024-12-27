import enum

@enum.unique
class GamePieces(enum.Enum):
    WHITE = enum.auto()
    BLACK = enum.auto()