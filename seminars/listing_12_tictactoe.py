"""
Programming 2026.

Seminar 12.

Brainstorm from the lecture on designing a TicTacToe game.
"""

# pylint:disable=too-few-public-methods
from typing import Literal

MarkerType = Literal["X", "O"]


class Move:
    """
    Store information about move: coordinates and label.

    Instance attributes:
        row (int): Index of row
        col (int): Index of column
        label (MarkerType): Label to put

    Instance methods:
        N/A
    """


class Player:
    """
    Enable player functionality: store playing label (O or X), make moves.

    Instance attributes:
        label (MarkerType): label to play (O or X)

    Instance methods:
        make_move(self, row: int, col: int) -> Move: Create instance of Move

    """


class Board:
    """
    Store moves made on the playing board and show its state.

    Instance attributes:
        _size (int): size of playing board (most commonly 3)
        _moves_left (int): number of empty cells on the playing board
        _moves (list[Move]): already made moves

    Instance methods:
        show(self, ...): Print current state of the board
        add_move(self, ...): Add new valid move
        get_moves(self, ...): Get already made moves
        get_size(self, ...): Get size of board
    """


class Game:
    """
    Store game status and enable moves.

    Instance attributes:
        _board (Board): playing board (most commonly 3x3)
        _players (tuple[Player, ...]): tuple with 2 players
        _current_player_idx (int): index of the player that should make a move next
        _finished (bool): flag if the game has finished: there was winner or tie

    Instance methods:
        _next_player(self): Update the next player to make a move.
        _check_move(self, ...): Verify that the move can be made.
        _register_move(self, ...): Put the move on the playing board.
        _check_for_winner(self, ...): Check if win state is present
        play(self, ...): Process one step of game
    """


def main() -> None:
    """
    Launch tic-tac-toe game.
    """
    # 1. Create players
    print("Created players")

    # 2. Create game
    print("Created game")

    # 3. Make move
    print("Made move")

    # 4. Register move
    print("Registered move")

    # 5. Show current state
    print("Showed current state")


if __name__ == "__main__":
    main()
