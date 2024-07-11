# EARIN - 2nd mini-project
The goal is to write a program that plays "Connect 4" game with the user using min-max algorithm with alpha-beta pruning. The board of game is a 2D matrix with entries: “ “ (space) – empty cell, “X” – first player, “O” – second player.

Requirements for the game:
- User chooses to play first or second
- Depending on the choice, the first player makes move with “X”
- Players make moves one by one
- During his/her move, the player chooses the column where drop his mark
- Game finishes if one of the players has 4 of his marks in a row, column, or diagonal
- Game finishes if the board is full, in this case the result of the game is draw

Proper evaluation function will be implemented using functions `evaluate_position` and `evaluate_window` given in the code template.
The board will be printed after every move. Instructions will also be provided into the console for the user, trying to the game user-friendly. The result of the game will be displayed to the console at the end.
The code will be tested by different scenarios of the game, to make sure it works properly. The inputs of the user will be tested for correctness. Computer strategy efficiency will also be taken into account.

The evaluation function used for scoring the position of the game will be discussed in a report, as well as weaknesses and strengths of the chosen evaluation strategy.
