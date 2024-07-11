import copy
import random

EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'
ROWS = 6
COLS = 7


class ConnectFour:
    """
    Class for game Connect 4

    """
    def __init__(self):
        self.board = [[EMPTY] * COLS for _ in range(ROWS)]
        self.current_player = PLAYER_X

    def print_board(self):
        for row in self.board:
            print('|'.join(row))
        print('-' * (COLS * 2 - 1))
        print(' '.join(str(i) for i in range(COLS)))

    # Implement any additional functions needed here

    def evaluate_window(self, window, piece):
        """
        Evaluation of given window. Helper function to evaluate the separate parts of the board called windows.
        A window is a list of 4 elements, which can be a row, column, or diagonal of the board.

        Parameters:
        - window: list containing values of evaluated window
        - piece: PLAYER_X or PLAYER_O depending on which player's position we evaluate

        Returns:
        - score of the window

        """
        # window is the first 4 elements of the given list
        window = window[:4]

        player = piece
        opponent = PLAYER_X if piece == PLAYER_O else PLAYER_O

        score = 0

        # the numeric values given bellow are arbitrary and can be changed
        # We set them as seen in order to account for the different 'weights' of the different situations
        # We prioritize winning, then positions that are close to winning, but we give them a much lower score.
        # Finally, a position that is close to winning for the opponent is given a negative score, as we want to avoid that.
        if window.count(player) == 4:
            score += 100

        elif window.count(player) == 3 and window.count(EMPTY) == 1:
            score += 8

        elif window.count(player) == 2 and window.count(EMPTY) == 2:
            score += 4

        if window.count(opponent) == 3 and window.count(EMPTY) == 1:
            score -= 6

        return score

    def evaluate_position(self, board, piece):
        """
        Evaluation of position
        Parameters:
        - board: 2d matrix representing evaluated state of the board
        - piece: PLAYER_X or PLAYER_O depending on which player's position we evaluate

        Returns:
        - score of the position

        """

        score = 0

        # Score center column
        # We give extra weight to the center column, as it is the column that gives most options
        # for creating winning patterns. Thus we also include a multiplier to the count of our pieces
        # in it.
        center_array = [board[r][COLS // 2] for r in range(ROWS)]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Score Horizontal
        for r in range(ROWS):
            row_array = [board[r][c] for c in range(COLS)]
            for c in range(COLS - 3):
                window = row_array[c:c + 4]
                score += self.evaluate_window(window, piece)

        # Score Vertical
        for c in range(COLS):
            col_array = [board[r][c] for r in range(ROWS)]
            for r in range(ROWS - 3):
                window = col_array[r:r + 4]
                score += self.evaluate_window(window, piece)

        # Score positive sloped diagonal
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        # Score negative sloped diagonal
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [board[r + 3 - i][c + i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        
        return score




    def minimax(self, board, depth, maximizing_player, alpha, beta):
        """
        Minimax with alpha-beta pruning algorithm

        Parameters:
        - board: 2d matrix representing the state, each cell contains either ' ' (empty cell), 'X' (player1), or 'O' (player2) 
        - depth: depth
        - maximizing_player: boolean which is equal to True when the player tries to maximize the score
        - alpha: alpha variable for pruning
        - beta: beta variable for pruning

        Returns:
        - Best value 
        - Best move found

        """

        #Your code starts here

        # determine the current player and the opponent
        player = self.current_player
        opponent = PLAYER_X if player == PLAYER_O else PLAYER_O

        # get the valid locations, used later to determine the best move
        valid_locations = self.get_valid_locations(board)
        # check if the game is over
        is_terminal = self.is_terminal(board)

        if depth == 0 or is_terminal:
            if is_terminal:
                if self.is_winner(board, player):
                    return (1_000_000_000, None) # a really big score
                elif self.is_winner(board, opponent):
                    return (-1_000_000_000, None) # a similarly big negative score
                else: # Game is over, no more valid moves
                    return (0, None)
                
            else: # Depth is zero
                return (self.evaluate_position(board, player), None)
            
        if maximizing_player: # Maximizing player
            value = float('-inf')
            column = random.choice(valid_locations) # start with a random column
            for col in valid_locations:
                row = self.get_open_row(board, col) # get the row where the piece will fall
                board_copy = copy.deepcopy(board) # create a copy of the board
                self.drop_piece(board_copy, row, col, player) # drop the piece to the board
                new_score = self.minimax(board_copy, depth - 1, False, alpha, beta)[0]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value) # keep increasing alpha
                if alpha >= beta:
                    break
            return value, column
        
        else: # Minimizing player
            value = float('inf')
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_open_row(board, col)
                board_copy = copy.deepcopy(board)
                self.drop_piece(board_copy, row, col, opponent)
                new_score = self.minimax(board_copy, depth - 1, True, alpha, beta)[0]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value) # keep decreasing beta
                if alpha >= beta:
                    break
            return value, column
    
    def is_winner(self, board, piece):
        """
        Check if the player has won

        Parameters:
        - board: 2d matrix representing the state, each cell contains either ' ' (empty cell), 'X' (player1), or 'O' (player2) 
        - piece: PLAYER_X or PLAYER_O depending on which player's position we evaluate

        Returns:
        - True if the player has won, False otherwise

        """

        # Check horizontal locations for win
        for c in range(COLS - 3):
            for r in range(ROWS):
                if board[r][c:c + 4].count(piece) == 4:
                    return True
                
        # Check vertical locations for win
        for c in range(COLS):
            for r in range(ROWS - 3):
                if [board[r + i][c] for i in range(4)].count(piece) == 4:
                    return True
                
        # Check positively sloped diagonals
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if [board[r + i][c + i] for i in range(4)].count(piece) == 4:
                    return True
                
        # Check negatively sloped diagonals
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if [board[r + 3 - i][c + i] for i in range(4)].count(piece) == 4:
                    return True
        
        return False
    
    def is_valid_move(self, board, col):
        """
        Check if the move is valid

        Parameters:
        - board: 2d matrix representing the state, each cell contains either ' ' (empty cell), 'X' (player1), or 'O' (player2) 
        - col: column where the player wants to put the piece

        Returns:
        - True if the move is valid, False otherwise

        """

        return board[0][col] == EMPTY
    
    def get_valid_locations(self, board):
        """
        Get valid locations

        Parameters:
        - board: 2d matrix representing the state, each cell contains either ' ' (empty cell), 'X' (player1), or 'O' (player2) 

        Returns:
        - list of valid locations

        """

        valid_locations = []
        for col in range(COLS):
            if self.is_valid_move(board, col):
                valid_locations.append(col)
        return valid_locations
    
    def is_terminal(self, board):
        """
        Check if the game is over

        Parameters:
        - board: 2d matrix representing the state, each cell contains either ' ' (empty cell), 'X' (player1), or 'O' (player2) 

        Returns:
        - True if the game is over, False otherwise

        """

        return self.is_winner(board, PLAYER_X) or self.is_winner(board, PLAYER_O) or len(self.get_valid_locations(board)) == 0
        
    def get_open_row(self, board, col):
        """
        Get the open row

        Parameters:
        - board: 2d matrix representing the state, each cell contains either ' ' (empty cell), 'X' (player1), or 'O' (player2) 
        - col: column where the player wants to put the piece

        Returns:
        - row where the piece will be placed

        """

        # we prefer the deepest row, so we start from the bottom
        for r in range(ROWS - 1, -1, -1):
            if board[r][col] == EMPTY:
                return r
            

    def drop_piece(self, board, row, col, piece):
        """
        Drop the piece to the board

        Parameters:
        - board: 2d matrix representing the state, each cell contains either ' ' (empty cell), 'X' (player1), or 'O' (player2) 
        - row: row where the piece will be placed
        - col: column where the player wants to put the piece
        - piece: PLAYER_X or PLAYER_O depending on which player's position we evaluate

        """

        board[row][col] = piece

    def set_players(self, turn):
        """
        Set the current player

        Parameters:
        - turn: turn of the player (0 means player goes first, 1 means computer goes first)

        """

        self.human_player = PLAYER_X if turn == 0 else PLAYER_O
        self.computer_player = PLAYER_O if turn == 0 else PLAYER_X

def main():
    """
    Main game loop implementation. Player1 should play first with 'X', player2 plays second with 'O'
    """
    game = ConnectFour()
    print("\nWelcome to Connect Four!\n")
    while True:
        try:
            # our code: set the player
            print("Enter '1' to play first or '2' to play second")
            turn = int(input()) - 1
            if turn not in[0, 1]:
                print("\nInvalit number\n")
                continue
            game.set_players(turn)
            break
        except ValueError:
            print("\nInvalid input. Please enter a number\n")
            
    while True:
        game.print_board()
        print()

        if game.current_player == game.human_player:
            print("Human turn")
            while True:
                try:
                    col = int(input("Enter column (0-6): "))
                    if col < 0 or col >= COLS:
                        print("\nColumn must be between 0 and 6\n")
                        continue
                    elif not game.is_valid_move(game.board, col):
                        print("\nColumn is full. Try a different one\n")
                        continue
                    row = game.get_open_row(game.board, col)
                    game.drop_piece(game.board, row, col, game.human_player)
                    break
                except ValueError:
                    print("\nInvalid input. Please enter a number\n")
                    continue

            if game.is_winner(game.board, game.human_player):
                game.print_board()
                print("Human wins!\n")
                break
            game.current_player = game.computer_player

        else:
            print("Computer turn")
            _, col = game.minimax(game.board, 5, True, float('-inf'), float('inf'))
            if game.is_valid_move(game.board, col):
                row = game.get_open_row(game.board, col)
                game.drop_piece(game.board, row, col, game.computer_player)
                if game.is_winner(game.board, game.computer_player):
                    game.print_board()
                    print()
                    print("Computer wins!\n")
                    break
                game.current_player = game.human_player

        # check if the game is over
        if game.is_terminal(game.board):
            game.print_board()
            print("It's a tie!")
            break

if __name__ == "__main__":
    main()
