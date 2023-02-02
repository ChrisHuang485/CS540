import random
import copy

class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
    
    def check_drop_phase(self,state):
        num = sum([row.count('r') for row in state]) + sum([row.count('b') for row in state])
        
        if(num < 8):
            return True
        return False
    
    def succ(self, state, piece): 
        drop_phase = self.check_drop_phase(state)
        result = []
        temp = copy.deepcopy(state)
        if drop_phase:
            for row in range(5):
                for col in range(5):
                    if(state[row][col] == ' '):
                        temp[row][col] = piece
                        result.append(temp)
                        temp = copy.deepcopy(state)
        else:
            for row in range(5):
                for col in range(5):
                    if(temp[row][col] == piece):
                        if(row-1>=0 and col-1>=0) and temp[row-1][col-1] == ' ':
                            temp[row][col] = ' '
                            temp[row-1][col-1] = piece
                            result.append(temp)
                            temp = copy.deepcopy(state)
                          
                        if(row-1>=0) and temp[row-1][col] == ' ':
                            temp[row][col] = ' '
                            temp[row-1][col] = piece
                            result.append(temp)
                            temp = copy.deepcopy(state)
                        
                        if(row-1>=0) and (col+1<=4) and temp[row-1][col+1] == ' ':
                            temp[row][col] = ' '
                            temp[row-1][col+1] = piece
                            result.append(temp)
                            temp = copy.deepcopy(state)
                        
                        if(row+1<=4) and (col-1>=0) and temp[row+1][col-1] == ' ':
                            temp[row][col] = ' '
                            temp[row+1][col-1] = piece
                            result.append(temp)
                            temp = copy.deepcopy(state)
                        
                        if(row+1<=4) and (col+1<=4) and temp[row+1][col+1] == ' ':
                            temp[row][col] = ' '
                            temp[row+1][col+1] = piece
                            result.append(temp)
                            temp = copy.deepcopy(state)
                            
                        if(col+1<=4) and temp[row][col+1] == ' ':
                            temp[row][col] = ' '
                            temp[row][col+1] = piece
                            result.append(temp)
                            temp = copy.deepcopy(state)
                        
                        if(row-1>=0) and temp[row-1][col] == ' ':
                            temp[row][col] = ' '
                            temp[row-1][col] = piece
                            result.append(temp)
                            temp = copy.deepcopy(state)
                        
                        if(col-1>=0) and temp[row][col-1] == ' ':
                            temp[row][col] = ' '
                            temp[row][col-1] = piece
                            result.append(temp)
                            temp = copy.deepcopy(state)
                                  
        return result

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
  
        drop_phase = self.check_drop_phase(state)   
        move = []
        n_range = -100000
        p_range = 100000
        move_next = None
  
        for s in self.succ(state, self.my_piece):
            succ_val = self.min_value(n_range, p_range, s, 0)
            if n_range < succ_val:
                move_next = s
                n_range = succ_val
                
        if not drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] != ' ' and state[i][j] != move_next[i][j]:
                        (source_row, source_col) = (i,j)
                        move.append((source_row, source_col))

                        

        for i in range(5):
            for j in range(5):
                if state[i][j] == ' ' and state[i][j] != move_next[i][j]:
                    (row, col) = (i,j)
                    move.insert(0, (row, col))
        return move
    

    
    def max_row(self, state, piece):
        for row in state:
            for col in range(2):
                max_row = 0
                for i in range(3):
                    if row[col+i] == piece:
                        max_row += 0.25
                    elif row[col+i] != ' ':
                        max_row -= 0.05
                if max_row > 0:
                    return max_row
                else:
                    return 0

    
    def max_col(self, state, piece):
        for col in range(5):
            for row in range(2):
                max_col = 0
                for i in range(3):
                    if state[row+i][col] == piece:
                        max_col += 0.25
                    elif state[row+i][col] != ' ':
                        max_col -= 0.05
                if max_col > 0:
                    return max_col
                else:
                    return 0

                
    def max_d1(self, state, piece):
        for row in range(2):
            for col in range(2):
                max_d1 = 0
                for i in range(3):
                    if state[row+i][col+i] == piece:
                        max_d1 += 0.25
                    elif state[row+i][col+i] != ' ':
                        max_d1 -= 0.04
                if max_d1 > 0:
                    return max_d1
                else:
                    return 0

    
    def max_d2(self, state, piece):
        for row in range(2):
            for col in range(3,5):
                max_d2 = 0
                for i in range(3):
                    if state[row+i][col-i] == piece:
                        max_d2 += 0.25
                    elif state[row+i][col-i] != ' ':
                        max_d2 -= 0.04
                if max_d2 > 0:
                    return max_d2
                else:
                    return 0
                
    def max_square(self, state, piece):
        for row in range(3):
            for col in  range(3):
                max_square = 0
                if state[row][col] == piece:
                    max_square += 0.25
                if state[row][col+2] == piece:
                    max_square += 0.25
                if state[row+2][col] == piece:
                    max_square += 0.25
                if state[row+2][col+2] == piece:
                    max_square += 0.25
                if state[row+1][col+1] != ' ':
                    max_square -= 0.20
                if max_square > 0:
                    return max_square
                else:
                    return 0

    
    def heuristic_game_value(self, state, piece):
        
        if self.game_value(state) != 0:
            return self.game_value(state)  
        
        max_row = self.max_row(state, piece) + random.uniform(-0.02,0.02)
        max_col = self.max_col(state, piece) + random.uniform(-0.02,0.02)
        max_d1 = self.max_d1(state, piece) + random.uniform(-0.02,0.02)
        max_d2 = self.max_d2(state, piece) + random.uniform(-0.02,0.02)
        max_square = self.max_square(state, piece) + random.uniform(-0.02,0.02)
        
        if piece == self.my_piece:
            return max(max_row, max_col, max_d1, max_d2, max_square)
        else:
            return -1 * max(max_row, max_col, max_d1, max_d2, max_square)

        
    def max_value(self, alpha, beta, state, h):
        if(self.game_value(state) != 0):
            return self.game_value(state)
        if(h >= 1):
            return self.heuristic_game_value(state,self.my_piece)
        for succ in self.succ(state, self.my_piece):
            alpha = max(alpha, self.min_value(alpha, beta, succ, h + 1))
            if alpha >= beta:
                return beta
        return alpha
    
    def min_value(self, alpha, beta, state, h):
        if(self.game_value(state) != 0):
            return self.game_value(state)
        if(h >= 1):
            return self.heuristic_game_value(state,self.opp)
        for succ in self.succ(state, self.opp):
            beta = min(beta, self.max_value(alpha, beta, succ, h + 1))
            if alpha >= beta:
                return beta
        return beta
    

    
    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i] == self.my_piece else -1

        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        for i in range(2):
            for j in range(2):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3]:
                    return 1 if state[i][j] == self.my_piece else -1

        for i in range(2):
            for j in range(3,5):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j-1] == state[i+2][j-2] == state[i+3][j-3]:
                    return 1 if state[i][j] == self.my_piece else -1

        for i in range(3):
            for j in range(3):
                if state[i][j] != ' ' and state[i][j] == state[i][j+2] == state[i+2][j] == state[i+2][j+2]:
                    if state[i+1][j+1] == ' ':
                        return 1 if state[i][j] == self.my_piece else -1
        return 0

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
