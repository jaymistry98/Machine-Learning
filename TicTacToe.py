'''

@project       : Queens College CSCI 381/780 Machine Learning
@Instructor    : Dr. Alex Pang

@Date          : Spring 2022

A Object-Oriented Implementation of the TicTacToe Game 

references

'''

import enum

class GamePiece(enum.Enum):
    CROSS = "X"
    CIRCLE = "O"
    

class GameBoard(object):
    '''
    TODO: explain what the class is about, definition of various terms
    etc
    '''
    
    def __init__(self):
        self.nsize = 3
        self._board = []
        for r in range(self.nsize):
            self._board.append(['' for c in range(self.nsize)])

        # set the board to its initial state
        self.reset()
        
    def display(self):
        '''
        display the game board which will look like something like this

           1 | X | 3
           4 | 5 | O
           7 | 8 | 9

        '''

        # TODO
        num = 1
        
        for row in self._board:
            for place in row:
                if (num == 'X'):
                    print("", 'X'  , "|", end="")
                elif (num == 'O'):
                    print("", 'O'  , "|", end="")
                else:
                    print("", num, "|", end="") 
                num += 1
            print()
        
        # end TODO

    def reset(self):
        '''
        Reset the game board so that each cell is index from 1 to 9.
        So when display it will look like

           1 | 2 | 3
           4 | 5 | 6
           7 | 8 | 9

        '''

        # TODO
        num = 1
        for row in self.board:
            for place in row:
                print("", num, "|", end="")
                num += 1
            print()
        # end TODO


    
    def place_into(self, symbol, spot):
        '''
        Find the cell that spot is located, then replace the cell by 
        the symbol X or O
        '''

        # TODO
        num = 1
        for row in self._board:
            for place in row:
                if (num == self.spot):
                    str(num).replace(self.symbol) 
                num += 1 
        # end TODO


    def has_winner(self):
        '''
        Determine if one side has won (ie a winning row, column or a winning diagonal.
        If there is a winner, display who is the winner and return true
        otherwise return false
        '''
        # TODO
        if ((self.board[0][0] == self.board[0][1] == self.board[0][2]) or
            (self.board[1][0] == self.board[1][1] == self.board[1][2]) or
            (self.board[2][0] == self.board[2][1] == self.board[2][2]) or
            (self.board[0][0] == self.board[1][0] == self.board[2][0]) or
            (self.board[0][1] == self.board[1][1] == self.board[2][1]) or
            (self.board[0][2] == self.board[1][2] == self.board[2][2]) or
            (self.board[0][0] == self.board[1][1] == self.board[2][2]) or
            (self.board[2][0] == self.board[1][1] == self.board[0][2])):
            return True
        else:
            return False
        # end TODO

    def is_valid(self, spot):
        '''
        return true if spot is a valid location that you can place a symbol into
        ie. it has not been occupied by an X or an O
        '''

        # TODO
        for row in self._board:
            for place in row:
                if (self.spot == 'X' or self.spot == 'O'):
                    return False 
                else:
                    return True
        # end TODO 

    
def run():

    count = 0
    turn = GamePiece.CROSS
    
    start = input("Do you want to play Tic-Tac-Toe? (y/n)")
    if start.lower() == "y":
        board = GameBoard()
        board.display()
        
        while count < 9:

            print(f"It is {turn} turn. Which spot you want to pick?")
            spot = input()

            if board.is_valid(spot):
                board.place_into(turn, spot)

                #check if there is a winner, if yes, announce who is the winner
                if board.has_winner(): 
                    print(turn) 
                    exit()

                
                # and close the game, otherwise set the turn to the other player
                
                # TODO

                # end TODO
                else:
                    count = count + 1
                    if turn == GamePiece.CROSS:
                        turn = GamePiece.CIRCLE
                    elif turn == GamePiece.CIRCLE:
                        turn = GamePiece.CROSS
                    
            else:
                print("Invalid spot. Please try again")

        #TODO announce it is a tie game

        #end TODO

if __name__ == "__main__":
    run()
