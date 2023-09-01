import time


def punish(board):
    """
    Sends serial signal to arduino to open relay

    :params:
        - board: pyfirmata object, represents arduino board
    """
    board.digital[7].write(1)
    time.sleep(1)
    board.digital[7].write(0)