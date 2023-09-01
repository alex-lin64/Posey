def punish():
    """
    Sends serial signal to arduino to open relay
    """
    BOARD.digital[7].write(1)
    time.sleep(1)
    BOARD.digital[7].write(0)