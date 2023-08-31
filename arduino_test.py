import pyfirmata
import time


def board_test():
    board = pyfirmata.Arduino('COM3')

    while True:
        board.digital[7].write(1)
        time.sleep(1)
        board.digital[7].write(0)
        time.sleep(0.5)

if __name__ == "__main__":
    board_test()
