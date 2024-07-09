"""TODO: fill this docstring in"""
import numpy as np
import network
import loader

import questionary


net = network.Network(28 * 28, [16, 16, 10])

if __name__ == '__main__':
    while True:
        answer = questionary.select(
            "Choose an option: ",
            [
                "Train",
                "Evaluate",
                "Exit"
            ]).ask()
        if answer == "Train":
            print("Train neural net")
        elif answer == "Evaluate":
            # print("Evaluate neural net")
            print(net.evaluate(np.array([1] * 28 * 28)))
        elif answer == "Exit":
            print("Exiting...")
            exit(0)
        else:
            print("ERROR: invalid choice")
            exit(1)
