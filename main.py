"""The main file where the neural network is loaded and a CLI provided to interact with it."""
import network
import loader
import questionary

training_data, validation_data, test_data = loader.load_data("./data/mnist.pkl.gz")
net = network.Network([784, 16, 16, 10])

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
            print("Evaluate neural net")
        elif answer == "Exit":
            print("Exiting...")
            exit(0)
        else:
            print("ERROR: invalid choice")
            exit(1)
