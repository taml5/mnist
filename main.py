"""The main file where the neural network is loaded and a CLI provided to interact with it."""
import network
import loader
import questionary

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 3
HIDDEN_LAYER_SIZE = 16

training_data, validation_data, test_data = loader.load_data("./data/mnist.pkl.gz")
net = network.Network([784, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, 10])

if __name__ == '__main__':
    print(f"Neural network with 784 inputs neurons, 2 hidden layers with "
          f"{HIDDEN_LAYER_SIZE} neurons, and 10 output neurons")
    print("Testing initial success rate...")
    rate = net.evaluate(test_data)
    print(f"Initial success rate: {rate} of {len(test_data)}.")
    while True:
        answer = questionary.select(
            "Choose an option: ",
            [
                "Train",
                "Evaluate",
                "Train & Evaluate",
                "Test",
                "Exit"
            ]).ask()
        if answer == "Train":
            print(f"Training with epochs=1, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}")
            net.train(training_data, 1, BATCH_SIZE, LEARNING_RATE)
            print("Completed training.")
        elif answer == "Evaluate":
            success = net.evaluate(test_data)
            print(f"Success rate: {success} of {len(test_data)}.")
            print(f"Previous rate was {rate} of {len(test_data)}.")
            rate = success
        elif answer == "Train & Evaluate":
            print(f"Initial success: {rate} of {len(test_data)}.")
            print(f"Training with epochs={EPOCHS}, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}")
            net.train(training_data, EPOCHS, BATCH_SIZE, LEARNING_RATE, test_data)
            print("Completed training.")
            rate = net.evaluate(test_data)
        elif answer == "Test":
            print(f"Selecting random image...")
            guess, ans = net.test_one(test_data)
            print(f"Network sees {guess}. Correct answer was {ans}.")
        elif answer == "Exit":
            print("Exiting...")
            exit(0)
        else:
            print("ERROR: invalid choice")
            exit(1)
