"""The main file where the neural network is loaded and a CLI provided to interact with it."""
import network
import loader
import questionary

EPOCHS = 2
BATCH_SIZE = 100
LEARNING_RATE = 0.01

training_data, validation_data, test_data = loader.load_data("./data/mnist.pkl.gz")
net = network.Network([784, 16, 16, 10])
baseline = 0.10

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
            print(f"Training with epochs={EPOCHS}, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}")
            net.train(training_data, EPOCHS, BATCH_SIZE, LEARNING_RATE)
            print("Completed training.")
        elif answer == "Evaluate":
            success = net.evaluate(test_data)
            rate = success / len(test_data)
            print(f"Success rate of {success} of {len(test_data)} ({rate}).")
            print(f"Previous rate was {baseline}.")
            baseline = rate
        elif answer == "Exit":
            print("Exiting...")
            exit(0)
        else:
            print("ERROR: invalid choice")
            exit(1)
