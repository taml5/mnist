"""The main file where the neural network is loaded."""
from flask import Flask, request, jsonify, Response

import network
import loader

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 3
HIDDEN_LAYER_SIZE = 16

training_data, validation_data, test_data = loader.load_data("./backend/data/mnist.pkl.gz")
net = network.Network([784, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, 10])
rate = net.evaluate(test_data)

app = Flask(__name__)

@app.route('/')
def init():
    return jsonify({"rate": rate})

@app.route('/evaluate')
def evaluate():
    rate = net.evaluate(test_data)
    return jsonify({"rate": rate})

@app.route('/train')
def train():
    data = request.form
    net.train(training_data=training_data,
              epochs=data["epochs"] if "epochs" in data else EPOCHS,
              batch_size=data["batch_size"] if "batch_size" in data else BATCH_SIZE,
              learning_rate=data["learning_rate"] if "learning_rate" in data else LEARNING_RATE,
              test_data=test_data if "test" in data and data["test"] == True else None)
    return Response(status=200)
    
@app.route('/test')
def test_one():
    guess, ans, image = net.test_one(test_data)
    return jsonify({
        "guess": guess.tolist(),
        "answer": str(ans),
        "image": image
    })

if __name__ == '__main__':
    app.run(debug=True)