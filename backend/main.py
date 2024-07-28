"""The main file where the neural network is loaded."""
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import network
import loader

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 3
HIDDEN_LAYER_SIZE = 16

training_data, validation_data, test_data = loader.load_data("./backend/data/mnist.pkl.gz")
net = network.Network([784, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, 10])

app = Flask(__name__)
CORS(app)

@app.route('/evaluate', methods = ['GET'])
def evaluate():
    return jsonify({"rate": net.evaluate(test_data)})

@app.route('/train', methods = ['POST'])
def train():
    data = request.get_json()
    net.train(training_data=training_data,
              epochs=data["epochs"] if "epochs" in data else EPOCHS,
              batch_size=data["batch_size"] if "batch_size" in data else BATCH_SIZE,
              learning_rate=data["learning_rate"] if "learning_rate" in data else LEARNING_RATE,
              test_data=None)
    return Response(status=200)
    
@app.route('/test', methods = ['GET'])
def test_one():
    guess, ans, image = net.test_one(test_data)
    return jsonify({
        "activations": guess.tolist(),
        "guess": max(range(len(guess)), key=guess.__getitem__),
        "answer": str(ans),
        "image": image
    })

@app.route('/reset', methods = ['POST'])
def reset():
    net.__init__([784, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, 10])
    return Response(status=200)

if __name__ == '__main__':
    app.run()