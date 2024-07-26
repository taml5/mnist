import { Dispatch, SetStateAction, useState } from "react";
import { Button, Slider, Tooltip } from "@mui/material";
import "./ControlMenu.css";

interface IControlMenu {
    setCertainty: Dispatch<SetStateAction<number>>,
    setPred: Dispatch<SetStateAction<number>>,
    setActual: Dispatch<SetStateAction<number>>,
    setRate: Dispatch<SetStateAction<number>>,
    currRate: number,
    setPrevRate: Dispatch<SetStateAction<number>>,
    setActivations: Dispatch<SetStateAction<Array<number>>>,
    setImgSrc: Dispatch<SetStateAction<string>>
}

// TODO: update the colour scheme
function ControlMenu(prop: IControlMenu) {
    const [training, setTraining] = useState<boolean>(false);
    const [evaluating, setEvaluating] = useState<boolean>(false);
    const [epochs, setEpochs] = useState<number>(1);
    const [batchSize, setBatchSize] = useState<number>(5);
    const [learningRate, setLearningRate] = useState<number>(3 * 20);

    return (<div className="control-menu">
        <div className="buttons">
            <Tooltip 
                title="Evaluate the number of successful predictions over the entire testing image set."
                placement="left"
                enterDelay={1000}
            >
                <Button 
                    className="Evaluate" 
                    variant="contained" 
                    size="large"
                    disabled={evaluating}
                    sx= {{
                        width: 200
                    }}
                    onClick={async () => {
                        setEvaluating(true);
                        try {
                            const response = await fetch(import.meta.env.VITE_API_URL + '/evaluate');
                            if (response) {
                                const data = await response.json();
                                prop.setPrevRate(prop.currRate);
                                prop.setRate(data.rate);
                            }
                        } catch (error) {
                            console.log(error);
                        }
                        setEvaluating(false);
                    }}
                >
                    Evaluate
                </Button>
            </Tooltip>
            <Tooltip 
                title="Test the neural network on a single random input picture."
                enterDelay={1000}
            >
                <Button 
                    className="Test" 
                    variant="contained" 
                    size="large"
                    sx= {{
                        width: 200
                    }}
                    onClick={async () => {
                        try {
                            const response = await fetch(import.meta.env.VITE_API_URL + '/test');
                            if (response) {
                                const data = await response.json();
                                prop.setPred(data.guess);
                                prop.setActual(data.answer);
                                prop.setActivations(data.activations);
                                prop.setImgSrc(data.image);
                            }
                        } catch (error) {
                            console.log(error);
                        }
                    }}
                >
                    Test
                </Button>
            </Tooltip>
            <Tooltip 
                title="Train the neural network with the below hyperparameters. This could take a while."
                placement="right"
                enterDelay={1000}
            >
                <Button 
                    className="Train" 
                    variant="contained" 
                    size="large" 
                    disabled={training}
                    sx= {{
                        width: 200
                    }}
                    onClick={async () => {
                        setTraining(true);
                        try {
                            await fetch(import.meta.env.VITE_API_URL + '/train', {
                                method: "POST",
                                body: JSON.stringify({
                                    "epochs": epochs,
                                    "batch_size": 2 ** batchSize,
                                    "learning_rate": learningRate / 20
                                }),
                                headers: {
                                    "Content-type": "application/json; charset=UTF-8"
                                }
                            });
                        } catch (error) {
                            console.log(error);
                        }
                        setTraining(false);
                    }}
                >
                    {training ? "Training..." : "Train"}
                </Button>
            </Tooltip>
        </div>
        <div className="sliders">
            <div className="slider">
                <Tooltip title="How many times the neural network trains on the entire training dataset." placement="left">
                    <p>Epochs</p>
                </Tooltip>
                <Slider
                    className="mui-slider"
                    aria-label="Epochs"
                    valueLabelDisplay="auto"
                    value={epochs}
                    step={1}
                    min={1}
                    max={5}
                    marks
                    onChange={(_, value: number | number[]) => {
                        const epochs = Array.isArray(value) ? value[0] : value;
                        setEpochs(epochs);
                    }}
                />
            </div>
            <div className="slider">
                <Tooltip title="The number of samples that are tested before the weights are updated in the network." placement="left">
                    <p>Batch Size</p>
                </Tooltip>
                <Slider
                    className="mui-slider"
                    aria-label="Batch Size"
                    valueLabelDisplay="auto"
                    value={batchSize}
                    min={0}
                    max={10}
                    step={1}
                    marks
                    scale={(value) => (2 ** value)}
                    onChange={(_, value: number | number[]) => {
                        const size = Array.isArray(value) ? value[0] : value;
                        setBatchSize(size);
                    }}
                />
            </div>
            <div className="slider">
                <Tooltip title="How much the network changes in response to the error in training a batch." placement="left">
                    <p>Learning Rate</p>
                </Tooltip>
                <Slider
                    className="mui-slider"
                    aria-label="Learning Rate"
                    valueLabelDisplay="auto"
                    value={learningRate}
                    scale={(value) => (value / 20)}
                    onChange={(_, value: number | number[]) => {
                        const rate = Array.isArray(value) ? value[0] : value;
                        setLearningRate(rate);
                    }}
                />
            </div>
        </div>
    </div>)
}

export default ControlMenu;