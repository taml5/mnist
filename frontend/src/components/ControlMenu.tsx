import { Dispatch, SetStateAction, useState } from "react";
import { Button, Slider } from "@mui/material";
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

function ControlMenu(prop: IControlMenu) {
    const [training, setTraining] = useState<boolean>(false);
    const [evaluating, setEvaluating] = useState<boolean>(false);

    const [trainingText, setTrainingText] = useState<string>("Train");
    const [epochs, setEpochs] = useState<number>(1);
    const [batchSize, setBatchSize] = useState<number>(5);
    const [learningRate, setLearningRate] = useState<number>(3 * 20);

    return (<div className="control-menu">
        <div className="buttons">
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
                    setTrainingText("Training...")
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
                    setTrainingText("Train")
                }}
            >
                {trainingText}
            </Button>
        </div>
        <div className="sliders">
            <div className="slider">
                <p>Epochs</p>
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
                <p>Batch Size</p>
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
                <p>Learning Rate</p>
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