import { useState } from 'react';
import Infoboard from './components/Infoboard';
import Neuron from './components/Neuron';
import ControlMenu from './components/ControlMenu';
import './App.css';

function App() {
  const [certainty, setCertainty] = useState<number>(NaN);
  const [pred, setPred] = useState<number>(NaN);
  const [actual, setActual] = useState<number>(NaN);
  const [rate, setRate] = useState<number>(NaN);
  const [prevRate, setPrevRate] = useState<number>(NaN);
  const [activations, setActivations] = useState<Array<number>>(Array(10).fill(0.0));
  const [imgSrc, setImgSrc] = useState<string>("./placeholder.png");

  return (
    <>
      <h1>MNIST Neural Network</h1>
      <div className="port">
        <img src={imgSrc === "./placeholder.png" ? imgSrc : "data:image/png;base64," + imgSrc}></img>
        <Infoboard certainty={certainty} pred={pred} actual={actual} rate={rate} prevRate={prevRate}/>
      </div>
      <div className="neurons">
        {activations.map((activation, i) => <Neuron digit={i} activation={activation} key={i}/>)}
      </div>
      <ControlMenu
        setCertainty={setCertainty}
        setPred={setPred}
        setActual={setActual}
        setRate={setRate}
        currRate={rate}
        setPrevRate={setPrevRate}
        setActivations={setActivations}
        setImgSrc={setImgSrc}
      />
    </>
  )
}

export default App
