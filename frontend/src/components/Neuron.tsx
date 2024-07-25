import { Tooltip } from '@mui/material';
import './Neuron.css';

interface INeuron {
    digit: number,
    activation: number
}

function Neuron(prop: INeuron) {
    const colour = `rgb(${255 * prop.activation}, ${255 * prop.activation}, ${255 * prop.activation})`

    return (
    <Tooltip title={`${prop.digit}: ${Math.round(prop.activation * 100) / 100}`} placement="top" arrow>
        <span className="neuron" style={{backgroundColor: colour}}/>
    </Tooltip>
    )
}

export default Neuron