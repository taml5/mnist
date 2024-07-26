import './Infoboard.css';

interface IStats {
    pred: number,
    certainty: number,
    actual: number,
    rate: number,
    prevRate: number
}


function Infoboard(prop: IStats) {
    return (
        <div className="infoboard">
            <p>Predicted Digit: {!isNaN(prop.pred) && <span>{prop.pred} {!isNaN(prop.certainty) && <>({Math.round(prop.certainty * 100) / 100} certainty)</>}</span>}</p>
            <p>Actual Digit: {!isNaN(prop.actual) && <span>{prop.actual}</span>}</p>
            <p>Rate: {!isNaN(prop.rate) && <span>{prop.rate} of 10000 ({Math.round(prop.rate) / 100}%)</span>} </p>
            <p>Previous Rate: {!isNaN(prop.prevRate) && <span>{prop.prevRate} of 10000 ({Math.round(prop.prevRate) / 100}%)</span>} </p>
        </div>
    )
}

export default Infoboard