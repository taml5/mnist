import { Slider } from "@mui/material";

interface ISlider {
    label: string
    discrete: boolean
}

function SliderSetting(prop: ISlider) {
    const slider = prop.discrete ? <Slider discrete/> : <Slider/>

    return (<div className="slider">
        <p>{prop.label}</p>
        {slider}
    </div>
    )
}

export default SliderSetting;