import { useState } from "react";
import { Button, createTheme, ThemeProvider, Tooltip } from "@mui/material";
import { IControlMenu } from "./ControlMenu";

const theme = createTheme({
    palette: {
        primary: {
            main: "#b5decc",
            contrastText: "#34454c"
        },
        secondary: {
            main: "#ae5224",
            contrastText: "#eeb480"
        }
    }
});

function ResetButton(prop: IControlMenu ) {
    const [confirm, setConfirm] = useState<boolean>(false);

    if (!confirm) {
        return (
            <Tooltip 
                    title="Reset the neural network. This will discard any training performed!"
                    placement="right"
            >
                <Button 
                    color="error"
                    variant="contained" 
                    onClick={() => {
                        setConfirm(true);
                    }}
                    size="small"
                >
                    Reset
                </Button>
            </Tooltip>
        )
    } else {
        return (
            <div style={{display: "flex", columnGap: "1rem"}}>
                <ThemeProvider theme={theme}>
                    <Button size="small" onClick={() => {setConfirm(false);}}>Cancel</Button>
                </ThemeProvider>
                
                    <Button 
                        color="error"
                        variant="contained" 
                        onClick={async () => {
                            setConfirm(false);
                            try {
                                await fetch(import.meta.env.VITE_API_URL + '/reset');
                                
                                prop.setCertainty(NaN);
                                prop.setPred(NaN);
                                prop.setActual(NaN);
                                prop.setActivations(Array(10).fill(0.0));
                                prop.setImgSrc("./placeholder.png");

                                if (!isNaN(prop.currRate)) {
                                    prop.setPrevRate(prop.currRate);
                                    prop.setRate(NaN);
                                }
                            } catch (error) {
                                console.log(error);
                            }
                        }}
                        size="small"
                    >
                        Confirm
                    </Button>
            </div>
    )}
}

export default ResetButton;