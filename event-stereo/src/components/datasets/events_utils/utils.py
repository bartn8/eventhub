import numpy as np


#TODO: restore rendering functionality
def render(img: np.ndarray, x: np.ndarray, y: np.ndarray, pol: np.ndarray, t: np.ndarray, color_neg = (255,0,0,255), color_pos = (0,0,255,255), scale_factor: int = 1) -> np.ndarray:
    
    #1: Vanilla negative polarity Blue #0000ff
    #2: Vanilla positive polarity Red #ff0000
    #3: Fictitious negative polarity magenta #ff00ff
    #4: Fictitious positive polarity Lime #00ff00

    #Use BGRA instead

    assert x.size == y.size == pol.size == t.size
    N_BINS,H,W = img.shape[:3]
    H,W = H*scale_factor,W*scale_factor
    
    #img = np.full((H,W,3), fill_value=255,dtype='uint8')
    min_t, max_t = t[0], t[-1]
    range_t = (max_t-min_t)

    ts_prev = 0

    for b in range(N_BINS):
        ts = ((b+1)/N_BINS) * range_t + min_t

        mask = np.zeros((H//scale_factor,W//scale_factor),dtype='int32')
        pol = pol.astype('int')#implicit copy
        pol[pol==0]=-1

        mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)&(ts_prev<=t)&(t<=ts)
        mask[np.floor(y[mask1]//scale_factor).astype(np.int32),np.floor(x[mask1]//scale_factor).astype(np.int32)]=pol[mask1]

        img[b][(mask==-1)] = color_neg 
        img[b][(mask== 1)] = color_pos

        ts_prev = ts

    return img
