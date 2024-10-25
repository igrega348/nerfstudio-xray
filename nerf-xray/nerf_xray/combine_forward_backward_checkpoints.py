from typing import Optional
from pathlib import Path
import torch
import tyro

def main(
    fwd_ckpt: Path, 
    bwd_ckpt: Path, 
    out_fn: Path
):
    assert fwd_ckpt.exists() and bwd_ckpt.exists()
    combined_state_dict = {'pipeline':{}}
    for direction, ckpt in zip(['f', 'b'], [fwd_ckpt, bwd_ckpt]):
        data = torch.load(ckpt)
        for key, val in data.items():
            if key=='step':
                if key not in combined_state_dict:
                    combined_state_dict[key] = val
            elif key=='pipeline':
                for kk in val.keys():
                    itms = kk.split('.')
                    if itms[1] not in ['field', 'deformation_field']:
                        combined_state_dict['pipeline'][kk] = val[kk]
                    else:
                        itms[1] = itms[1] + '_' + direction
                        k = '.'.join(itms)
                        combined_state_dict['pipeline'][k] = val[kk]
                
    
    print(f'Saving modified checkpoint to {out_fn}')
    torch.save(combined_state_dict, out_fn)
    
if __name__ == '__main__':
    tyro.cli(main)