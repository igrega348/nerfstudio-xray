from typing import Optional
from pathlib import Path
import torch
import tyro

def main(load_checkpoint: Path, out_fn: Optional[Path] = None):
    assert load_checkpoint.exists()
    data = torch.load(load_checkpoint)
    data['pipeline'].pop('_model.camera_optimizer.pose_adjustment')

    if out_fn is None:
        out_fn = load_checkpoint.with_name(load_checkpoint.stem+'-mod.ckpt')
    
    print(f'Saving modified checkpoint to {out_fn}')
    torch.save(data, out_fn)
    
if __name__ == '__main__':
    tyro.cli(main)