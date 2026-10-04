"""Stage-1 APD-Net training on synthetic speckle/reference pairs."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np,torch,yaml
from torch.utils.data import DataLoader
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.hc18_dataset import HC18Dataset,read_split
from models.apdnet import APDNet
from losses.total_loss import APDLoss
from noise.speckle import apply_speckle

def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def run(model,loader,loss_fn,device,strength,train):
    model.train(train); values=[]
    for batch in loader:
        clean=batch['image'].to(device); noisy=apply_speckle(clean,strength)
        with torch.set_grad_enabled(train):
            out=model(noisy); loss,parts=loss_fn(out['image'],clean)
            if train: optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        values.append(float(loss.item()))
    return float(np.mean(values))
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/apdnet.yaml');a=p.parse_args();cfg=yaml.safe_load(a.config.read_text());seed(cfg['seed']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train=DataLoader(HC18Dataset(read_split(cfg['train_split']),cfg['image_size']),batch_size=cfg['batch_size'],shuffle=True);val=DataLoader(HC18Dataset(read_split(cfg['val_split']),cfg['image_size']),batch_size=cfg['batch_size'])
    model=APDNet(cfg['base_channels']).to(device); loss_fn=APDLoss(cfg['loss']); global optimizer;optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']);best=float('inf');history=[]
    print(f'APD-Net Stage 1: train={len(train.dataset)} val={len(val.dataset)} device={device}')
    for epoch in range(1,cfg['epochs']+1):
        tr=run(model,train,loss_fn,device,cfg['speckle_strength'],True);va=run(model,val,loss_fn,device,cfg['speckle_strength'],False);history.append({'epoch':epoch,'train_loss':tr,'val_loss':va});torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch},ROOT/cfg['checkpoint']);
        if va<best: best=va;torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch},ROOT/cfg['checkpoint'].replace('best','best'))
        print(f'epoch={epoch}/{cfg["epochs"]} train={tr:.5f} val={va:.5f}')
    out=ROOT/'results';out.mkdir(exist_ok=True);(out/'apd_stage1_metrics.json').write_text(json.dumps({'dataset':'HC18','split':'validation','synthetic_reference':True,'results':history},indent=2));
    with (out/'apd_stage1_metrics.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=history[0]);w.writeheader();w.writerows(history)
if __name__=='__main__':main()
