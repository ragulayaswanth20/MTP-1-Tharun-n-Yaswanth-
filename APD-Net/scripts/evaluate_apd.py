"""Evaluate APD-Net on synthetic HC-18 validation corruption."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,torch,yaml
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.hc18_dataset import HC18Dataset,read_split
from models.apdnet import APDNet
from noise.speckle import apply_speckle

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/apdnet.yaml');a=p.parse_args();cfg=yaml.safe_load(a.config.read_text());device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=APDNet(cfg['base_channels']).to(device);ckpt=torch.load(ROOT/cfg['checkpoint'],map_location=device);model.load_state_dict(ckpt['model']);model.eval();loader=DataLoader(HC18Dataset(read_split(cfg['val_split']),cfg['image_size']),batch_size=cfg['batch_size']);rows={'Noisy':[],'APD-Net':[]}
 with torch.no_grad():
  for batch in loader:
   clean=batch['image'].to(device); noisy=apply_speckle(clean,cfg['speckle_strength']); den=model(noisy)['image']
   for name,pred in [('Noisy',noisy),('APD-Net',den)]:
    for x,y in zip(pred[:,0].cpu().numpy(),clean[:,0].cpu().numpy()):
     mse=np.mean((x-y)**2);gx,gy=np.gradient(x);tx,ty=np.gradient(y);rows[name].append({'psnr':10*np.log10(1/max(mse,1e-12)),'ssim':structural_similarity(y,x,data_range=1),'mae':np.mean(np.abs(x-y)),'gradient_error':np.mean(np.abs(gx-tx)+np.abs(gy-ty))})
 summary=[]
 for name,values in rows.items(): summary.append({'method':name,'images':len(values),**{k:float(np.mean([v[k] for v in values])) for k in values[0]}})
 out=ROOT/'results';out.mkdir(exist_ok=True);(out/'apd_metrics.json').write_text(json.dumps({'dataset':'HC18','split':'validation','synthetic_reference':True,'sigma':cfg['speckle_strength'],'results':summary},indent=2));
 with (out/'apd_metrics.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
