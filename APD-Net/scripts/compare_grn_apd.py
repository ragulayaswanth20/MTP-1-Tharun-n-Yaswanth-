"""Fair HC18 synthetic-speckle comparison between GRN and APD-Net."""
from __future__ import annotations
import argparse,csv,json,random,sys
from pathlib import Path
import numpy as np,torch,torch.nn as nn,yaml
from PIL import Image
from torch.utils.data import DataLoader,Dataset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'scripts'))
from datasets.hc18_dataset import HC18Dataset,read_split
from models.apdnet import APDNet
from noise.speckle import apply_speckle
from run_grn_denoising import UNetGenerator,PatchDiscriminator,SimpleSegmenter
from skimage.metrics import structural_similarity

class FairDataset(Dataset):
    def __init__(self,records,size,sigma,seed): self.records=records;self.size=size;self.sigma=sigma;self.seed=seed
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]; image=np.asarray(Image.open(r.image_path).convert('L').resize((self.size,self.size),Image.Resampling.BILINEAR),dtype=np.float32)/255
        mask=np.asarray(Image.open(r.mask_path).convert('L').resize((self.size,self.size),Image.Resampling.NEAREST),dtype=np.float32)>0
        generator=torch.Generator().manual_seed(self.seed+i)
        noisy=(torch.from_numpy(image.copy())[None]*(1+self.sigma*torch.randn((1,self.size,self.size),generator=generator))).clamp(0,1)
        return torch.from_numpy(image.copy())[None],noisy,torch.from_numpy(mask.astype(np.int64)),r.filename

def metric(pred,target):
    mse=np.mean((pred-target)**2);gx,gy=np.gradient(pred);tx,ty=np.gradient(target)
    return {'psnr':float(10*np.log10(1/max(mse,1e-12))),'ssim':float(structural_similarity(target,pred,data_range=1)),'mae':float(np.mean(np.abs(pred-target))),'gradient_error':float(np.mean(np.abs(gx-tx)+np.abs(gy-ty)))}

def train_grn(loader,device,epochs):
    g=UNetGenerator().to(device);d=PatchDiscriminator().to(device);s=SimpleSegmenter(2).to(device);og=torch.optim.Adam(g.parameters(),2e-4,betas=(.5,.999));od=torch.optim.Adam(d.parameters(),2e-4,betas=(.5,.999));os=torch.optim.Adam(s.parameters(),2e-4,betas=(.5,.999));adv=nn.MSELoss();pix=nn.L1Loss();seg=nn.CrossEntropyLoss()
    for epoch in range(epochs):
        g.train();d.train();s.train()
        for clean,noisy,mask,_ in loader:
            clean,noisy,mask=clean.to(device),noisy.to(device),mask.to(device)
            clean_norm, noisy_norm = clean * 2.0 - 1.0, noisy * 2.0 - 1.0
            fake=g(noisy_norm).detach();ld=.5*(adv(d(clean_norm),torch.ones_like(d(clean_norm)))+adv(d(fake),torch.zeros_like(d(fake))));od.zero_grad();ld.backward();od.step()
            den=g(noisy_norm);lg=adv(d(den),torch.ones_like(d(den)))+100*pix(den,clean_norm)+50*seg(s(den),mask);og.zero_grad();lg.backward();og.step()
            with torch.no_grad(): den_for_s=g(noisy_norm)
            ls=seg(s(den_for_s),mask);os.zero_grad();ls.backward();os.step()
        print(f'GRN epoch {epoch+1}/{epochs}')
    return g.eval()

def main():
    p=argparse.ArgumentParser();p.add_argument('--apd-config',type=Path,default=ROOT/'configs/apdnet.yaml');p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--sigma',type=float,default=.2);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');cfg=yaml.safe_load(a.apd_config.read_text());train_records=read_split(cfg['train_split']);val_records=read_split(cfg['val_split']);train_loader=DataLoader(FairDataset(train_records,256,a.sigma,a.seed),batch_size=a.batch_size,shuffle=True);val_loader=DataLoader(FairDataset(val_records,256,a.sigma,a.seed),batch_size=a.batch_size)
    grn=train_grn(train_loader,device,a.epochs)
    torch.save({'model':grn.state_dict(),'epochs':a.epochs,'seed':a.seed,'sigma':a.sigma},ROOT/'checkpoints/grn_comparison.pt')
    apd=APDNet(cfg['base_channels']).to(device);apd.load_state_dict(torch.load(ROOT/cfg['checkpoint'],map_location=device)['model']);apd.eval();rows={'GRN':[],'APD-Net':[],'Noisy':[]}
    with torch.no_grad():
        for clean,noisy,_,_ in val_loader:
            clean,noisy=clean.to(device),noisy.to(device);grn_out=((grn(noisy*2.0-1.0).clamp(-1,1)*.5+.5).clamp(0,1));apd_out=apd(noisy)['image']
            for name,out in [('GRN',grn_out),('APD-Net',apd_out),('Noisy',noisy)]:
                for pred,target in zip(out[:,0].cpu().numpy(),clean[:,0].cpu().numpy()):rows[name].append(metric(pred,target))
    summary=[{'method':name,'images':len(values),**{k:float(np.mean([v[k] for v in values])) for k in values[0]}} for name,values in rows.items()];out=ROOT/'results';out.mkdir(exist_ok=True);(out/'grn_vs_apd.json').write_text(json.dumps({'dataset':'HC18','split':'fixed 799/200','sigma':a.sigma,'seed':a.seed,'grn_epochs':a.epochs,'results':summary},indent=2));
    with (out/'grn_vs_apd.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
