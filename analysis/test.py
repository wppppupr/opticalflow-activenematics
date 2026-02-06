import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import argparse
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
from pathlib import Path
import tifffile

import torch.nn.functional as F

def get_velocity(image1, image2, args, model):
    # numpy -> tensor
    if image1.ndim == 2:
        image1 = np.stack([image1] * 3, axis=-1)
        image2 = np.stack([image2] * 3, axis=-1)

    img1 = transforms.ToTensor()(image1).unsqueeze(0).to(torch.device('cuda'))
    img2 = transforms.ToTensor()(image2).unsqueeze(0).to(torch.device('cuda'))
    img1, img2 = img1 * 255.0, img2 * 255.0

    # --- メモリ対策: リサイズ ---
    # 27GB要求を10GB以下にするため、一旦面積を半分以下（0.5倍など）にする
    scale = 0.5 
    h, w = img1.shape[2], img1.shape[3]
    img1_s = F.interpolate(img1, scale_factor=scale, mode='bilinear', align_corners=True)
    img2_s = F.interpolate(img2, scale_factor=scale, mode='bilinear', align_corners=True)

    padder = InputPadder(img1_s.shape, 8)
    img1_s, img2_s = padder.pad(img1_s, img2_s)

    with torch.no_grad():
        # autocastを使用して混合精度を有効化
        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            _, flow_low = model(img1_s, img2_s, iters=args.iter, test_mode=True)
    
    # flowを元の解像度にアップサンプリングして戻す
    flow_up = F.interpolate(flow_low, size=(h, w), mode='bilinear', align_corners=True)
    
    # 座標系の補正（重要）
    flow_up[:, 0, :, :] *= (1.0 / scale)
    flow_up[:, 1, :, :] *= (1.0 / scale)

    flow_fw = flow_up.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return flow_fw

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='./models/weights.pth')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--save_location', help="save the results in local or oss")
    parser.add_argument('--save_path', help=" local path to save the result", default='./example/velocity_plot.png')
    parser.add_argument('--iter', type=int, default=24)
    args = parser.parse_args()
    args.mixed_precision = True

    # データロード (memmap)
    base_path = Path('/media/sasaki/myssd/Sasaki/MTSingleBeads')
    folder = base_path / '20260122/exp001'
    tiff_path = folder / 'MTs.tif'

    if not tiff_path.exists():
        raise FileNotFoundError(f"Tiff file not found: {tiff_path}")

    with tifffile.TiffFile(tiff_path) as tif:
        images = tif.asarray(out='memmap')
        print(f"Total frames: {images.shape[0]}, Shape: {images.shape}")

    # モデルのセットアップ
    torch.cuda.empty_cache()    
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))    
    model.cuda()
    model.eval()
    
    # 解析 (最初の2フレーム)
    img_frame0 = images[0]
    img_frame1 = images[5]
    
    flows = get_velocity(img_frame0, img_frame1, args, model)
    vx, vy = flows[:,:,0], flows[:,:,1]

    # 可視化
    fig, ax = plt.subplots(figsize=(8, 8))
    # img_frame0 がグレースケールの場合は cmap="gray"
    ax.imshow(img_frame0, cmap="gray")
    ax.set_title("Optical Flow Result")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # ベクトル描画の設定
    n = 20  
    h, w = vx.shape
    x, y = np.meshgrid(np.arange(0, w, n), np.arange(0, h, n))
    ax.quiver(x, y, vx[::n, ::n], vy[::n, ::n], 
              color="red", scale_units="xy", angles="xy", scale=0.5, width=0.002)
    
    plt.tight_layout()
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    print(f"Result saved to {save_path}")