import soundfile
import network_arch
import torch
import torch.nn.functional as Fx
import numpy as np
import librosa

from matplotlib import pyplot as plt
from scipy.signal import find_peaks

import seaborn #Only needed for plot

model_path = 'model/best_gated_emb.pkl'

def extract_embeddings(audio_file, trg_sr=44100, output_mel=False):
    emb_net = network_arch.EmbNet()
    emb_net.load_weight(model_path)
    emb_net.eval()
    
    y, src_sr = soundfile.read(audio_file)
    if len(y.shape) > 1:
        y = y[:,0]
        
    if src_sr != trg_sr:
        y = librosa.resample(y,src_sr,trg_sr)

    mel = librosa.feature.melspectrogram(y,trg_sr,n_fft=int(trg_sr*0.04),hop_length=int(trg_sr*0.02),n_mels=128)
    log_mel = librosa.power_to_db(mel).T
    
    mel_seq = torch.from_numpy(log_mel).float()
    with torch.no_grad():
        emb_seq = emb_net(mel_seq.unsqueeze(0).unsqueeze(1)).squeeze(0)

    if output_mel:
        return emb_seq.numpy(), log_mel
    else:
        return emb_seq.numpy()
    
def detect_change_point(seq, likelihood=False):
    seq = torch.from_numpy(seq)
    emb_m = Fx.conv2d(seq.unsqueeze(0).unsqueeze(1), torch.ones((25,1),dtype=torch.float).unsqueeze(0).unsqueeze(1)/25).squeeze()

    sim_mat = torch.mm(emb_m, emb_m.transpose(0,1))/torch.ger(emb_m.norm(dim=1), emb_m.norm(dim=1))
    dist_mat = 1 - sim_mat

    dist_vec = torch.zeros(len(emb_m) - 25)
    for i in range(len(emb_m) - 25):
        dist_vec[i] = dist_mat[i, i+25]
    

    peaks, _ = find_peaks(dist_vec, height=0.1)
    print (peaks)
    change_points = (peaks + 25).tolist()

    if likelihood:
        return change_points, dist_vec
    else:
        return change_points


    
def plot(audio_file):
    emb_seq, mel_seq = extract_embeddings(audio_file, output_mel=True)

    change_points, likelihood = detect_change_point(emb_seq, likelihood=True)
    
    plt.figure(0)
    #plt.subplot(311)
    ax1 = plt.subplot(3, 1, 1)

    seaborn.heatmap(mel_seq.T, ax=ax1, cbar=False)
    ax1.invert_yaxis()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_ticks([])
    ax1.set_ylabel('Mel-spectrum')

    for point in change_points:
        plt.axvline(x=point, color='w')

    ax2 = plt.subplot(3, 1, 2)
    seaborn.heatmap(emb_seq.T, ax=ax2, cbar=False)
    ax2.invert_yaxis()
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_ticks([])
    #ax2.spines['left'].set_visible(False)    
    #ax2.get_yaxis().set_visible(False)
    ax2.set_ylabel('Embeddings')

    for point in change_points:
        plt.axvline(x=point, color='w')


    ax3 = plt.subplot(3, 1, 3)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    dist_vec = likelihood.numpy()
    ax3.get_xaxis().set_visible(False)
    ax3.set_ylabel('Likelihood of change')
    
    plt.plot(np.arange(len(likelihood))+25, likelihood, 'b')
    plt.plot(change_points, likelihood[np.array(change_points)-25], "xr")
    
    print (change_points)
    
    
    plt.savefig("plot.png")
    
    
if __name__ == '__main__':
    import sys
    plot(sys.argv[1])
