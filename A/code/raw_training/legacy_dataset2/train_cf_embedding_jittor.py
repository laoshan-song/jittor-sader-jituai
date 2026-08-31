"""全量边 sampled-softmax 训练 user/item embedding (dataset2冷启动).
在服务器GPU跑. 评估 user·item dot 在近门/远门 MRR, 验证是否值得.
"""
import os, sys, time
os.environ.setdefault('use_mpi','0'); os.environ.setdefault('log_silent','1')
if os.environ.get('JT_USE_CUDA')!='1': os.environ['nvcc_path']=''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jittor as jt
from jittor import nn
if os.environ.get('JT_USE_CUDA')=='1' and jt.has_cuda:
    jt.flags.use_cuda=1

import cf_ranker_jittor as run
run.DATA = os.environ.get('DATA_PATH','data_A.zip')

DIM = int(os.environ.get('EMB_DIM','128'))
EPOCHS = int(os.environ.get('EPOCHS','15'))
NEG = int(os.environ.get('NEG','20'))
BATCH = int(os.environ.get('BATCH','8192'))
MODE = os.environ.get('GATE','near')  # near/far, 决定用哪段做历史/验证
LR = float(os.environ.get('LR','0.01'))
SEED = int(os.environ.get('SEED','20260711'))

# This component was originally invoked as an ad-hoc GPU script.  Fix both
# random sources so the retained raw rebuild has an explicit initialization.
np.random.seed(SEED)
jt.set_global_seed(SEED)

def mrr_fair(s,y):
    r=[]
    for i in range(len(y)):
        p=s[i,y[i]]; r.append(1.0+(s[i]>p).sum()+((s[i]==p).sum()-1)/2.0)
    return float(np.mean(1.0/np.array(r)))

scene=os.environ.get('SCENE','dataset2')
train,test=run.read_scene(scene); max_node=run.node_max(train,test)
pool,freq,src_freq=run.test_pool_freq(test,max_node)
hist,tr_pos,va_pos=run.split_edges(scene,train)

if MODE=='near':
    train_edges=np.vstack([hist,tr_pos]); vp=va_pos
elif MODE=='full':
    train_edges=np.vstack([hist,tr_pos,va_pos]); vp=va_pos
else:
    train_edges=hist; vp=np.vstack([tr_pos,va_pos])

n=max_node+1
print(f"emb_cf: dim={DIM} epochs={EPOCHS} neg={NEG} mode={MODE} seed={SEED} edges={len(train_edges)} nodes={n}", flush=True)

# 负采样分布: 用测试候选池频率(匹配线上分布). freq已是log1p, 用原始bincount
cand=test.iloc[:,2:].to_numpy(np.int64).ravel()
negpool=np.bincount(cand,minlength=n).astype(np.float64)
negpool=negpool/negpool.sum()
neg_items=np.where(negpool>0)[0]
neg_probs=negpool[neg_items]; neg_probs/=neg_probs.sum()

src_all=train_edges[:,0].astype(np.int64)
dst_all=train_edges[:,1].astype(np.int64)

user_emb=jt.array((np.random.randn(n,DIM)*0.05).astype(np.float32))
item_emb=jt.array((np.random.randn(n,DIM)*0.05).astype(np.float32))
item_bias=jt.array(np.zeros(n,np.float32))
user_emb.requires_grad=True; item_emb.requires_grad=True; item_bias.requires_grad=True
opt=nn.Adam([user_emb,item_emb,item_bias],lr=LR,weight_decay=1e-6)

# 验证集
vsrc,vtim,vcand,vy=run.sample_groups(vp,pool,30000,12345)

def eval_dot():
    ue=user_emb.detach().numpy(); ie=item_emb.detach().numpy(); ib=item_bias.detach().numpy()
    sc=(ue[vsrc,None,:]*ie[vcand]).sum(axis=2)+ib[vcand]
    return mrr_fair(sc,vy)

rng=np.random.default_rng(SEED)
nE=len(src_all)
OUT=os.environ.get('OUT','/data1/songwentao/track1_claude/cf_emb.npz')
best_mrr=-1.0
for ep in range(EPOCHS):
    perm=rng.permutation(nE)
    tot=0.0; nb=0
    t0=time.time()
    for s in range(0,nE,BATCH):
        idx=perm[s:s+BATCH]
        u=src_all[idx]; v=dst_all[idx]
        negs=rng.choice(neg_items,size=(len(idx),NEG),p=neg_probs).astype(np.int64)
        ub=user_emb[jt.array(u)]           # (B,D)
        vb=item_emb[jt.array(v)]           # (B,D)
        vbias=item_bias[jt.array(v)]       # (B,)
        nb_e=item_emb[jt.array(negs.reshape(-1))].reshape((len(idx),NEG,DIM))
        nbias=item_bias[jt.array(negs.reshape(-1))].reshape((len(idx),NEG))
        pos=(ub*vb).sum(1)+vbias           # (B,)
        neg=(ub.unsqueeze(1)*nb_e).sum(2)+nbias  # (B,NEG)
        # sampled softmax: -log( exp(pos)/(exp(pos)+sum exp(neg)) )
        logits=jt.concat([pos.unsqueeze(1),neg],dim=1)  # (B,1+NEG)
        loss=nn.cross_entropy_loss(logits,jt.zeros(len(idx),dtype='int32'))
        opt.step(loss)
        tot+=float(loss.data); nb+=1
    mrr=eval_dot()
    star=""
    if mrr>best_mrr:
        best_mrr=mrr; star="*"
        np.savez(OUT, user=user_emb.detach().numpy(), item=item_emb.detach().numpy(), ibias=item_bias.detach().numpy())
    print(f"ep={ep+1} loss={tot/nb:.4f} dot_MRR={mrr:.5f}{star} ({time.time()-t0:.1f}s)",flush=True)
print(f"saved best cf_emb dot_MRR={best_mrr:.5f}", flush=True)
