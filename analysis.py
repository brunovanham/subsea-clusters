# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

SEED = 42

# company, revenue_usdbn, pct_subsea, sps, surf_inst, flex, cables, services, wells, fleet, brazil
data = [
    ("TechnipFMC",9.0,85,1,1,1,1,0,0,16,2),("Subsea7",6.8,100,0,1,0,0,1,0,36,2),
    ("Saipem",13.0,50,0,1,0,0,1,1,20,1),("SLB (OneSubsea)",36.0,10,1,0,0,0,1,1,0,2),
    ("Baker Hughes",27.5,15,1,0,1,0,0,1,0,2),("Aker Solutions",4.5,60,1,0,0,1,0,0,0,1),
    ("Oceaneering",2.7,70,0,0,0,1,1,0,5,2),("McDermott",3.0,40,0,1,0,0,0,0,10,1),
    ("NOV",8.8,20,1,0,1,0,0,1,0,1),("Halliburton",23.0,5,1,0,0,0,0,1,0,1),
    ("Prysmian",18.0,15,0,0,0,1,0,0,4,0),("Nexans",8.0,20,0,0,0,1,0,0,3,0),
    ("DOF Group",2.0,100,0,1,0,0,1,0,50,2),("Helix Energy",1.3,100,0,0,0,0,1,1,10,1),
    ("Innovex (Dril-Quip)",1.0,60,1,0,0,0,0,1,0,1),
]
cols = ["company","revenue_usdbn","pct_subsea","sps","surf_inst","flex","cables","services","wells","fleet","brazil"]
df = pd.DataFrame(data, columns=cols)
df["subsea_revenue_est"] = df["revenue_usdbn"]*df["pct_subsea"]/100
df["n_segments"] = df[["sps","surf_inst","flex","cables","services","wells"]].sum(axis=1)
features = ["subsea_revenue_est","pct_subsea","n_segments","fleet","brazil","sps","surf_inst","cables","services"]
X = StandardScaler().fit_transform(df[features].values)

# choose k
ks = range(2,8); inertia=[]; sil=[]
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
K = list(ks)[int(np.argmax(sil))]
fig, ax = plt.subplots(1,2, figsize=(11,4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
ax[1].plot(list(ks), sil, "o-", color="green"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig("images/01_k_selection.png", dpi=150, bbox_inches="tight"); plt.close()

# kmeans + pca
km = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(X)
df["cluster"] = km.labels_
pca = PCA(2); XY = pca.fit_transform(X); df["pc1"], df["pc2"] = XY[:,0], XY[:,1]
plt.figure(figsize=(11,7))
colors = plt.cm.tab10(np.linspace(0,1,K))
for c in range(K):
    s = df[df.cluster==c]
    plt.scatter(s.pc1, s.pc2, s=s.subsea_revenue_est*40+80, color=colors[c], alpha=.75, label=f"Cluster {c}")
for _, r in df.iterrows():
    plt.annotate(r.company, (r.pc1, r.pc2), fontsize=9, xytext=(6,4), textcoords="offset points")
v = pca.explained_variance_ratio_
plt.xlabel(f"PC1 ({v[0]:.0%})"); plt.ylabel(f"PC2 ({v[1]:.0%})")
plt.title(f"Top 15 subsea: {K} strategic groups (bubble = est. subsea revenue)")
plt.legend(); plt.grid(alpha=.3)
plt.savefig("images/02_clusters_pca.png", dpi=150, bbox_inches="tight"); plt.close()

# ward dendrogram (colored by cluster, annotated)
Z = linkage(X, method="ward")
plt.figure(figsize=(12,5.5))
dn = dendrogram(Z, labels=df.company.values, leaf_rotation=75, leaf_font_size=10,
                color_threshold=0.55*Z[:,2].max())
plt.title("Who really competes with whom: the subsea family tree", fontweight="bold")
plt.ylabel("Ward distance (the lower they join, the more alike)")
pos = {name: 5+i*10 for i, name in enumerate(dn["ivl"])}
xmid = (pos["Saipem"]+pos["Subsea7"])/2
plt.annotate("Subsea7 + Saipem: the Saipem7 merger, in numbers",
             xy=(xmid, 0.4), xytext=(xmid, Z[:,2].max()*0.5),
             ha="center", fontsize=9, color="#1a5c7a",
             arrowprops=dict(arrowstyle="->", color="#1a5c7a"))
plt.tight_layout(); plt.savefig("images/03_dendrogram_ward.png", dpi=150, bbox_inches="tight"); plt.close()

# leave-one-feature-out
labels_base = df.cluster.values
lofo = {}
for fr in features:
    fs = [f for f in features if f != fr]
    Xs = StandardScaler().fit_transform(df[fs].values)
    lofo[fr] = adjusted_rand_score(labels_base, KMeans(n_clusters=K, n_init=50, random_state=SEED).fit_predict(Xs))

# monte carlo (seeded, fully reproducible)
rng = np.random.default_rng(SEED); N=300; est = np.zeros(len(df))
for it in range(N):
    dp = df.copy()
    for c in ["pct_subsea","fleet"]:
        dp[c] = dp[c]*rng.uniform(.8,1.2,len(dp))
    dp["subsea_revenue_est"] = dp["revenue_usdbn"]*dp["pct_subsea"]/100
    Xp = StandardScaler().fit_transform(dp[features].values)
    lab = KMeans(n_clusters=K, n_init=10, random_state=it).fit_predict(Xp)
    for i in range(len(df)):
        same_before = set(np.where(labels_base==labels_base[i])[0])-{i}
        same_after = set(np.where(lab==lab[i])[0])-{i}
        est[i] += len(same_before&same_after)/max(len(same_before|same_after),1) if (same_before or same_after) else 1
df["stability_pct"] = (est/N*100).round(0)

# stability chart (reproducible; used on the study page)
ss = df.sort_values("stability_pct")
plt.figure(figsize=(11,7))
plt.barh(ss.company, ss.stability_pct, color=[colors[c] for c in ss.cluster])
for y,(name,val) in enumerate(zip(ss.company, ss.stability_pct)):
    plt.text(val+1, y, f"{val:.0f}%", va="center", fontsize=10)
plt.axvline(70, ls="--", color="gray")
plt.text(69, len(ss)-0.5, "confidence threshold (70%)", rotation=90, va="top", ha="right",
         fontsize=8, color="gray")
plt.title("Stress test: does each company's cluster survive ±20% data error?\n"
          "colors = strategic group | perturbation applied to subsea share & fleet estimates",
          fontsize=12, fontweight="bold")
plt.xlabel("% of 300 Monte Carlo runs in which the company stayed in the same group")
plt.xlim(0, 112)
plt.tight_layout(); plt.savefig("images/en_03_stability.png", dpi=150, bbox_inches="tight"); plt.close()

# distance decomposition: for each isolated company, split the squared distance
# to its nearest neighbour by feature -> names the variable driving the isolation.
def decompose_isolation(company):
    i = df.index[df.company==company][0]
    d = np.sqrt(((X - X[i])**2).sum(axis=1)); d[i] = np.inf
    j = int(d.argmin())
    contrib = (X[i]-X[j])**2
    share = contrib/contrib.sum()
    top = sorted(zip(features, share), key=lambda t: -t[1])[:3]
    return df.company[j], top

# gower
cont = ["subsea_revenue_est","pct_subsea","n_segments","fleet","brazil"]
binv = ["sps","surf_inst","cables","services"]
n = len(df); D = np.zeros((n,n)); rng_amp = {c:(df[c].max()-df[c].min()) or 1 for c in cont}
for i in range(n):
    for j in range(i+1,n):
        dc = [abs(df[c].iloc[i]-df[c].iloc[j])/rng_amp[c] for c in cont]
        dbin = [int(df[b].iloc[i]!=df[b].iloc[j]) for b in binv]
        D[i,j] = D[j,i] = np.mean(dc+dbin)
Zg = linkage(squareform(D), method="average")
plt.figure(figsize=(12,5))
dendrogram(Zg, labels=df.company.values, leaf_rotation=75, leaf_font_size=10)
plt.title("Dendrogram with Gower distance (method cross-validation)")
plt.ylabel("Gower distance")
plt.tight_layout(); plt.savefig("images/04_dendrogram_gower.png", dpi=150, bbox_inches="tight"); plt.close()

df.drop(columns=["pc1","pc2"]).to_csv("data/top15_subsea_clusters.csv", index=False)

# report
print(f"k chosen: {K} | silhouette: {max(sil):.3f}")
for c in sorted(df.cluster.unique()):
    print(f"Cluster {c}: " + ", ".join(df[df.cluster==c].company))
print(f"LOFO mean ARI: {np.mean(list(lofo.values())):.2f} (min {min(lofo.values()):.2f} dropping '{min(lofo,key=lofo.get)}')")
print(f"Monte Carlo mean stability: {df.stability_pct.mean():.1f}%")
print("Most stable:", ", ".join(df.sort_values('stability_pct',ascending=False).head(5).company))
print("Borderline:", ", ".join(df.sort_values('stability_pct').head(3).company))
for singleton in df.groupby("cluster").filter(lambda g: len(g)==1).company:
    neigh, top = decompose_isolation(singleton)
    parts = ", ".join(f"{f} {s:.0%}" for f, s in top)
    print(f"Isolation of {singleton} (nearest: {neigh}): {parts}")
