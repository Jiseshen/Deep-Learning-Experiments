from main import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

model.eval()
cp_embed = None
for name, i in model.named_parameters():
    if "cp_embed" in name:
        cp_embed = i.data


tsne = TSNE(perplexity=50)
X_proj = tsne.fit_transform(cp_embed.numpy())

state_label = [state[cp] for cp in cp_map]
data = pd.DataFrame({"x": X_proj[:, 0], "y": X_proj[:, 1], "label": state_label})
sns.scatterplot(data, x="x", y="y", hue="label", legend='full')
plt.show()

party_label = [party[cp] for cp in cp_map]
data = pd.DataFrame({"x": X_proj[:, 0], "y": X_proj[:, 1], "label": party_label})
sns.scatterplot(data, x="x", y="y", hue="label", legend='full')
plt.show()

