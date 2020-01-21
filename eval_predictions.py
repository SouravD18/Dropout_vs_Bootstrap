import os
import pickle
import numpy as np

fnames = os.listdir("./predictions")

result = []
hs_acc = []
hs_loss = []
for i in fnames:
    prediction = pickle.load(open("./predictions/" + i, "rb"))
    result.append(prediction)
    history = pickle.load(open("./history/" + i, "rb"))
    hs_acc.append(history["val_acc"][-1])
    hs_loss.append(history["val_loss"][-1])

print("AVG_last_acc", np.average(hs_acc))
print("AVG_last_loss", np.average(hs_loss))

"""
import matplotlib.pyplot as plt

plt.hist(x=results, 
        bins=[i for i in range(0, 101)])
plt.ylabel("Frequency")
plt.show()
"""
predictions = np.array(result) 
pickle.dump(predictions, open("bootstrap_predictions", "wb"))
