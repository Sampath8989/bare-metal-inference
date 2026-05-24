import numpy as np


w1 =np.full((128,64),0.01,dtype=np.float32)
b1=np.zeros(64,dtype=np.float32)

w2 =np.full((64,32),0.01,dtype=np.float32)
b2=np.zeros(32,dtype=np.float32)


w3=np.full((32,10),0.01,dtype=np.float32)
b3=np.zeros(10,dtype=np.float32)

x=np.ones((1,128),dtype=np.float32)

o1=np.maximum(0,x @ w1 +b1)
o2=np.maximum(0,o1 @ w2 +b2)
o3=np.maximum(0,o2 @ w3 + b3)


# ADD these at the bottom of golden.py
print("=== GOLDEN REFERENCE OUTPUT ===")
for i, v in enumerate(o3[0]):
    print(f"  golden[{i}] = {v:.6f}")

np.savetxt("golden_output.txt", o3, fmt="%.6f")
print("\nSaved to golden_output.txt")