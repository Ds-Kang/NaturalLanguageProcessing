import torch

scores = torch.Tensor(
  [[ 0,  0.1261572,   1.1627575 ],
 [ 0, -0.0675799,  -0.21310908],
 [0,  -0.12525336, -0.06508598],
 [ 2,  0.07172455,  0.2353122 ],
 [ 0.8219606,  -0.32560882, -0.77807254]]
)

b=torch.sum(scores,0)           
print(b)