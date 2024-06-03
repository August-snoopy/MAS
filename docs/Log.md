205-432



### SiMLPe 

Epoch [900/1000], Train Loss: 0.0033, Val Loss: 0.0062
Epoch [910/1000], Train Loss: 0.0031, Val Loss: 0.0069
Epoch [920/1000], Train Loss: 0.0030, Val Loss: 0.0072
Epoch [930/1000], Train Loss: 0.0030, Val Loss: 0.0064
Epoch [940/1000], Train Loss: 0.0031, Val Loss: 0.0065
Epoch [950/1000], Train Loss: 0.0027, Val Loss: 0.0070
Epoch [960/1000], Train Loss: 0.0033, Val Loss: 0.0078
Epoch [970/1000], Train Loss: 0.0029, Val Loss: 0.0067
Epoch [980/1000], Train Loss: 0.0029, Val Loss: 0.0064
Epoch [990/1000], Train Loss: 0.0029, Val Loss: 0.0068
Epoch [1000/1000], Train Loss: 0.0030, Val Loss: 0.0071



### 单头注意力

在时间模块输出后加入Attention

```python
class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        attn_scores = torch.matmul(x, self.attention_weights)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = x * attn_weights.unsqueeze(-1)
        return attn_output
```

Epoch [580/1000], Train Loss: 3.3452, Val Loss: 7.8022
Epoch [590/1000], Train Loss: 3.3375, Val Loss: 7.2203
Epoch [600/1000], Train Loss: 3.2401, Val Loss: 7.0001
Epoch [610/1000], Train Loss: 3.1947, Val Loss: 6.3934
Epoch [620/1000], Train Loss: 35.6672, Val Loss: 99.4643
Epoch [630/1000], Train Loss: 4.2530, Val Loss: 19.9501
Epoch [640/1000], Train Loss: 3.9611, Val Loss: 77.5354

Epoch [900/1000], Train Loss: 2.4710, Val Loss: 6.0004
Epoch [910/1000], Train Loss: 2.5747, Val Loss: 5.6928
Epoch [920/1000], Train Loss: 2.4507, Val Loss: 5.9154
Epoch [930/1000], Train Loss: 2.4671, Val Loss: 4.8787
Epoch [940/1000], Train Loss: 2.5110, Val Loss: 6.0536
Epoch [950/1000], Train Loss: 2.4556, Val Loss: 6.1230
Epoch [960/1000], Train Loss: 2.4078, Val Loss: 5.4827
Epoch [970/1000], Train Loss: 2.4868, Val Loss: 5.3424
Epoch [980/1000], Train Loss: 2.3760, Val Loss: 5.6534
Epoch [990/1000], Train Loss: 2.3820, Val Loss: 5.0942
Epoch [1000/1000], Train Loss: 2.3699, Val Loss: 5.6842

可能中间出现梯度爆炸，但是后续能够拟合回来，最优能达到4.8

Epoch [850/1000], Train Loss: 2.3387, Val Loss: 4.1331
Epoch [860/1000], Train Loss: 91.6815, Val Loss: 37.9666
Epoch [870/1000], Train Loss: 3.2647, Val Loss: 6.4627
Epoch [880/1000], Train Loss: 2.9856, Val Loss: 6.0627
Epoch [890/1000], Train Loss: 2.7626, Val Loss: 5.4623
Epoch [900/1000], Train Loss: 2.7396, Val Loss: 5.1027
Epoch [910/1000], Train Loss: 2.6823, Val Loss: 5.2599
Epoch [920/1000], Train Loss: 2.5173, Val Loss: 4.7139
Epoch [930/1000], Train Loss: 2.6107, Val Loss: 4.4314
Epoch [940/1000], Train Loss: 2.5886, Val Loss: 5.0669
Epoch [950/1000], Train Loss: 2.6134, Val Loss: 5.0510
Epoch [960/1000], Train Loss: 2.5227, Val Loss: 4.9447
Epoch [970/1000], Train Loss: 2.5312, Val Loss: 4.5294
Epoch [980/1000], Train Loss: 2.3989, Val Loss: 4.5055
Epoch [990/1000], Train Loss: 2.3977, Val Loss: 4.5865
Epoch [1000/1000], Train Loss: 2.3861, Val Loss: 4.6209





### 前后都加入Attention

这让我思考不清楚，按理说在前后都加入注意力后，会更加注重一些不同的值。

Epoch [900/1000], Train Loss: 15.9480, Val Loss: 21.8337
Epoch [910/1000], Train Loss: 15.6131, Val Loss: 21.1792
Epoch [920/1000], Train Loss: 15.2285, Val Loss: 21.6246
Epoch [930/1000], Train Loss: 15.1998, Val Loss: 20.9608
Epoch [940/1000], Train Loss: 15.0074, Val Loss: 20.2196
Epoch [950/1000], Train Loss: 14.8287, Val Loss: 20.5558
Epoch [960/1000], Train Loss: 14.4982, Val Loss: 19.9119
Epoch [970/1000], Train Loss: 14.9477, Val Loss: 20.1238
Epoch [980/1000], Train Loss: 14.7683, Val Loss: 20.5007
Epoch [990/1000], Train Loss: 14.5035, Val Loss: 19.8885
Epoch [1000/1000], Train Loss: 14.1984, Val Loss: 19.3536

最优只能达到19





### 使用Scaled Dot-Product Attention

最优结果是15，可能还在收敛，这个收敛速度可能比较慢。

Epoch [900/1000], Train Loss: 9.8561, Val Loss: 18.8512
Epoch [910/1000], Train Loss: 9.5414, Val Loss: 18.6046
Epoch [920/1000], Train Loss: 9.3541, Val Loss: 18.2803
Epoch [930/1000], Train Loss: 9.1778, Val Loss: 18.2102
Epoch [940/1000], Train Loss: 8.9394, Val Loss: 17.8222
Epoch [950/1000], Train Loss: 8.5530, Val Loss: 17.2974
Epoch [960/1000], Train Loss: 8.4883, Val Loss: 17.3361
Epoch [970/1000], Train Loss: 8.1279, Val Loss: 17.0743
Epoch [980/1000], Train Loss: 8.0696, Val Loss: 17.0195
Epoch [990/1000], Train Loss: 7.8101, Val Loss: 15.8506
Epoch [1000/1000], Train Loss: 7.4788, Val Loss: 15.8729





### 多头注意力

当前是3头，感觉如果设置为21个节点后使用7头或者9头效果会更好。有过拟合风险

Epoch [900/1000], Train Loss: 3.2571, Val Loss: 9.4321
Epoch [910/1000], Train Loss: 2.8318, Val Loss: 9.5273
Epoch [920/1000], Train Loss: 3.0488, Val Loss: 9.8587
Epoch [930/1000], Train Loss: 2.8114, Val Loss: 9.4689
Epoch [940/1000], Train Loss: 2.9323, Val Loss: 9.1296
Epoch [950/1000], Train Loss: 2.5851, Val Loss: 9.3283
Epoch [960/1000], Train Loss: 2.5351, Val Loss: 9.3732
Epoch [970/1000], Train Loss: 2.7443, Val Loss: 9.0960
Epoch [980/1000], Train Loss: 2.5717, Val Loss: 9.4339
Epoch [990/1000], Train Loss: 2.4565, Val Loss: 9.7835
Epoch [1000/1000], Train Loss: 2.2369, Val Loss: 8.7176







### 自注意力

自注意力也能达到比较低的误差

Epoch [900/1000], Train Loss: 2.7901, Val Loss: 6.5546
Epoch [910/1000], Train Loss: 2.7334, Val Loss: 5.8115
Epoch [920/1000], Train Loss: 2.6109, Val Loss: 6.7872
Epoch [930/1000], Train Loss: 2.7219, Val Loss: 6.3462
Epoch [940/1000], Train Loss: 2.6897, Val Loss: 5.9213
Epoch [950/1000], Train Loss: 2.7366, Val Loss: 6.6212
Epoch [960/1000], Train Loss: 2.5023, Val Loss: 5.7727
Epoch [970/1000], Train Loss: 2.5352, Val Loss: 6.6949
Epoch [980/1000], Train Loss: 2.3884, Val Loss: 5.6647
Epoch [990/1000], Train Loss: 2.3641, Val Loss: 6.2258
Epoch [1000/1000], Train Loss: 2.2540, Val Loss: 5.8823



使用torch.nn.MultiHeadAttention的自注意力结果

Epoch [900/1000], Train Loss: 2.4240, Val Loss: 6.8947
Epoch [910/1000], Train Loss: 2.4417, Val Loss: 6.0894
Epoch [920/1000], Train Loss: 2.3337, Val Loss: 7.5659
Epoch [930/1000], Train Loss: 2.2422, Val Loss: 6.8796
Epoch [940/1000], Train Loss: 2.2294, Val Loss: 7.0211
Epoch [950/1000], Train Loss: 2.2358, Val Loss: 6.6488
Epoch [960/1000], Train Loss: 2.0031, Val Loss: 6.8727
Epoch [970/1000], Train Loss: 2.0351, Val Loss: 7.5569
Epoch [980/1000], Train Loss: 1.9847, Val Loss: 6.5539
Epoch [990/1000], Train Loss: 2.1519, Val Loss: 6.8290
Epoch [1000/1000], Train Loss: 1.9059, Val Loss: 6.4271

很有可能过拟合