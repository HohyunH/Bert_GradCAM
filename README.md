# Bert_GradCAM
- Bert를 이용한 준지도학습 모델에 GradCAM을 추가한 실험입니다.

### How to use
<pre>
<code>
python main.py --max_len 512 --epochs 1 --batch 16 --label_batch 200 --masked True --xai True --test_num 10000 --class_num 100 --gpu_set choose[1,2,3,4]
</code>
</pre>

### Requirement

- python >= 3.8
- pandas
- numpy
- pytorch
- transformers
- seaborn
- nltk
- cv2

### Command

- pip install pandas
- pip install numpy
- pip install seaborn
- pip intsall nltk
- pip intsall transformers
- pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

### Model Framework

![image](https://user-images.githubusercontent.com/46701548/139094004-266b0ed0-4ab6-49e9-a089-4e9069707b55.png)

- BERT 이후에 붙혀진 CNN 에서 마지막 Convolution layer에서 Grad-CAM score를 산출한다.

- 인공 신경망 연산 과정에서 계산되는 gradients 값과 activation 값의 연산으로 Grad-CAM score를 산출할 수 있다.
```python
gradients = grad_cam.get_activations_gradient()    
activations = grad_cam.get_activations(sentence, masks).detach()

# global average pooling : 각 채널별로 평균 구함.
pooled_gradients = torch.mean(gradients, dim=[0, 2])
for k in range(gradients.shape[1]):
    activations[:, k, :] *= pooled_gradients[k]

heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = heatmap.view(1, -1).cpu().numpy()
heatmap = cv2.resize(heatmap, dsize=(512, 1))
heatmap = np.multiply(heatmap, masks.cpu().numpy())
```

### 문장내 단어 중요도 추출 예시

![image](https://user-images.githubusercontent.com/46701548/140642315-8d850b29-9b09-4643-9bba-b0b16278e3d1.png)


- 아래 Text Self training with XAI 실험에서 Bert+CNN 과 GradCAM을 활용한 코드 입니다.


![image](https://user-images.githubusercontent.com/46701548/139091521-3bf1c868-b5a0-4671-879e-a9a23a3f1fbf.png)
