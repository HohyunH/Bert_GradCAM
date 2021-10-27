# Bert_GradCAM
- Bert를 이용한 준지도학습 모델에 GradCAM을 추가한 실험입니다.

### How to use
<pre>
<code>
python main.py --max_len 512 --epochs 1 --batch 16 --label_batch 200 --masked True --xai True --test_num 10000 --class_num 100 --gpu_set choose[1,2,3,4]
</code>
</pre>

### Model Framework

![image](https://user-images.githubusercontent.com/46701548/139094004-266b0ed0-4ab6-49e9-a089-4e9069707b55.png)


- 아래 Text Self training with XAI 실험에서 Bert+CNN 과 GradCAM을 활용한 코드 입니다.


![image](https://user-images.githubusercontent.com/46701548/139091521-3bf1c868-b5a0-4671-879e-a9a23a3f1fbf.png)
