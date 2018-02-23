
# Optimizer


## Model & Optimizer Graphs

![](./assets/models.png)

## Simple Experiments(old)

Test such as following models.

```python
Level 1:tensorflow:Created variable dense/kernel:0 with shape (784, 540)
Level 1:tensorflow:Created variable dense_1/kernel:0 with shape (540, 300)
Level 1:tensorflow:Created variable dense_1/bias:0 with shape (300,)
Level 1:tensorflow:Created variable dense_2/kernel:0 with shape (300, 10)
Level 1:tensorflow:Created variable dense_2/bias:0 with shape (10,)
```

The below describe is outputs of `backpropagation`.

```python
0 (128, 300) (300, 10)
1 (128, 540) (540, 300)
2 (128, 784) (784, 540)
updates [<tf.Tensor 'Assign:0' shape=(300, 10) dtype=float32_ref>, <tf.Tensor 'Assign_1:0' shape=(540, 300) dtype=float32_ref>, <tf.Tensor 'Assign_2:0' shape=(784, 540) dtype=float32_ref>]
SemiNMF.loss = 7.382757030427456e-05
SemiNMF.loss = 2.2847769287182018e-05
SemiNMF.loss = 4.647569585358724e-05
SemiNMF.loss = 1.235315176018048e-05
SemiNMF.loss = 6.353859589580679e-06
SemiNMF.loss = 3.3170092592627043e-06
SemiNMF.loss = 0.0007002330385148525
SemiNMF.loss = 3.8916336961847264e-06
SemiNMF.loss = 7.233204087242484e-05
SemiNMF.loss = 1.0133316209248733e-05
BiasNSNMF.loss = 1.1047664880752563
BiasNSNMF.loss = 1.1882364749908447
BiasNSNMF.loss = 1.2767796516418457
BiasNSNMF.loss = 1.3705438375473022
BiasNSNMF.loss = 1.4590647220611572
BiasNSNMF.loss = 1.541094183921814
BiasNSNMF.loss = 1.6265606880187988
BiasNSNMF.loss = 1.701664924621582
BiasNSNMF.loss = 1.777556300163269
BiasNSNMF.loss = 1.8421251773834229
NonlinearSNMF.loss = 0.82808518409729
NonlinearSNMF.loss = 0.7898814678192139
NonlinearSNMF.loss = 0.761573076248169
NonlinearSNMF.loss = 0.745311439037323
NonlinearSNMF.loss = 0.734568178653717
NonlinearSNMF.loss = 0.7275324463844299
NonlinearSNMF.loss = 0.7228933572769165
NonlinearSNMF.loss = 0.7197899222373962
NonlinearSNMF.loss = 0.7177244424819946
NonlinearSNMF.loss = 0.7163248062133789
```
