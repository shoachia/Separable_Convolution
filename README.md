# 1.Saperable Convolution
Accroding to that the convolution is associative, the convolution opration can be improved if the kernel is separable.

![](https://i.imgur.com/e8tOwlA.png)
can be rewritten as below:

![](https://i.imgur.com/8IJInKC.png)

Therefore if the kernel can be divided into two smaller kernel, we can convolve the image in two steps.

**Computational advantage of separable convolution**

Separable convolution can reduce the computation time. For example, filter an image of MxN with a kernel of PxQ, the computation times will be **MNPQ multiplies and additions**. In contrast, if we apply separable kernel, the computation times in the **first steps will be MNP. MNQ in the second step.** Therefore, the overall computation times will become **MN(P+Q)**.

So, the computation time will be reduce ![](https://i.imgur.com/4EnpDTJ.png) times.

For a 9-by-9 filter kernel, that's a theoretical speed-up of **4.5**.

---

Another separable rule is by using the distributive law in convolution operation.
![](https://i.imgur.com/Qkq8Dl0.png)

can be rewritten as below:
![](https://i.imgur.com/9mV8WUJ.png)

Different from the product separation operation, we should find separated kernels which follows ![](https://i.imgur.com/ZHBfNOh.png). 

For example, a Laplacian kernel can be separated by this law.

![](https://i.imgur.com/6CdH2oT.png)

can be separate into as following:

![](https://i.imgur.com/z9KRXzT.png)

---
# 2. Derivative Filter
Here we show some properties relate to derivative filter.
#### 1. Relation between forward gradient and backward gradient
![](https://i.imgur.com/QjyA0Tj.png)
```python=
b = ['periodical','extension','zero-padding','mirror']
for i in range(len(b)):
    first = np.sum(np.multiply(x,convolve_sep(y,kernel('grad1_backward'), boundary = b[i], separable = None)))
    second = -np.sum(np.multiply(convolve_sep(x, kernel('grad1_forward'), boundary = b[i], separable = None),y))
    print(b[i] + "  :  " + str(np.isclose(first,second)))
```
**Output:**
periodical  :  True
extension  :  False
zero-padding  :  True
mirror  :  False

***Proof:***

![](https://i.imgur.com/IZXUjLJ.png)
**Zero-padding can also be True**

![](https://i.imgur.com/GFmact1.png)

#### 2. ![](https://i.imgur.com/KnvMWMn.png)

***Proof:***

![](https://i.imgur.com/vP9f6B3.png)

#### 3. ![](https://i.imgur.com/eJct2Gm.png)

Laplacian kernel in the first direction is the convolution of the grad1_forward kernel and grad1_backward. Besides, only the periodical boundary can hold this equation. In the other boundary conditions, we have lost some information after doing grad1_forward. Therefore, when doing the grad1_backward we are not using the all the image information which is different from Laplacian.
```python=
for i in range(len(b)):
    first = convolve_sep(y,kernel('grad1_forward'), boundary = b[i], separable = None)
    first = convolve_sep(first,kernel('grad1_backward'), boundary = b[i], separable = None)
    second = convolve_sep(y,kernel('laplacian1'), boundary = b[i], separable = None)kp
    print(b[i] + "  :  " + str(np.allclose(first,second)))
```
**Output**

periodical  :  True
extension  :  False
zero-padding  :  False
mirror  :  False

***Proof:***
![](https://i.imgur.com/Ik7fWIU.png)

#### 4. Divergence
![](https://i.imgur.com/WNqXGeY.png)

**Implementation**
```python=
def div(f, boundary='periodical'):
    # f is a nxnx2x3 array for RGB image
    d = convolve_sep(f[:,:,0],kernel('grad1_backward')) + convolve_sep(f[:,:,1],kernel('grad2_backward'))
    return d
```















