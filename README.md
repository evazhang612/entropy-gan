# entropy-gan

CS 236 Project.

1. Download dataset from cohn-kanade+. 

2. Face Alignment (Square Cropping/Greyscale RGB conversion included in transforms) 

```python preprocessing/face_alignment.py ```

3. Run DCGAN Baseline 

``` python mod_cifar_dcgan.py --cuda --epochs=150```

TODO (evazhang612): Add more layers. 
