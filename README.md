# entropy-gan

CS 236 Project.

1. Download dataset from cohn-kanade+. 

2. Face Alignment (Square Cropping/Greyscale RGB conversion included in transforms) 

```python preprocessing/face_alignment.py ```

3. Run DCGAN Baseline 

``` python conditional_gan.py``` 

or Run new version of Baseline (rewritten with better comments on dim.)

``` python cdgan_new.py```

TODO (evazhang612): patch batch size issues. 
