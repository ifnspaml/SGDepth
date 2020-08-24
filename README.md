# [Self-Supervised Monocular Depth Estimation: Solving the Dynamic Object Problem by Semantic Guidance](https://arxiv.org/abs/2008.01484)

[Marvin Klingner](https://www.tu-braunschweig.de/en/ifn/institute/team/sv/klingner), [Jan-Aike Termöhlen](https://www.tu-braunschweig.de/en/ifn/institute/team/sv/termoehlen), Jonas Mikolajczyk, and [Tim Fingscheidt](https://www.tu-braunschweig.de/en/ifn/institute/team/sv/fingscheidt) – ECCV 2020


[Link to paper](https://arxiv.org/abs/2007.06936)  


**Code will be published soon....**

## Idea Behind the Method

Self-supervised monocular depth estimation usually relies on the assumption of a static world during training which is violated by dynamci objects. 
In our paper we introduce a multi-task learning framework that semantically guides the self-supervised depth estimation to handle such objects.

<p align="center">
  <img src="imgs/intro.png" width="600" />
</p>

## Improved Depth Estimation Results

As a consequence of the multi-task training, dynamic objects are more clearly shaped and small objects such as traffic signs or traffic lights are better recognised in comparison to previous methods.

<p align="center">
  <img src="imgs/qualitative.png" width="600" />
</p>

## Citation

If you find our work useful or interesting, please consider citing [our paper](https://arxiv.org/abs/2007.06936):

```
@inproceedings{klingner2020selfsupervised,
 title   = {{Self-Supervised Monocular Depth Estimation: Solving the Dynamic Object Problem by Semantic Guidance}},
 author  = {Marvin Klingner and
            Jan-Aike Term\"{o}hlen and
            Jonas Mikolajczyk and
            Tim Fingscheidt
           },
 booktitle = {{European Conference on Computer Vision ({ECCV})}},
 year = {2020}
}
```
