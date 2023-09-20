# TwinTex: Geometry-aware Texture Generation for Abstracted 3D Architectural Models

Implementation for our paper "TwinTex: Geometry-aware Texture Generation for Abstracted 3D Architectural Models" (ACM SIGGRAPH Asia 2023).

## SOP_TwinTex 

In `C:\Users\xxx\Documents\houdini19.5\dso` directory：

```
----
 ├──SOP_TwinTex.dll
 ├──libgmp-10.dll
 ├──opencv_world460.dll
```

And then you can add `TwinTex` in Geometry node.

![image-20230906162939471](assets\image-20230906162939471.png)

### HOW  TO USE

1. First specify the configuration file `default_config.yaml` and the path to the scene directory, which has the following structure:

   ```
   ----
    ├──Mesh
       ├──original_mesh.ply
       ├──simplify_mesh.ply
       ├──simplify_mesh_sub.ply
    ├──Scene
    ├──Results(auto）
   ```

2. Press the `Load Data` button to load scene data

3. (Optional) `Load Results` : Load the results of `Result\finale_result`.

4. (Optional) `Show Views`: show the position and direction of the view (only works on a single plane mode). The single plane  mode and id is specified in the configuration file。

5. (Optional) `Show Image`: show the output of ViewSelection or ImageStitching (only works on a single plane).

6. `ViewSelection`: Parameters of ViewSelection and apply buttuon

7. `ImageStitching`：Parameters of ImageStitchingand and apply buttuon

8. `Inpainting`:

   - `MMRP Path`： Specify the path of the MMRP project, see `MMRP\README.md` for dependency details
   - `Inpainting`:   Inpaint with plane to be inpainted

9. `start`: will run ViewSelection and ImageStitching

If houdini crashes, please first check that you have followed the environment, configuration and steps above.

## Citation

If you find this useful for your research, please cite the following paper.

```
@article{TwinTex23,
title={TwinTex: Geometry-aware Texture Generation for Abstracted 3D Architectural Models},
author={Weidan Xiong and Hongqian Zhang and Botao Peng and Ziyu Hu and Yongli Wu and Jianwei Guo and Hui Huang},
journal={ACM Transactions on Graphics (Proceedings of SIGGRAPH ASIA)},
volume={42},
number={6},
pages={},
year={2023},
}
```

**Copyright©2023. All rights reserved.**