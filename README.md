# 并行计算gauss_seidel算子加速报告
## 问题介绍和分析

本次作业使用cuda加速gauss_seidel算子和residue norm计算。

- gauss_seidel算子伪码如下：


```python
def gauss_seidel(u, b)
    for x in 512:
        for y in 512:
            for z in 512:
                u[x][y][z] = b[x][y][z]
                for dir in directions:
                    u[x][y][z] += u[x+dir.x][y+dir.y][z+dir.z]
                u[x][y][z] /= 6.0
```

每个u[x][y][z]被访问6次，可以将数据暂存到shared memory来利用temporal locality。

如果直接按gauss_seidel算子进行计算，那么所有计算只有唯一的计算顺序。为释放并行度，考虑在x,y,z维度通过tiling的方式做并行。



- residue_norm算子的伪码如下
```python
def residue_norm(u, b)
    norm = 0
    for x in 512:
        for y in 512:
            for z in 512:
                res = b[x][y][z]
                for dir in directions:
                    res += u[x+dir.x][y+dir.y][z+dir.z]
                norm += (u[x][y][z] * 6.0 - res)^2;
```
residue_norm的循环体中的res恰好是psedo-code中的u[x][y][z]的新值，利用这一点可以将residue_norm的计算与gauss_seidel算子整合(fusion)到一起，得到一个residue cube，然后对这个residue cube用标准的方式做sum reduction。

## 加速方案
对gauss seidel kernel的加速使用了Tiling + red-black的做法, 对residue_norm的加速使用了fusion + sum reduction的做法, 下面进行详细叙述。
### gauss_seidel_kernel
对x,y,z三个维度分别成4级tile, 由外至内分别对应blockIdx, regIdx, shareMemoryIdx, threadIdx。将最外层的三层循环绑定到blockDim上，最内层的三层循环绑定到threadIdx上, 在中间的两层循环分别把数据加载到register和shared memory中
```python
def gauss_seidel:
    reg[REG_DIM_X][REG_DIM_Y][REG_DIM_Z]
    __shared__ u_shared[SHARE_DIM_X][SHARE_DIM_Y][SHARE_DIM_Z] 
    for x_o,y_o,z_o in GRID_DIM_X X GRID_DIM_Y X GRID_DIM_Z:
        for x_reg, y_reg, z_reg in REG_DIM_X X REG_DIM_Y X REG_DIM_Z:
            load u from global memory to register.
            for x_share, y_share, z_share in SHARE_DIM_X X:
                load reg to shared
                for x_in, y_in, z_in in THREAD_DIM_X X THREAD_DIM_Y X THREAD_DIM_Z:
                    Calculation, store result in regs.
            store u from reg to global memory.
```

同时，为了加快收敛速度，使用red-black方式进行迭代更新, 即先对下标x,y,z中有奇数个奇数的点(红块)进行更新，再更新具有偶数个的(黑块)。同时，尽量充分的利用register和sharedMemory的空间。

使用red-black方法进行的calculation流程如下:
1. 初始时：红块和黑块都在register中
2. 将黑块加载到shared memory中
3. 计算红块，将计算结果保存在register中
4. 将红块的计算结果存储到shared memory中
5. 计算黑块，将计算结果保存在register中
6. 将红块、黑块的计算结果从register存到global memory中。
上述算法中，在register中为红块和黑块开辟独立的空间，而shared memory同一时刻只保留红块数据或黑块数据。这样比只用shared memory来存数据时shared memory的利用率有提升。


### residue norm
residue norm 的计算分为求residue cube和reduction两个阶段，求residue cube阶段与gauss_seidel的计算合并，在计算红块和黑块时计算残差, 即:
1. 初始时：红块和黑块都在register中
2. 将黑块加载到shared memory中
3.0 计算红块，`利用计算结果和register中的旧红块数据计算residue，保存到global memory中`
3.1 将计算结果保存在register中
4. 将红块的计算结果存储到shared memory中
5.0 计算黑块，`利用计算结果和register中的旧黑块数据计算residue，保存到global memory中`
5.1 将计算结果保存在register中
6. 将红块、黑块的计算结果从register存到global memory中。

对reduction的计算参考了nvidia的[官方教程](https://developer.download.nvidia.cn/assets/cuda/files/reduction.pdf), 通过下标变换使得bank conflict减少，通过unrolling减少instruction overhead, 通过warp内kernel减少syncthreads()的调用。具体见代码中residue和warpReduce函数。

## 加速结果
0.90~0.95s, 37 iterations
```
Mid Tiling: 128 128 128.
Inner Tiling(block): 32 8 8
grid  dim:  4, 4, 4.
block dim: 16, 8, 8.
iter 0, res 10528.038449
iter 1, res 7980.810127
iter 2, res 4059.382029
iter 3, res 2029.126616
iter 4, res 1013.927747
iter 5, res 508.209311
iter 6, res 256.318544
iter 7, res 130.742770
iter 8, res 67.942088
iter 9, res 36.294953
iter 10, res 20.099297
iter 11, res 11.588409
iter 12, res 6.943500
iter 13, res 4.294897
iter 14, res 2.720788
iter 15, res 1.754192
iter 16, res 1.147164
iter 17, res 0.760536
iter 18, res 0.512117
iter 19, res 0.351454
iter 20, res 0.246775
iter 21, res 0.177803
iter 22, res 0.131558
iter 23, res 0.099790
iter 24, res 0.077316
iter 25, res 0.060926
iter 26, res 0.048634
iter 27, res 0.039208
iter 28, res 0.031855
iter 29, res 0.026055
iter 30, res 0.021447
iter 31, res 0.017772
iter 32, res 0.014835
iter 33, res 0.012489
iter 34, res 0.010615
iter 35, res 0.009120
iter 36, res 0.007928
iter 37, res 0.006977
norm0 7572.464802, normr 0.006977
elapsed time 0.90434, normr/norm0 0.000001, time/iter 24.441713
Iteration 38, normr/normr0=5.65124e-07
Converged with 38 iterations.
time: 2.26004
Residual norm: 0.00427938
total bandwidth: 212.082 GB/s
relative error: 1.16733
openmp max num threads: 1
```
## 性能影响程度分析
下表是去除某个优化技巧后的时间

|优化技巧|收敛循环数|时间(s)|时间(ms)/Iter|速度降低比例|
|-|-|-|-|-|
|default|37|0.91|24|1|
|no shared memory|51|1.36|27|1.36|
|no share layer|54|1.1|20|1.2|
|no register layer|37|0.95|25|1.05|
|tiling + gauss_seidel|270|4.2|15|4.6|
|no fusion|37|1.2|32|1.3|

这里的share layer, register layer指做tiling时是否生成对应的层, 可以看到, share layer引入降低了迭代次数，但增加了单次迭代时间，前者因为引入sharelayer使得一个block的计算量增加，由于block随机调度而引起的迭代依赖打破减少，而单次迭代时间增加的原因可能是每个block块变大后要`syncthreads()`产生的延迟不能很好被block调度掩盖。

对其他的优化技巧red-black,shared memory降低了迭代次数。fusion技巧降低了单次迭代的时间。

## 结果复现
1. 加载必要库文件
```bash
module load cuda
module load gcc
```
2. compile 
```bash
make all
```
3. Run 
```bash
sbatch ./gpu_run.slurm
```