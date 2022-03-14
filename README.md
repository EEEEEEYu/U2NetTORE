# TODO

- 改小输出channel number，增大batch size (√)
- 设计并改动让每个GPU单独work
    - batch 如何分配到GPU上
    - 如何让每个GPU上的内容分别work
    - 单个x改为16个samples，单个label改为16个results
        - loss 方面的处理
- Precision set to float16并debug (√)
- argparse bool
- 当前log folder位置以及正确配置profiling
- 当前没有考虑到不同folder下文件的取整等问题
- 文件总数使用meta中的描述
- ntore改成新的reader
- 

tic=time.time()
gen_tore_plus(n1)
print(time.time()-tic)
