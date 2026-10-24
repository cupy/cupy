
# build CANN extra ops packages
```bash
# aussuming all ops repos has been clone
# git clone -b ${tag_version} https://gitcode.com/cann/ops-sparse.git
cd ../repos/
cd ops-sparse/

git pull oriign master

# check and make sure 
echo $ASCEND_HOME_PATH

# ops-sparse has a special option:  --ops=spmv
bash build.sh --pkg --soc=ascend950 --ops=coo2csr,coo_get,coosort,cscsort,csr2coo,csr2csc_ex2,csr2gebsr,\
csrgeam2,csrsort,cube_spmm,densetosparse,gather,gebsr2gebsc,gpsv_interleaved_batch,\
gtsv2,gtsv2_nopivot,gtsv2_strided_batch,gtsv_interleaved_batch,nnz,scatter,sddmm,\
spgemm,spmm,spmm_op,spmv,spmv_op,spsm,spsv,spvv

# 默认 = --soc=ascend910b3，全量算子
# -> sparse/sparse2dense/arch22 编不过（op_host 引用 device-only 的 __aicore__），整次构建中断


# if fine, sh file will be generated
./build_out/cann-*-ops-sparse-*.run --install --install-path="$ASCEND_HOME_PATH/.."


[OpsSparse] [2026-10-17 11:27:10] [INFO]: Installing ops_sparse to /home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1/
[OpsSparse] [2026-10-17 11:27:46] [INFO]: ops_sparse installed successfully to /home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1/
[OpsSparse] [2026-10-17 11:27:46] [INFO]: Libraries: /home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1/lib64/ (1 .so files)
[OpsSparse] [2026-10-17 11:27:46] [INFO]: Headers:   /home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1/include/

```
$ASCEND_HOME_PATH=/home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1，
所以是 .../cann-9.0.1/cann/lib64/libops_sparse.so —— 注意不在 $ASCEND_HOME_PATH/lib64，
所以安装 `--install-path="$ASCEND_HOME_PATH/.."`, 这样就可以安装到
/home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1/lib64/ (1 .so files)

这样复用cann toolkit的 lib, include的路径set_env.sh设置


makeself 和gtest需要build同时需要下载, 导致服务器无法访问外网时候, 出现失败, 已经想ops-sparse提出issue, 暂时local做了patch.

This is similar for other `ops-xxx`